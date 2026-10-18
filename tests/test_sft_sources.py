"""Legal SFT loaders, tested on synthetic rows shaped like the real schemas:
CaseHOLD parquet (example_id, citing_prompt, holding_0..4, label) and raw
LawInstruct jsonl (instruction, prompt, answer, *_language, ...)."""

import pytest

from data.sft_prep import SFTPrepConfig, build_sft_split, validate_sft_split
from data.sft_sources import (
    casehold_to_messages, lawinstruct_split_of, lawinstruct_to_messages,
    load_casehold, load_lawinstruct, load_source,
)
from tests.fake_tokenizer import FakeTokenizer


def ch_row(i, label=2):
    return {"example_id": i,
            "citing_prompt": f"Case {i}: the court relied on Smith v. Jones (<HOLDING>).",
            **{f"holding_{k}": f"holding that rule {i}-{k} applies" for k in range(5)},
            "label": label}


def li_rec(i, subset="RedditLegalQA-reddit_legal_qa", lang="en", answer=None):
    return {"dataset_name": subset.split("-")[0], "subset_name": subset,
            "prompt_language": lang, "answer_language": lang,
            "instruction": "Answer the legal question.",
            "prompt": f"Question {i}: can my landlord keep my deposit?",
            "answer": f"Answer {i}: generally no." if answer is None else answer}


# ---------------------------------------------------------------- CaseHOLD
def test_casehold_both_forms_and_answers():
    gen = mc = 0
    for i in range(200):
        msgs = casehold_to_messages(ch_row(i))
        user, asst = msgs[0]["content"], msgs[1]["content"]
        assert "<HOLDING>" in user and msgs[1]["role"] == "assistant"
        assert f"rule {i}-2 applies" in asst            # always the correct holding text
        if asst.startswith("(C) "):
            mc += 1
            assert "(A) holding that" in user and "(E) holding that" in user
        else:
            gen += 1
            assert asst.startswith("Holding that") and asst.endswith(".")
    assert gen > 60 and mc > 60                         # roughly half and half


def test_casehold_deterministic():
    assert casehold_to_messages(ch_row(7)) == casehold_to_messages(ch_row(7))


@pytest.mark.parametrize("bad", [
    {**ch_row(1), "label": 5},
    {**ch_row(1), "label": None},
    {**ch_row(1), "citing_prompt": "no marker here"},
    {**ch_row(1), "holding_2": ""},
])
def test_casehold_unusable_rows_skipped(bad):
    assert casehold_to_messages(bad) is None


def test_casehold_split_mapping_and_report():
    ex, rep = load_casehold("test", rows=[ch_row(i) for i in range(10)] + [{**ch_row(99), "label": 9}])
    assert rep["upstream_split"] == "validation"        # CaseHOLD's own test is never used
    assert rep["read"] == 11 and rep["kept"] == 10 and rep["skipped_unusable"] == 1
    with pytest.raises(ValueError):
        load_casehold("validation", rows=[])


# ---------------------------------------------------------------- LawInstruct
def test_lawinstruct_conversion_and_skips():
    msgs, why = lawinstruct_to_messages(li_rec(1))
    assert why is None
    assert msgs[0]["content"] == "Answer the legal question.\n\nQuestion 1: can my landlord keep my deposit?"
    assert msgs[1]["content"] == "Answer 1: generally no."
    assert lawinstruct_to_messages(li_rec(1, lang="de"))[1] == "not_english"
    assert lawinstruct_to_messages(li_rec(1, answer="  "))[1] == "no_answer"
    assert lawinstruct_to_messages({"answer": "x"})[1] == "no_prompt"


def test_lawinstruct_split_disjoint_dedup_and_cap():
    recs = {"A-a": [li_rec(i, "A-a") for i in range(3000)] + [li_rec(0, "A-a")],  # 1 duplicate
            "B-b": [li_rec(i, "B-b", lang="fr") for i in range(5)]}
    train, rep_tr = load_lawinstruct("train", subsets=["A-a", "B-b"], max_per_subset=None, records=recs)
    test, rep_te = load_lawinstruct("test", subsets=["A-a", "B-b"], max_per_subset=None, records=recs)
    tr = {e["messages"][0]["content"] for e in train}
    te = {e["messages"][0]["content"] for e in test}
    assert not tr & te and len(tr) + len(te) == 3000
    assert 10 <= len(te) <= 60                          # ~1% held out
    assert rep_tr["subsets"]["A-a"]["skipped_duplicate"] == 1
    assert rep_tr["subsets"]["B-b"]["skipped_not_english"] == 5
    capped, rep = load_lawinstruct("train", subsets=["A-a"], max_per_subset=100, records=recs)
    assert len(capped) == 100 and rep["subsets"]["A-a"]["skipped_over_cap"] == len(tr) - 100
    again, _ = load_lawinstruct("train", subsets=["A-a"], max_per_subset=100, records=recs)
    assert capped == again                               # seeded, reproducible


def test_lawinstruct_split_is_stable():
    m = lawinstruct_to_messages(li_rec(5))[0]
    assert lawinstruct_split_of("X-y", m) == lawinstruct_split_of("X-y", m)


def test_default_subsets_exclude_casehold_derived():
    from data.sft_sources import LAWINSTRUCT_SUBSETS
    assert not [s for s in LAWINSTRUCT_SUBSETS if "case_hold" in s or "casehold" in s]


# ---------------------------------------------------------------- end to end
def test_casehold_and_lawinstruct_build_and_validate(tmp_path):
    fp, tok = "fake-fp", FakeTokenizer()
    ex, info = load_source("casehold", "train", rows=[ch_row(i) for i in range(40)])
    cfg = SFTPrepConfig(dataset_root=str(tmp_path), source_name="casehold", split="train",
                        sequence_length=4096, shard_tokens=4096 * 4)
    entry = build_sft_split(cfg, ex, tok, fp, source_info=info)
    assert entry["stats"]["examples_kept"] == 40 and entry["source_info"]["kept"] == 40
    validate_sft_split(tmp_path, "casehold", "train")

    recs = {"A-a": [li_rec(i, "A-a") for i in range(300)]}
    opts = {"subsets": ["A-a"], "max_per_subset": 200}
    ex, info = load_source("lawinstruct", "train", records=recs, **opts)
    cfg = SFTPrepConfig(dataset_root=str(tmp_path), source_name="lawinstruct", split="train",
                        sequence_length=4096, shard_tokens=4096 * 4, loader_options=opts)
    entry = build_sft_split(cfg, ex, tok, fp, source_info=info)
    assert entry["stats"]["examples_kept"] == 200
    assert "academic" in entry["source_info"]["licence"]
    validate_sft_split(tmp_path, "lawinstruct", "train")
