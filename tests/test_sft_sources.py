"""Legal SFT loaders, tested on synthetic rows shaped like the real schemas:
CaseHOLD parquet (example_id, citing_prompt, holding_0..4, label) and raw
LawInstruct jsonl (instruction, prompt, answer, *_language, ...)."""

import pytest

from data.sft_prep import SFTPrepConfig, build_sft_split, validate_sft_split
from data.sft_sources import (
    casehold_to_messages, lawinstruct_split_of, lawinstruct_to_messages,
    load_casehold, load_lawinstruct, load_reddit_domain, load_source,
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


def test_lawinstruct_non_english_instruction_replaced_with_english_one():
    en = {**li_rec(1), "instruction": "Answer like a Reddit comment.", "instruction_language": "en"}
    lt = {**li_rec(2), "instruction": "Atsakykite i si klausima.", "instruction_language": "lt"}
    train, rep = load_lawinstruct("train", subsets=["A-a"], max_per_subset=None,
                                  records={"A-a": [en, lt] * 1 + [{**lt, "prompt": f"Q{i}"} for i in range(50)]})
    users = [e["messages"][0]["content"] for e in train]
    assert users and all(u.startswith("Answer like a Reddit comment.") for u in users)
    assert not any("Atsakykite" in u for u in users)
    assert rep["subsets"]["A-a"]["instruction_replaced"] == 51
    msgs, _ = lawinstruct_to_messages(lt)                 # no English instruction known -> prompt only
    assert msgs[0]["content"] == lt["prompt"]


def test_lawinstruct_embedded_questions_and_empty_reddit_posts():
    brief = {**li_rec(1, "CaseBriefs-case_briefs"), "dataset_name": "CaseBriefs", "prompt": "",
             "answer": "Question: Plaintiff photographed X. Is posting it fair use?\nAnalysis: The court weighed the factors."}
    msgs, _ = lawinstruct_to_messages(brief)
    assert msgs[0]["content"].endswith("Is posting it fair use?") and msgs[1]["content"] == "The court weighed the factors."
    memo = {**li_rec(2, "OLCMemos-olc_memos"), "dataset_name": "OLCMemos", "prompt": "",
            "answer": "Authority of the Customs Service\nThe Service lacks authority.\n\nMEMORANDUM OPINION\nBody."}
    msgs, _ = lawinstruct_to_messages(memo)
    assert msgs[0]["content"].endswith("Authority of the Customs Service The Service lacks authority.")
    assert msgs[1]["content"] == "MEMORANDUM OPINION\nBody."
    empty = {**li_rec(3), "dataset_name": "RedditLegalQA", "prompt": "Question: "}
    assert lawinstruct_to_messages(empty) == (None, "empty_question")


def test_lawinstruct_split_is_stable():
    m = lawinstruct_to_messages(li_rec(5))[0]
    assert lawinstruct_split_of("X-y", m) == lawinstruct_split_of("X-y", m)


def test_default_subsets_exclude_casehold_derived():
    from data.sft_sources import LAWINSTRUCT_SUBSETS
    assert not [s for s in LAWINSTRUCT_SUBSETS if "case_hold" in s or "casehold" in s]


def test_default_subsets_drop_validwills_and_cabar():
    from data.sft_sources import LAWINSTRUCT_SUBSETS
    assert not [s for s in LAWINSTRUCT_SUBSETS if s.startswith(("ValidWills", "CABarExamEssays"))]


# ---------------------------------------------------------------- reddit_domain
REDDIT = "RedditLegalQA-reddit_legal_qa"
_TOPICS = ["My ex wants full custody of our kids after the divorce.",
           "I was arrested for shoplifting and have a court date next week.",
           "My landlord will not return my security deposit."]


def reddit_rec(i):
    return {**li_rec(i, REDDIT), "prompt": f"Post {i}: {_TOPICS[i % 3]}"}


def test_reddit_domain_is_disjoint_from_lawinstruct_sample():
    recs = [reddit_rec(i) for i in range(900)]
    li, _ = load_lawinstruct("train", subsets=[REDDIT], max_per_subset=300, records={REDDIT: recs})
    rd, rep = load_source("reddit_domain", "train", max_per_subset=300, records=recs)
    used = {e["messages"][0]["content"] for e in li}
    got = {e["messages"][0]["content"] for e in rd}
    assert len(li) == 300 and got and not used & got           # disjoint from the 50k-style sample
    assert rep["in_lawinstruct_sample"] == 300
    assert rep["pool"] == rep["kept"] + rep["skipped_other_domain"]
    # the pool is exactly the complement: every train example is in one of the two
    _, rep_all = load_lawinstruct("train", subsets=[REDDIT], max_per_subset=None, records={REDDIT: recs})
    assert rep["pool"] + 300 == rep_all["subsets"][REDDIT]["kept"]
    assert all("landlord" not in c for c in got)               # "other" posts are dropped
    assert {e["source"] for e in rd} <= {"reddit_domain:family", "reddit_domain:criminal",
                                         "reddit_domain:both"}
    assert rep["kept_family"] and rep["kept_criminal"] and "academic" in rep["licence"]


def test_reddit_domain_without_cap_has_no_train_and_test_is_tagged_test_split():
    recs = [reddit_rec(i) for i in range(900)]
    rd, rep = load_reddit_domain("train", max_per_subset=None, records=recs)
    assert rd == [] and rep["pool"] == 0                       # lawinstruct uses everything
    test, _ = load_reddit_domain("test", records=recs)
    li_test, _ = load_lawinstruct("test", subsets=[REDDIT], records={REDDIT: recs})
    tagged = {e["messages"][0]["content"] for e in test}
    assert tagged and tagged <= {e["messages"][0]["content"] for e in li_test}


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


# ---------------------------------------------------------------- pinned upstream revisions
def test_loaders_pass_pinned_or_given_revision(monkeypatch):
    import data.sft_sources as ss
    seen = []
    monkeypatch.setattr(ss, "_read_casehold_parquet", lambda split, revision: seen.append(("casehold", revision)) or [])
    monkeypatch.setattr(ss, "_read_lawinstruct_subset", lambda subset, revision: seen.append(("li", revision)) or [])
    ss.load_casehold("test")
    ss.load_casehold("test", revision="abc")
    ss.load_lawinstruct("test", subsets=["A-a"])
    ss.load_reddit_domain("test", revision="def")
    # (LawInstruct subsets are read twice: English instructions first, then conversion)
    assert set(seen) == {("casehold", ss.REVISIONS["casehold"]), ("casehold", "abc"),
                         ("li", ss.REVISIONS["lawinstruct"]), ("li", "def")}
    assert all(len(r) == 40 for r in ss.REVISIONS.values())          # full commit SHAs


def test_smol_smoltalk_passes_revision(monkeypatch):
    import sys
    import types

    import data.sft_sources as ss
    calls = {}

    class FakeDS(list):
        def shuffle(self, seed):
            return self

        def flatten_indices(self):
            return self

    fake = types.SimpleNamespace(load_dataset=lambda name, split, revision: calls.update(revision=revision) or FakeDS([1]))
    monkeypatch.setitem(sys.modules, "datasets", fake)
    _, info = ss.load_smol_smoltalk("test")
    assert calls["revision"] == ss.REVISIONS["smol_smoltalk"] == info["revision"]
