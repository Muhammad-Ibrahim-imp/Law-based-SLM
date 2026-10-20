"""SFT source loaders: each turns one dataset into chat ``messages``.

Every loader returns ``(examples, report)``:

* ``examples``: a list (or HF Dataset) of ``{"messages": [...], "source": str}``
  in a deterministic, seeded shuffled order, ready for
  ``data.sft_prep.build_sft_split``.
* ``report``: counts of what was read, kept and skipped, stored in the
  manifest entry as ``source_info``.

None of these need the old ``datasets<4`` library:

* smol-smoltalk is Parquet-native.
* CaseHOLD is read from Hugging Face's automatic Parquet conversion (branch
  ``refs/convert/parquet``), not through its ``casehold.py`` loading script.
* LawInstruct is read straight from its ``data/*.jsonl.xz`` files, not through
  its ``lawinstruct.py`` loading script.

Splits (``train`` / ``test`` here mean SFT train / SFT held-out):

* smol-smoltalk: its own ``train`` / ``test``.
* CaseHOLD: ``train`` -> CaseHOLD ``train``; ``test`` -> CaseHOLD
  ``validation``. CaseHOLD's own ``test`` split is never used, so it stays
  clean for Member C's benchmark.
* LawInstruct ships train only; a deterministic 1% of records (by content
  hash) is held out as ``test``.
* reddit_domain: RedditLegalQA posts left out by LawInstruct's 50k cap,
  kept when the domain tagger (``data/domains.py``) marks them family and/or
  criminal. Same 1% hash split; train is disjoint from ``lawinstruct``.

Legal SFT data is the same for both SLMs (it is general US legal Q&A and
case-law reasoning, not family/criminal specific).
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import random
import re
from collections import Counter
from typing import Callable, Iterable, Iterator, Mapping, Sequence

SEED_DEFAULT = 1234

# Upstream revisions the reference SFT build used (20 Oct 2026), so a rebuild on Modal reads
# exactly the same data even if a dataset changes upstream. casehold = its refs/convert/parquet commit.
REVISIONS = {
    "smol_smoltalk": "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
    "casehold": "8a4dbd58704b80b9433264ce9c66421fc5bf6c19",
    "lawinstruct": "578e100783654ba791e55e5ebaaac343c66f408d",
}


def _h(*parts: str) -> int:
    """Stable 64-bit hash (Python's hash() is salted per process)."""
    d = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big")


def _shuffled(items: list, seed: int) -> list:
    random.Random(seed).shuffle(items)
    return items


# ===================================================================== CaseHOLD
CASEHOLD_REPO = "casehold/casehold"
CASEHOLD_SPLITS = {"train": "train", "test": "validation"}   # CaseHOLD test never used
LETTERS = "ABCDE"

_CASEHOLD_GEN = (
    "The following excerpt from a U.S. court opinion cites a case, but the holding "
    "of the cited case has been replaced with <HOLDING>.\n\n{context}\n\n"
    "What is the holding of the cited case?",
    "Read this passage from a judicial opinion. The holding of the cited case is "
    "marked <HOLDING>.\n\n{context}\n\nState the holding that belongs in place of "
    "<HOLDING>.",
    "{context}\n\nIn the passage above, the holding of a cited case is missing and "
    "marked <HOLDING>. What did that case hold?",
)
_CASEHOLD_MC = (
    "The following excerpt from a U.S. court opinion cites a case, but the holding "
    "of the cited case has been replaced with <HOLDING>.\n\n{context}\n\n"
    "Which holding belongs in place of <HOLDING>?\n\n{options}",
    "{context}\n\nThe passage above omits the holding of a cited case (marked "
    "<HOLDING>). Choose the correct holding.\n\n{options}",
    "Here is a passage from a court opinion with one cited holding removed "
    "(<HOLDING>):\n\n{context}\n\nOptions:\n{options}\n\nWhich option is the "
    "holding of the cited case?",
)


def _sentence(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def casehold_to_messages(row: Mapping) -> list[dict] | None:
    """One CaseHOLD row -> a single-turn conversation, or None if unusable.

    Half the rows (by stable hash of the example id) become open generation
    ("state the holding"), half become 5-way multiple choice whose answer is
    the letter AND the holding text, so the assistant turn is never just a
    single letter. The prompt template is also chosen by hash.
    """
    context = (row.get("citing_prompt") or "").strip()
    holdings = [(row.get(f"holding_{i}") or "").strip() for i in range(5)]
    label = row.get("label")
    try:
        label = int(label)
    except (TypeError, ValueError):
        return None
    if not context or "<HOLDING>" not in context or not 0 <= label < 5:
        return None
    if not holdings[label]:
        return None
    key = _h("casehold", str(row.get("example_id", "")), context[:200])
    answer = holdings[label]
    if key % 2 == 0:
        user = _CASEHOLD_GEN[(key >> 8) % len(_CASEHOLD_GEN)].format(context=context)
        assistant = _sentence(answer)
    else:
        if not all(holdings):
            return None
        options = "\n".join(f"({LETTERS[i]}) {h}" for i, h in enumerate(holdings))
        user = _CASEHOLD_MC[(key >> 8) % len(_CASEHOLD_MC)].format(
            context=context, options=options)
        assistant = f"({LETTERS[label]}) {_sentence(answer)}"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def load_casehold(split: str, seed: int = SEED_DEFAULT, rows: Iterable[Mapping] | None = None,
                  revision: str = REVISIONS["casehold"]):
    """``rows`` lets tests pass synthetic rows; normally read from the Hub at ``revision``."""
    if split not in CASEHOLD_SPLITS:
        raise ValueError(f"split must be one of {sorted(CASEHOLD_SPLITS)}")
    if rows is None:
        rows = _read_casehold_parquet(CASEHOLD_SPLITS[split], revision)
    report = Counter()
    out = []
    for row in rows:
        report["read"] += 1
        msgs = casehold_to_messages(row)
        if msgs is None:
            report["skipped_unusable"] += 1
            continue
        out.append({"messages": msgs, "source": "casehold"})
    report["kept"] = len(out)
    return _shuffled(out, seed), {
        "dataset": CASEHOLD_REPO, "upstream_split": CASEHOLD_SPLITS[split], "revision": revision,
        "read_via": "refs/convert/parquet (no loading script)", **report}


def _read_casehold_parquet(upstream_split: str, revision: str = REVISIONS["casehold"]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    pattern = f"datasets/{CASEHOLD_REPO}@{revision}/all/{upstream_split}/*.parquet"
    paths = sorted(fs.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no CaseHOLD parquet files match {pattern}")
    for path in paths:
        with fs.open(path, "rb") as f:
            table = pq.read_table(f)
        yield from table.to_pylist()


# ===================================================================== LawInstruct
LAWINSTRUCT_REPO = "lawinstruct/lawinstruct"

# English, US-relevant subsets (dataset-subset names as in data/<name>-train-N.jsonl.xz).
# Chosen for the project: legal Q&A in conversational form, US bar-exam essays and
# explanations, US case briefs/memos/summaries, wills. Deliberately left out:
#   * CaseHOLD-derived subsets (LexGLUE-case_hold, NaturalInstructionsLegal-casehold_*):
#     we use CaseHOLD directly, and LexGLUE's split differs from casehold/casehold's,
#     so its train rows can overlap CaseHOLD's test set (eval leakage).
#   * Non-English or non-US jurisdictions (Swiss, Brazilian, Chinese, Indian, EU, ...).
#   * Narrow label tasks whose answer is a single class name (little to learn from).
#   * ValidWills-valid_wills_entailment: answers are yes/no (~5 tokens each).
#   * CABarExamEssays-MainSubset: each of its 62 records is a whole exam booklet
#     (all questions + selected answers) in ``answer`` with an empty prompt, and
#     every one is far longer than 8,192 tokens; there is no question field to use.
LAWINSTRUCT_SUBSETS = (
    "RedditLegalQA-reddit_legal_qa",
    "StackExchangeQuestionsLegal-stack_exchange_questions_legal",
    "MCExamsLaw-mc_exams_law_explain",
    "CaseBriefs-case_briefs",
    "OLCMemos-olc_memos",
    "MultiLexSum-long_to_short",
)
LAWINSTRUCT_TEST_FRACTION = 0.01
LAWINSTRUCT_MAX_PER_SUBSET = 50_000   # stops RedditLegalQA (by far the largest) dominating


def lawinstruct_to_messages(rec: Mapping, english_instructions: Sequence[str] = ()
                            ) -> tuple[list[dict] | None, str | None]:
    """One raw LawInstruct record -> (messages, None) or (None, skip_reason).

    LawInstruct machine-translates ~95% of its instructions into 23 other EU
    languages (``instruction_language``); the prompt and answer stay English.
    A non-English instruction is replaced by one of the subset's own English
    instructions (picked by a stable hash of the prompt), or dropped when the
    subset has none, so every user turn is English.
    """
    if (rec.get("answer_language") or "en") != "en" or (rec.get("prompt_language") or "en") != "en":
        return None, "not_english"
    answer = (rec.get("answer") or "").strip()
    if not answer:
        return None, "no_answer"
    instruction = (rec.get("instruction") or "").strip()
    prompt = (rec.get("prompt") or "").strip()
    if _EMPTY_QUESTION.fullmatch(prompt):
        # CaseBriefs and OLCMemos ship an empty prompt with the question inside the answer.
        prompt, answer = _split_embedded_question(rec.get("dataset_name") or "", answer)
        if not prompt and rec.get("dataset_name") == "RedditLegalQA":
            return None, "empty_question"   # "Question:" with no post
    if (rec.get("instruction_language") or "en") != "en":
        instruction = (english_instructions[_h("li-instr", prompt) % len(english_instructions)]
                       if english_instructions else "")
    user = "\n\n".join(p for p in (instruction, prompt) if p)
    if not user:
        return None, "no_prompt"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}], None


def lawinstruct_split_of(subset: str, messages: Sequence[Mapping]) -> str:
    key = _h("lawinstruct-split", subset, messages[0]["content"], messages[1]["content"])
    return "test" if key < LAWINSTRUCT_TEST_FRACTION * (1 << 64) else "train"


_EMPTY_QUESTION = re.compile(r"\s*(?:question\s*:)?\s*", re.I)
_CASEBRIEF = re.compile(r"\s*Question:\s*(.+?)\n\s*Analysis:\s*(.+)", re.S)


def _split_embedded_question(dataset: str, answer: str) -> tuple[str, str]:
    """(question, answer) for subsets whose question sits inside the answer; ("", answer) otherwise.
    CaseBriefs: "Question: <facts + issue>\\nAnalysis: <analysis>".
    OLCMemos: the opening block (title + headnote) becomes the research question."""
    if dataset == "CaseBriefs" and (m := _CASEBRIEF.match(answer)):
        return m.group(1).strip(), m.group(2).strip()
    if dataset == "OLCMemos" and "\n\n" in answer:
        head, body = answer.split("\n\n", 1)
        if head.strip() and body.strip():
            return " ".join(head.split()), body.strip()
    return "", answer


def english_instructions_of(rows: Iterable[Mapping]) -> list[str]:
    """The distinct English instructions of one subset, sorted (deterministic)."""
    return sorted({(r.get("instruction") or "").strip() for r in rows
                   if (r.get("instruction_language") or "en") == "en" and (r.get("instruction") or "").strip()})


def _lawinstruct_candidates(subset: str, split: str, rows: Callable[[], Iterable[Mapping]]
                            ) -> tuple[list[dict], Counter]:
    """All usable, de-duplicated examples of one subset in ``split``, in file order.
    ``rows()`` is called twice: once to collect the English instructions, once to convert."""
    c = Counter()
    seen: set[int] = set()
    kept: list[dict] = []
    english = english_instructions_of(rows())
    c["english_instructions"] = len(english)
    for rec in rows():
        c["read"] += 1
        if (rec.get("instruction_language") or "en") != "en":
            c["instruction_replaced"] += 1
        msgs, why = lawinstruct_to_messages(rec, english)
        if msgs is None:
            c[f"skipped_{why}"] += 1
            continue
        dup = _h(msgs[0]["content"], msgs[1]["content"])
        if dup in seen:
            c["skipped_duplicate"] += 1
            continue
        seen.add(dup)
        if lawinstruct_split_of(subset, msgs) != split:
            continue
        kept.append({"messages": msgs, "source": subset})
    return kept, c


def _lawinstruct_cap(n: int, seed: int, subset: str, max_per_subset: int | None) -> list[int] | None:
    """Indices of the seeded per-subset sample, or None when there is no cap to apply.
    ``Random.sample`` picks the same indices for ``range(n)`` as for any length-n list."""
    if max_per_subset is None or n <= max_per_subset:
        return None
    return random.Random(_h(str(seed), subset)).sample(range(n), max_per_subset)


def load_lawinstruct(split: str, seed: int = SEED_DEFAULT,
                     subsets: Sequence[str] = LAWINSTRUCT_SUBSETS,
                     max_per_subset: int | None = LAWINSTRUCT_MAX_PER_SUBSET,
                     records: Mapping[str, Iterable[Mapping]] | None = None,
                     revision: str = REVISIONS["lawinstruct"]):
    """``records`` maps subset -> iterable of raw records (for tests);
    normally read from the Hub at ``revision``."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    report: dict = {"dataset": LAWINSTRUCT_REPO, "revision": revision,
                    "read_via": "data/*.jsonl.xz (no loading script)",
                    "test_fraction": LAWINSTRUCT_TEST_FRACTION, "max_per_subset": max_per_subset,
                    "subsets": {}}
    out = []
    for subset in subsets:
        rows = ((lambda s=subset: records[s]) if records is not None
                else (lambda s=subset: _read_lawinstruct_subset(s, revision)))
        kept, c = _lawinstruct_candidates(subset, split, rows)
        # Cap per subset with a seeded sample (not "first N"), then record counts.
        idx = _lawinstruct_cap(len(kept), seed, subset, max_per_subset)
        if idx is not None:
            c["skipped_over_cap"] += len(kept) - len(idx)
            kept = [kept[i] for i in idx]
        c["kept"] = len(kept)
        report["subsets"][subset] = dict(c)
        out.extend(kept)
    report["kept"] = len(out)
    return _shuffled(out, seed), report


# ===================================================================== reddit_domain
REDDIT_SUBSET = "RedditLegalQA-reddit_legal_qa"
REDDIT_DOMAINS = ("family", "criminal")


def load_reddit_domain(split: str, seed: int = SEED_DEFAULT,
                       max_per_subset: int | None = LAWINSTRUCT_MAX_PER_SUBSET,
                       records: Iterable[Mapping] | None = None,
                       revision: str = REVISIONS["lawinstruct"]):
    """Family- and criminal-tagged RedditLegalQA posts that ``lawinstruct`` does not use.

    train: the complement of the exact seeded 50k sample ``load_lawinstruct``
    keeps (same seed, same skips, same 1% hash split), filtered to posts the
    domain tagger marks family and/or criminal. Disjoint from ``lawinstruct``.
    test: the Reddit test split is far below the cap, so ``lawinstruct`` keeps
    all of it and the complement would be empty; instead this is the
    family/criminal-tagged part of the Reddit test split (a subset of
    ``lawinstruct``'s test, still disjoint from every train split).
    Tagging uses the user turn with ``data.domains.profile_for`` (short profile
    for posts up to 4,000 chars, loose above).
    """
    from data.domains import classify, profile_for

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    rows = (lambda: records) if records is not None else (lambda: _read_lawinstruct_subset(REDDIT_SUBSET, revision))
    kept, c = _lawinstruct_candidates(REDDIT_SUBSET, split, rows)
    if split == "train":
        idx = _lawinstruct_cap(len(kept), seed, REDDIT_SUBSET, max_per_subset)
        used = set(idx) if idx is not None else set(range(len(kept)))
        c["in_lawinstruct_sample"] = len(used)
        pool = [ex for i, ex in enumerate(kept) if i not in used]
    else:
        pool = kept
    c["pool"] = len(pool)
    out = []
    for ex in pool:
        user = ex["messages"][0]["content"]
        tags = classify(user, profile_for(user)) & set(REDDIT_DOMAINS)
        if not tags:
            c["skipped_other_domain"] += 1
            continue
        label = "both" if len(tags) == 2 else tags.pop()
        c[f"kept_{label}"] += 1
        out.append({"messages": ex["messages"], "source": f"reddit_domain:{label}"})
    c["kept"] = len(out)
    return _shuffled(out, seed), {
        "dataset": LAWINSTRUCT_REPO, "revision": revision, "subset": REDDIT_SUBSET,
        "split_rule": "lawinstruct 1% content hash",
        "max_per_subset": max_per_subset, "tagger": "data.domains short/loose (profile_for)", **c}


def _read_lawinstruct_subset(subset: str, revision: str = REVISIONS["lawinstruct"]) -> Iterator[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"data/{subset}-train-"
    files = sorted(f for f in HfApi().list_repo_files(LAWINSTRUCT_REPO, repo_type="dataset", revision=revision)
                   if f.startswith(prefix) and f.endswith(".jsonl.xz"))
    if not files:
        raise FileNotFoundError(f"LawInstruct subset {subset!r} not found ({prefix}*.jsonl.xz)")
    for fname in files:
        path = hf_hub_download(LAWINSTRUCT_REPO, fname, repo_type="dataset", revision=revision)
        with lzma.open(path, "rb") as raw, io.TextIOWrapper(raw, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ===================================================================== smol-smoltalk
def load_smol_smoltalk(split: str, seed: int = SEED_DEFAULT, revision: str = REVISIONS["smol_smoltalk"]):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split, revision=revision)
    return ds.shuffle(seed=seed).flatten_indices(), {
        "dataset": "HuggingFaceTB/smol-smoltalk", "upstream_split": split, "revision": revision, "rows": len(ds)}


# ===================================================================== registry
SOURCES = {
    "smol_smoltalk": load_smol_smoltalk,
    "casehold": load_casehold,
    "lawinstruct": load_lawinstruct,
    "reddit_domain": load_reddit_domain,
}

LICENCES = {
    "smol_smoltalk": "Apache-2.0",
    "casehold": "CaseHOLD (Zheng et al., 2021); HF card states no licence - verify before release",
    "lawinstruct": "LawInstruct tagged MIT, but each upstream subset keeps its own licence "
                   "(Reddit / StackExchange / bar-exam content is not cleared for commercial use) "
                   "- academic use only, like Pile of Law",
    "reddit_domain": "RedditLegalQA via LawInstruct: Reddit content, not cleared for commercial use "
                     "- academic use only",
}


def load_source(name: str, split: str, seed: int = SEED_DEFAULT, **options):
    if name not in SOURCES:
        raise ValueError(f"unknown SFT source {name!r}; choose from {sorted(SOURCES)}")
    examples, report = SOURCES[name](split, seed=seed, **options)
    report["licence"] = LICENCES[name]
    return examples, report
