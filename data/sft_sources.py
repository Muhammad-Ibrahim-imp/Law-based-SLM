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

Legal SFT data is the same for both SLMs (it is general US legal Q&A and
case-law reasoning, not family/criminal specific).
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import random
from collections import Counter
from typing import Iterable, Iterator, Mapping, Sequence

SEED_DEFAULT = 1234


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


def load_casehold(split: str, seed: int = SEED_DEFAULT, rows: Iterable[Mapping] | None = None):
    """``rows`` lets tests pass synthetic rows; normally read from the Hub."""
    if split not in CASEHOLD_SPLITS:
        raise ValueError(f"split must be one of {sorted(CASEHOLD_SPLITS)}")
    if rows is None:
        rows = _read_casehold_parquet(CASEHOLD_SPLITS[split])
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
        "dataset": CASEHOLD_REPO, "upstream_split": CASEHOLD_SPLITS[split],
        "read_via": "refs/convert/parquet (no loading script)", **report}


def _read_casehold_parquet(upstream_split: str) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    pattern = f"datasets/{CASEHOLD_REPO}@refs%2Fconvert%2Fparquet/all/{upstream_split}/*.parquet"
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
LAWINSTRUCT_SUBSETS = (
    "RedditLegalQA-reddit_legal_qa",
    "StackExchangeQuestionsLegal-stack_exchange_questions_legal",
    "CABarExamEssays-MainSubset",
    "MCExamsLaw-mc_exams_law_explain",
    "CaseBriefs-case_briefs",
    "OLCMemos-olc_memos",
    "MultiLexSum-long_to_short",
    "ValidWills-valid_wills_entailment",
)
LAWINSTRUCT_TEST_FRACTION = 0.01
LAWINSTRUCT_MAX_PER_SUBSET = 50_000   # stops RedditLegalQA (by far the largest) dominating


def lawinstruct_to_messages(rec: Mapping) -> tuple[list[dict] | None, str | None]:
    """One raw LawInstruct record -> (messages, None) or (None, skip_reason)."""
    if (rec.get("answer_language") or "en") != "en" or (rec.get("prompt_language") or "en") != "en":
        return None, "not_english"
    answer = (rec.get("answer") or "").strip()
    if not answer:
        return None, "no_answer"
    instruction = (rec.get("instruction") or "").strip()
    prompt = (rec.get("prompt") or "").strip()
    user = "\n\n".join(p for p in (instruction, prompt) if p)
    if not user:
        return None, "no_prompt"
    return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}], None


def lawinstruct_split_of(subset: str, messages: Sequence[Mapping]) -> str:
    key = _h("lawinstruct-split", subset, messages[0]["content"], messages[1]["content"])
    return "test" if key < LAWINSTRUCT_TEST_FRACTION * (1 << 64) else "train"


def load_lawinstruct(split: str, seed: int = SEED_DEFAULT,
                     subsets: Sequence[str] = LAWINSTRUCT_SUBSETS,
                     max_per_subset: int | None = LAWINSTRUCT_MAX_PER_SUBSET,
                     records: Mapping[str, Iterable[Mapping]] | None = None):
    """``records`` maps subset -> iterable of raw records (for tests);
    normally read from the Hub."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    report: dict = {"dataset": LAWINSTRUCT_REPO, "read_via": "data/*.jsonl.xz (no loading script)",
                    "test_fraction": LAWINSTRUCT_TEST_FRACTION, "max_per_subset": max_per_subset,
                    "subsets": {}}
    out = []
    for subset in subsets:
        rows = records[subset] if records is not None else _read_lawinstruct_subset(subset)
        c = Counter()
        seen: set[int] = set()
        kept: list[dict] = []
        for rec in rows:
            c["read"] += 1
            msgs, why = lawinstruct_to_messages(rec)
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
        # Cap per subset with a seeded sample (not "first N"), then record counts.
        if max_per_subset is not None and len(kept) > max_per_subset:
            c["skipped_over_cap"] += len(kept) - max_per_subset
            kept = random.Random(_h(str(seed), subset)).sample(kept, max_per_subset)
        c["kept"] = len(kept)
        report["subsets"][subset] = dict(c)
        out.extend(kept)
    report["kept"] = len(out)
    return _shuffled(out, seed), report


def _read_lawinstruct_subset(subset: str) -> Iterator[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"data/{subset}-train-"
    files = sorted(f for f in HfApi().list_repo_files(LAWINSTRUCT_REPO, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".jsonl.xz"))
    if not files:
        raise FileNotFoundError(f"LawInstruct subset {subset!r} not found ({prefix}*.jsonl.xz)")
    for fname in files:
        path = hf_hub_download(LAWINSTRUCT_REPO, fname, repo_type="dataset")
        with lzma.open(path, "rb") as raw, io.TextIOWrapper(raw, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


# ===================================================================== smol-smoltalk
def load_smol_smoltalk(split: str, seed: int = SEED_DEFAULT):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split)
    return ds.shuffle(seed=seed).flatten_indices(), {
        "dataset": "HuggingFaceTB/smol-smoltalk", "upstream_split": split, "rows": len(ds)}


# ===================================================================== registry
SOURCES = {
    "smol_smoltalk": load_smol_smoltalk,
    "casehold": load_casehold,
    "lawinstruct": load_lawinstruct,
}

LICENCES = {
    "smol_smoltalk": "Apache-2.0",
    "casehold": "CaseHOLD (Zheng et al., 2021); HF card states no licence - verify before release",
    "lawinstruct": "LawInstruct tagged MIT, but each upstream subset keeps its own licence "
                   "(Reddit / StackExchange / bar-exam content is not cleared for commercial use) "
                   "- academic use only, like Pile of Law",
}


def load_source(name: str, split: str, seed: int = SEED_DEFAULT, **options):
    if name not in SOURCES:
        raise ValueError(f"unknown SFT source {name!r}; choose from {sorted(SOURCES)}")
    examples, report = SOURCES[name](split, seed=seed, **options)
    report["licence"] = LICENCES[name]
    return examples, report
