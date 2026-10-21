"""Legal pretraining pipeline on tiny synthetic candidates: features -> dedup/split ->
tokenize -> write -> manifest, read back by ShardedTokenDataset and validate_shards."""

import gzip
import json

import numpy as np
import pytest
import torch

from data import legal_prep as lp
from data.dedup import ngram_hashes
from data.general_prep import validate_shards
from data.shard_dataset import ShardedTokenDataset


WORDS = ("court record counsel motion appeal judgment order hearing evidence witness statute claim "
         "party trial ruling issue review error fact law opinion jury verdict petition relief").split()


def body(i, topic):
    rng = np.random.default_rng(i)   # distinct text per document, so documents are not near-duplicates
    return f"In this {topic} matter " + " ".join(rng.choice(WORDS, 400)) + "."


EVAL_PASSAGE = "evaluation " + " ".join(np.random.default_rng(999).choice(WORDS, 100))   # ~100 words of eval text


def write_job(paths, name, docs):
    with gzip.open(paths.cand / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps({**d, "chars": len(d["text"])}) + "\n")
    stats = {d: {"matched": 0, "matched_chars": 0, "kept": 0, "kept_chars": 0, "dropped": {"too_short": 1}}
             for d in lp.DOMAINS}
    (paths.cand / f"{name}.stats.json").write_text(json.dumps({"job": name, **stats}))


def doc(i, source, fam, crim, text=None, kind="case"):
    return {"id": f"{source}:{i}", "source": source, "kind": kind, "family": fam, "criminal": crim,
            "text": text or body(i, "family" if fam else "criminal")}


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "VALIDATION_RATIO", 0.3)   # enough validation docs in a tiny corpus
    monkeypatch.setattr(lp, "SHARD_TOKENS", 4096)
    monkeypatch.setattr(lp, "LOG_PATH", tmp_path / "progress.log")
    paths = lp.Paths(tmp_path / "work")
    dup = body(1, "family")
    cl = [doc(i, "courtlistener", i % 3 != 0, i % 3 != 1) for i in range(30)]
    cl.append(doc(100, "courtlistener", True, False, text=dup))                      # exact dup of cl:1
    hfl = [doc(200, "hfforlegal", True, False, text=dup.replace("In this", "In that", 1)),  # near dup
           doc(201, "hfforlegal", False, True, text=body(201, "criminal") + " " + EVAL_PASSAGE)]
    stat = [doc(300 + i, "uscode", False, True, kind="statute") for i in range(8)]
    write_job(paths, "cl", cl)
    write_job(paths, "hfl", hfl)
    write_job(paths, "stat", stat)
    np.save(paths.eval / "casehold.npy", ngram_hashes(EVAL_PASSAGE))
    np.save(paths.eval / "legalbench.npy", np.zeros(0, np.uint64))
    jobs = ["cl", "hfl", "stat"]
    for j in jobs:
        lp.features_job(j, str(paths.work))
    dedup = lp.dedup_and_split(paths, jobs)
    for j in jobs:
        lp.tokenize_job(j, str(paths.work))
    root = tmp_path / "out"
    written = {}
    for d in lp.DOMAINS:
        for s in ("train", "validation"):
            cap = 3000 if (d, s) == ("criminal", "train") else None
            w = lp.write_category(paths, jobs, str(root), d, s, cap)
            lp.register_category(str(root), w["category"], lp.manifest_entry(w, d, s, dedup, {"cl": json.loads(
                (paths.cand / "cl.stats.json").read_text())}, cap))
            written[w["category"]] = w
    return paths, dedup, root, written


def test_dedup_decontam_and_split_consistency(built):
    paths, dedup, _, _ = built
    fam = dedup["family"]
    assert fam["exact_duplicates_removed"] >= 1 and fam["near_duplicates_removed"] >= 1
    assert dedup["criminal"]["decontaminated"] == 1 and dedup["criminal"]["decontaminated_by"]["casehold"] == 1
    assert dedup["split_consistent_in_both"] and dedup["overlap_docs_in_both"] > 0
    # the CourtListener copy wins the near-dup cluster over HFforLegal
    sel = np.load(paths.feat / "hfl.select.npz")
    assert sel["family"][0] == -1


def test_split_is_deterministic_and_cluster_level():
    assert lp.split_of("cl:123") == lp.split_of("cl:123")
    vals = [lp.split_of(f"cl:{i}") for i in range(20000)]
    assert 0.002 < np.mean(vals) < 0.008          # ~0.5%


def test_manifest_schema_readable_by_loaders(built):
    _, _, root, written = built
    m = json.loads((root / "manifest.json").read_text())
    for cat in written:
        e = m["categories"][cat]
        for key in ("shards", "total_tokens_written", "total_docs_written", "sequence_length", "shard_tokens_target",
                    "split", "per_source_token_counts", "per_source_doc_counts", "total_exact_duplicates_removed",
                    "total_near_duplicates_removed", "total_decontaminated", "filter_counts", "started_at"):
            assert key in e, (cat, key)
        assert set(e["shards"][0]) == {"file", "tokens", "checksum_sha256"}
        assert e["started_at"].endswith("Z")
        assert validate_shards(str(root), cat)
        ds = ShardedTokenDataset(str(root), cat, sequence_length=64)
        ex = ds[0]
        assert ex["input_ids"].dtype == torch.long and torch.equal(ex["labels"][:-1], ex["input_ids"][1:])
    with pytest.raises(RuntimeError):
        lp.register_category(str(root), "family_law_train", {})      # never overwrite an entry


def test_criminal_cap_keeps_statutes_and_reports_left_out(built):
    _, _, _, written = built
    w = written["criminal_law_train"]
    assert w["per_source_doc_counts"].get("uscode", 0) >= 1          # statutes always kept
    assert w["left_out_by_cap"]["docs"] > 0 and w["total_tokens"] >= 3000
    assert w["total_tokens"] - 3000 < max(w["per_source_token_counts"].values())


def test_eos_after_every_document(built):
    paths, _, root, written = built
    from tokenizer import load_smollm2_tokenizer
    eos = load_smollm2_tokenizer().eos_token_id
    arr = np.concatenate([np.fromfile(root / "family_law_train" / s["file"], dtype=np.uint16)
                          for s in written["family_law_train"]["shards"]])
    assert arr[-1] == eos and (arr == eos).sum() == written["family_law_train"]["docs"]
