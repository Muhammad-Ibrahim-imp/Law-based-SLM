"""Reference-build comparison of scripts/modal_build_sft.py on fake manifests (no Modal, no network)."""

import copy
import json
from pathlib import Path

from scripts.modal_build_sft import BUILD_ORDER, compare_with_expected, expected_from_manifest


def fake_manifest():
    sources = {}
    for i, (source, split) in enumerate(BUILD_ORDER):
        sources.setdefault(source, {})[split] = {
            "status": "complete", "total_tokens": 8192 * (i + 1),
            "config": {"source_name": source, "split": split, "seed": 1234, "max_tokens": None},
            "stats": {"examples_kept": 10 + i, "kept_tokens": 1000 + i, "trained_tokens": 100 + i, "rows": i + 1},
            "shards": [{"index": 0, "num_tokens": 8192 * (i + 1),
                        "files": {"tokens": "shard_00000.bin", "mask": "shard_00000.mask.bin"},
                        "sha256": {"tokens": f"t{i}", "mask": f"m{i}"}}],
        }
    return {"tokenizer_fingerprint": "fp", "categories": {"sft": {"sources": sources}}}


def test_identical_build_matches():
    m = fake_manifest()
    assert compare_with_expected(m, expected_from_manifest(m)) == []


def test_differences_are_reported_precisely():
    m = fake_manifest()
    exp = expected_from_manifest(m)
    built = copy.deepcopy(m)
    s = built["categories"]["sft"]["sources"]
    s["casehold"]["test"]["stats"]["kept_tokens"] += 1
    s["lawinstruct"]["train"]["shards"][0]["sha256"]["mask"] = "other"
    s["smol_smoltalk"]["train"]["config"]["max_tokens"] = 5
    s["reddit_domain"]["test"]["status"] = "in_progress"
    diffs = compare_with_expected(built, exp)
    assert len(diffs) == 4
    assert any("casehold/test: kept_tokens" in d for d in diffs)
    assert any("lawinstruct/train: shard 0 shard_00000.mask.bin sha256 other" in d for d in diffs)
    assert any("smol_smoltalk/train: config" in d for d in diffs)
    assert any("reddit_domain/test: not built" in d for d in diffs)


def test_only_requested_splits_are_compared_and_fingerprint_checked():
    m = fake_manifest()
    exp = expected_from_manifest(m)
    partial = copy.deepcopy(m)
    del partial["categories"]["sft"]["sources"]["smol_smoltalk"]
    assert compare_with_expected(partial, exp, ["casehold/test", "lawinstruct/test"]) == []
    partial["tokenizer_fingerprint"] = "other"
    assert compare_with_expected(partial, exp, ["casehold/test"])[0].startswith("tokenizer_fingerprint")


def test_committed_reference_file_covers_every_split():
    exp = json.loads((Path(__file__).resolve().parents[1] / "reports" / "sft_expected.json").read_text())
    assert set(exp["splits"]) == {f"{s}/{p}" for s, p in BUILD_ORDER}
    assert exp["splits"]["smol_smoltalk/train"]["config"]["max_tokens"] == 248_366_163
    assert all(len(sh["sha256"]["tokens"]) == 64 for v in exp["splits"].values() for sh in v["shards"])
