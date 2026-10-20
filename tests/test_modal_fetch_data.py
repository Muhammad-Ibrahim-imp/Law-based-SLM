"""Local logic of scripts/modal_fetch_data.py on fake shards (no Modal, no network)."""

import hashlib
import json

import numpy as np
import pytest

from scripts.modal_fetch_data import existing_categories, install, verify_downloaded

HEADER = {"manifest_version": "1.0", "dtype": "uint16", "tokenizer_checkpoint": "x", "tokenizer_fingerprint": "fp1"}


def make_category(root, cat, n_tokens=1000, seed=0):
    arr = np.random.default_rng(seed).integers(0, 49152, n_tokens, dtype=np.uint16)
    (root / cat).mkdir(parents=True)
    (root / cat / "shard_00000.bin").write_bytes(arr.tobytes())
    return {"shards": [{"file": "shard_00000.bin", "tokens": n_tokens,
                        "checksum_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}],
            "total_tokens_written": n_tokens}


@pytest.fixture
def incoming(tmp_path):
    inc = tmp_path / "_incoming"
    cats = {"family_law_train": make_category(inc, "family_law_train", seed=1),
            "criminal_law_train": make_category(inc, "criminal_law_train", seed=2)}
    (inc / "manifest.json").write_text(json.dumps({**HEADER, "categories": cats}))
    return inc


def test_verify_detects_missing_size_and_checksum(incoming):
    m = json.loads((incoming / "manifest.json").read_text())
    cats = ["family_law_train", "criminal_law_train"]
    assert verify_downloaded(incoming, m, cats) == []
    shard = incoming / "family_law_train" / "shard_00000.bin"
    data = bytearray(shard.read_bytes())
    data[0] ^= 1
    shard.write_bytes(bytes(data))                                    # same size, wrong content
    (incoming / "criminal_law_train" / "shard_00000.bin").write_bytes(b"\0" * 10)
    problems = verify_downloaded(incoming, m, cats + ["family_law_validation"])
    assert any("SHA-256" in p for p in problems) and any("size" in p for p in problems)
    assert any("not in the downloaded manifest" in p for p in problems)


def test_install_moves_merges_and_keeps_other_categories(incoming, tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({**HEADER, "categories": {"general_train": {"g": 1}}}))
    install(incoming, root, ["family_law_train", "criminal_law_train"])
    m = json.loads((root / "manifest.json").read_text())
    assert set(m["categories"]) == {"general_train", "family_law_train", "criminal_law_train"}
    assert (root / "family_law_train" / "shard_00000.bin").exists()
    assert not (incoming / "family_law_train").exists()


def test_install_refuses_overwrite_and_fingerprint_mismatch(incoming, tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({**HEADER, "categories": {"family_law_train": {"old": 1}}}))
    assert existing_categories(root, ["family_law_train", "criminal_law_train"]) == ["family_law_train"]
    with pytest.raises(RuntimeError, match="--overwrite"):
        install(incoming, root, ["family_law_train"])
    assert json.loads((root / "manifest.json").read_text())["categories"]["family_law_train"] == {"old": 1}
    install(incoming, root, ["family_law_train"], overwrite=True)         # explicit overwrite works
    assert "shards" in json.loads((root / "manifest.json").read_text())["categories"]["family_law_train"]

    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({**HEADER, "tokenizer_fingerprint": "fp2", "categories": {}}))
    with pytest.raises(ValueError, match="tokenizer_fingerprint"):
        install(incoming, other, ["criminal_law_train"])
    assert not (other / "criminal_law_train").exists()                     # nothing moved on refusal
