import json

import pytest

from scripts.merge_manifest_section import merge_section

HEADER = {"manifest_version": "1.0", "dtype": "uint16", "tokenizer_checkpoint": "x", "tokenizer_fingerprint": "fp1"}


def write(root, manifest):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_merge_keeps_other_categories_and_replaces_section(tmp_path):
    dst = tmp_path / "remote"
    write(dst, {**HEADER, "categories": {"general_train": {"shards": [1]}, "sft": {"old": True}}})
    src = {**HEADER, "categories": {"sft": {"sources": {"casehold": {}}}, "general": {"local": True}}}
    out = merge_section(src, dst)
    assert out["categories"]["general_train"] == {"shards": [1]}        # untouched
    assert out["categories"]["sft"] == {"sources": {"casehold": {}}}     # replaced whole
    assert "general" not in out["categories"]                            # only the section is copied
    assert json.loads((dst / "manifest.json").read_text()) == out


def test_merge_refuses_other_tokenizer_and_missing_section(tmp_path):
    dst = tmp_path / "remote"
    write(dst, {**HEADER, "tokenizer_fingerprint": "fp2", "categories": {"general_train": {}}})
    with pytest.raises(ValueError, match="tokenizer_fingerprint"):
        merge_section({**HEADER, "categories": {"sft": {}}}, dst)
    assert json.loads((dst / "manifest.json").read_text())["tokenizer_fingerprint"] == "fp2"
    with pytest.raises(ValueError, match="no categories.sft"):
        merge_section({**HEADER, "categories": {}}, dst)


def test_merge_into_empty_target_copies_header(tmp_path):
    out = merge_section({**HEADER, "categories": {"sft": {"a": 1}}}, tmp_path / "new")
    assert out["tokenizer_fingerprint"] == "fp1" and out["categories"] == {"sft": {"a": 1}}
