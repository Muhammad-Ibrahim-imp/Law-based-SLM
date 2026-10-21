import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.sft_dataset import IGNORE_INDEX, SFTDataset
from data.sft_prep import SFTPrepConfig, _StopForTest, build_sft_split, validate_sft_split
from tests.fake_tokenizer import FakeTokenizer

L = 160           # tiny rows so packing and shard boundaries get exercised
SHARD = L * 4     # 4 rows per shard


def make_examples(n=200, seed=0):
    rng = random.Random(seed)
    words = "law court fee child order case judge rule term file".split()
    out = []
    for i in range(n):
        turns = []
        if rng.random() < 0.3:
            turns.append({"role": "system", "content": "be kind"})
        for _ in range(rng.randint(1, 2)):
            turns.append({"role": "user", "content": " ".join(rng.choices(words, k=rng.randint(1, 4)))})
            turns.append({"role": "assistant", "content": " ".join(rng.choices(words, k=rng.randint(1, 4)))})
        out.append({"messages": turns, "source": f"src{i % 3}"})
    # One of each drop reason
    out += [
        {"messages": [{"role": "user", "content": "x " * 100}, {"role": "assistant", "content": "y"}]},
        {"messages": [{"role": "user", "content": "<|im_end|>"}, {"role": "assistant", "content": "y"}]},
        {"messages": [{"role": "user", "content": "only a question"}]},
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}]},
        {"messages": [{"role": "robot", "content": "q"}]},
    ]
    return out


def cfg_for(root, **kw):
    return SFTPrepConfig(dataset_root=str(root), sequence_length=L, shard_tokens=SHARD,
                         max_open_rows=4, **kw)


def build(root, examples=None, **kw):
    examples = examples if examples is not None else make_examples()
    stop = kw.pop("_stop_after_shards", None)
    return build_sft_split(cfg_for(root, **{k: v for k, v in kw.items() if k != "overwrite"}),
                           examples, FakeTokenizer(), "fake-fp",
                           overwrite=kw.get("overwrite", False), _stop_after_shards=stop)


def test_build_validate_and_drops(tmp_path):
    entry = build(tmp_path)
    st = entry["stats"]
    assert st["dropped"] == {"bad_format": 1, "special_text": 1, "empty_assistant": 1,
                             "no_assistant": 1, "too_long": 1}
    assert st["examples_kept"] == 200
    assert sum(s["examples"] for s in st["per_source"].values()) == 200
    assert entry["total_tokens"] % L == 0
    assert len(entry["shards"]) >= 2  # incl. a partial final shard
    tot = validate_sft_split(tmp_path, "smol_smoltalk", "train")
    assert tot["examples"] == 200
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["categories"]["sft"]["sources"]["smol_smoltalk"]["train"]["status"] == "complete"
    assert entry["created_at"].endswith("Z")


def test_dataset_labels(tmp_path):
    build(tmp_path)
    ds = SFTDataset(tmp_path, [("smol_smoltalk", "train")], sequence_length=L)
    tok = FakeTokenizer()
    trained = 0
    for i in range(len(ds)):
        item = ds[i]
        ids, lab = item["input_ids"], item["labels"]
        assert ids.dtype == lab.dtype == torch.long and ids.shape == lab.shape == (L,)
        keep = lab != IGNORE_INDEX
        assert lab[-1] == IGNORE_INDEX
        # every target equals the next input token
        assert torch.equal(lab[:-1][keep[:-1]], ids[1:][keep[:-1]])
        trained += int(keep.sum())
        # targets never include role headers, im_start or padding
        assert not (lab[keep] == 1).any() and not (lab[keep] == 0).any()
    st = json.loads((tmp_path / "manifest.json").read_text())["categories"]["sft"]["sources"]["smol_smoltalk"]["train"]["stats"]
    # The first token of a row is never a target (it has no predecessor);
    # it is always <|im_start|> (mask 0), so trained counts match exactly.
    assert trained == st["trained_tokens"]
    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    assert batch["input_ids"].shape == batch["labels"].shape == (8, L)
    # Spot-check decoded targets are assistant text
    item = ds[0]
    text = tok.decode(item["labels"][item["labels"] != IGNORE_INDEX].tolist())
    assert "<|im_end|>" in text and "user" not in text


def test_resume_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    full = build(a)
    with pytest.raises(_StopForTest):
        build(b, _stop_after_shards=2)
    assert (b / "sft/smol_smoltalk/train/_checkpoint.npz").exists()
    resumed = build(b)
    assert resumed["stats"] == full["stats"]
    assert [s["sha256"] for s in resumed["shards"]] == [s["sha256"] for s in full["shards"]]
    assert not (b / "sft/smol_smoltalk/train/_checkpoint.npz").exists()


def test_resume_refuses_changed_config(tmp_path):
    with pytest.raises(_StopForTest):
        build(tmp_path, _stop_after_shards=1)
    with pytest.raises(RuntimeError, match="config changed"):
        build(tmp_path, seed=99)


def test_complete_build_not_overwritten_silently(tmp_path):
    build(tmp_path)
    with pytest.raises(RuntimeError, match="already built"):
        build(tmp_path)
    build(tmp_path, overwrite=True)


def test_max_tokens_cap(tmp_path):
    entry = build(tmp_path, max_tokens=1000)
    assert entry["budget_reached"]
    assert 1000 <= entry["stats"]["kept_tokens"] < 1000 + L
    validate_sft_split(tmp_path, "smol_smoltalk", "train")


def test_measure_only_writes_nothing(tmp_path):
    rep = build_sft_split(cfg_for(tmp_path), make_examples(), FakeTokenizer(), "fp",
                          measure_only=True)
    assert rep["examples_kept"] == 200 and 0 <= rep["padding_fraction"] < 1
    assert not any(tmp_path.iterdir())


def test_validation_catches_corruption(tmp_path):
    entry = build(tmp_path)
    p = tmp_path / "sft/smol_smoltalk/train" / entry["shards"][0]["files"]["mask"]
    data = np.fromfile(p, dtype=np.uint8)
    data[0] = 1
    data.tofile(p)
    with pytest.raises(ValueError, match="checksum"):
        validate_sft_split(tmp_path, "smol_smoltalk", "train")


# ---------------------------------------------------------------- shared manifest
def test_sft_first_then_general_prep_can_add(tmp_path, monkeypatch):
    """SFT creates the manifest; general_prep's loader must still work on it."""
    import data.general_prep as gp
    monkeypatch.setattr(gp, "get_tokenizer_fingerprint", lambda: "fake-fp")
    build(tmp_path)
    m = gp._load_or_init_manifest(str(tmp_path))
    for key in ("manifest_version", "dtype", "tokenizer_checkpoint", "tokenizer_fingerprint"):
        assert key in m
    m["categories"]["general_train"] = {"shards": [], "total_tokens_written": 0}
    gp._save_manifest(str(tmp_path), m)
    # SFT section survives general_prep's save, and is still readable.
    validate_sft_split(tmp_path, "smol_smoltalk", "train")
    SFTDataset(tmp_path, [("smol_smoltalk", "train")])


def test_general_first_then_sft(tmp_path):
    """SFT adds to an existing general manifest without touching other categories."""
    from data.shard_dataset import ShardedTokenDataset
    cat = tmp_path / "general_train"
    cat.mkdir()
    np.arange(L * 3, dtype=np.uint16).tofile(cat / "shard_00000.bin")
    general = {"manifest_version": "1.0", "dtype": "uint16",
               "tokenizer_checkpoint": "HuggingFaceTB/SmolLM2-135M",
               "tokenizer_fingerprint": "fake-fp",
               "categories": {"general_train": {
                   "shards": [{"file": "shard_00000.bin", "tokens": L * 3}]}}}
    (tmp_path / "manifest.json").write_text(json.dumps(general))
    build(tmp_path)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["categories"]["general_train"] == general["categories"]["general_train"]
    assert set(m["categories"]) == {"general_train", "sft"}
    assert len(ShardedTokenDataset(str(tmp_path), "general_train", L - 1)) == 3


def test_refuses_manifest_from_other_tokenizer(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"tokenizer_fingerprint": "other", "categories": {}}))
    with pytest.raises(RuntimeError, match="tokenizer_fingerprint"):
        build(tmp_path)


def test_dataset_takes_sequence_length_from_manifest(tmp_path):
    build(tmp_path)
    ds = SFTDataset(tmp_path, [("smol_smoltalk", "train")])
    assert ds.sequence_length == L and ds[0]["input_ids"].shape == (L,)
    with pytest.raises(ValueError, match="packed at"):
        SFTDataset(tmp_path, [("smol_smoltalk", "train")], sequence_length=2 * L)


def test_default_sequence_length_matches_200m_pretraining():
    assert SFTPrepConfig(dataset_root="x").sequence_length == 8192
    SFTPrepConfig(dataset_root="x").check()  # 2^28 shard is a whole number of 8,192 rows
