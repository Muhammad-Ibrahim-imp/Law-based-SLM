"""
data/general_prep.py

Reusable logic for building the "general" pretraining slice (SmolLM-Corpus)
for the law-SLM project. Target per README: 5.0B tokens.

Called from scripts/prepare_general_data.py (thin entry point), per README
section 17: reusable implementation belongs in the appropriate package, not
in scripts/.

Shard/format spec (finalized for this project):
    Shard size:     268,435,456 tokens (2^28) -> 512 MiB per full shard
    Storage dtype:  uint16 (vocab 49,152 fits safely)
    Sequence len:   2,048 -> 131,072 sequences/shard
    Micro-batch:    32    -> 4,096 micro-steps/shard
    Accumulation:   8     -> exactly 512 optimizer steps/shard
    Global batch:   524,288 tokens
    GPU:            H200

Output layout (dataset root is shared across all categories -- general,
family_law, criminal_law, sft -- each prep script writes into its own
subfolder and merges into one shared manifest.json at the root):

    dataset/
    ├── general/
    │   ├── shard_00000.bin
    │   └── ...
    ├── family_law/      (written by a separate script)
    ├── criminal_law/    (written by a separate script)
    ├── sft/              (written by a separate script)
    └── manifest.json     (shared, one entry per category -- see below)

Each .bin is a continuous uint16 token stream -- input tokens only. Labels
are derived by shifting one position at load time; cast to torch.long only
immediately before the batch hits the model. This file does none of that
shifting/casting -- that's training-side code.

NOTE ON TERMINOLOGY: shards are token-ID arrays on disk, not training
batches. Batching (sequence windows, shuffling, micro-batch/accumulation)
happens later, in training, and must produce the shared batch format from
README section 14: {"input_ids": Tensor[B, T], "labels": Tensor[B, T]}
(torch.long).
"""

import hashlib
import json
import os
import time

import numpy as np
from datasets import load_dataset

from tokenizer import load_smollm2_tokenizer, get_tokenizer_fingerprint, TOKENIZER_CHECKPOINT

MANIFEST_VERSION = "1.0"
SHARD_DTYPE = "uint16"


# Source streams


def get_source_streams():
    """
    Returns the two SmolLM-Corpus configs we're using, each as a streaming
    HF dataset. python-edu is deliberately excluded.
    """
    fineweb = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        split="train",
        streaming=True,
    )
    cosmopedia = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True,
    )
    return {"fineweb-edu-dedup": fineweb, "cosmopedia-v2": cosmopedia}


def mixed_text_iterator(streams: dict, fineweb_ratio: float, seed: int = 42):
    """
    Interleaves the two source streams at a fixed ratio so the final mix
    roughly matches `fineweb_ratio` (e.g. 0.8 -> 80% fineweb / 20% cosmopedia).
    Yields (text, source_name) tuples indefinitely -- caller stops it once
    the token budget is hit.
    """
    rng = np.random.default_rng(seed)
    iters = {name: iter(ds) for name, ds in streams.items()}
    weights = {"fineweb-edu-dedup": fineweb_ratio, "cosmopedia-v2": 1 - fineweb_ratio}
    names = list(weights.keys())
    probs = [weights[n] for n in names]

    exhausted = set()
    while len(exhausted) < len(names):
        choice = rng.choice(names, p=probs)
        if choice in exhausted:
            continue
        try:
            example = next(iters[choice])
        except StopIteration:
            exhausted.add(choice)
            continue
        text = example.get("text", "")
        if text:
            yield text, choice



# Manifest helpers (shared root manifest.json, one section per category)


def _load_or_init_manifest(dataset_root: str) -> dict:
    manifest_path = os.path.join(dataset_root, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dtype": SHARD_DTYPE,
        "tokenizer_checkpoint": TOKENIZER_CHECKPOINT,
        "tokenizer_fingerprint": get_tokenizer_fingerprint(),
        "categories": {},
    }


def _save_manifest(dataset_root: str, manifest: dict):
    manifest_path = os.path.join(dataset_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _shard_checksum(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()



# Shard writer


def build_shards(
    target_tokens: int,
    dataset_root: str,
    category: str = "general",
    shard_tokens: int = 268_435_456,   # 2^28, finalized project shard size
    fineweb_ratio: float = 0.8,
    sequence_length: int = 2048,
    training_config: dict | None = None,
    eos_between_docs: bool = True,
    split: str = "train",
):
    """
    Tokenizes the mixed stream and writes uint16 .bin shards of
    `shard_tokens` tokens each into dataset_root/<category>/, stopping once
    `target_tokens` total have been written. Updates (not overwrites) the
    shared manifest.json at dataset_root, writing only this category's
    section so other categories (family_law, criminal_law, sft) are left
    untouched if already present.

    Returns this category's manifest entry.
    """
    shard_dir = os.path.join(dataset_root, category)
    os.makedirs(shard_dir, exist_ok=True)

    if shard_tokens < sequence_length:
        print(
            f"WARNING: shard_tokens ({shard_tokens:,}) is smaller than "
            f"sequence_length ({sequence_length:,}) -- a shard this size "
            f"couldn't hold even one full training sequence."
        )

    tokenizer = load_smollm2_tokenizer()
    eos_id = tokenizer.eos_token_id
    streams = get_source_streams()
    text_iter = mixed_text_iterator(streams, fineweb_ratio)

    category_entry = {
        "source": "SmolLM-Corpus (fineweb-edu-dedup + cosmopedia-v2)",
        "split": split,
        "fineweb_ratio": fineweb_ratio,
        "sequence_length": sequence_length,
        "training_config": training_config or {},
        "eos_boundary_handling": (
            "eos_token_id appended after every source document before "
            "concatenation into the flat token stream; sequences sampled "
            "at train time may span multiple documents/EOS markers -- no "
            "additional packing boundary is enforced beyond the EOS token."
        ),
        "shard_tokens_target": shard_tokens,
        "target_tokens": target_tokens,
        "shards": [],
        "per_source_doc_counts": {"fineweb-edu-dedup": 0, "cosmopedia-v2": 0},
        "total_tokens_written": 0,
        "total_docs_written": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    buffer = []
    buffer_len = 0
    shard_idx = 0
    total_written = 0

    def write_shard(token_ids):
        nonlocal shard_idx
        arr = np.array(token_ids, dtype=np.uint16)
        shard_name = f"shard_{shard_idx:05d}.bin"
        arr.tofile(os.path.join(shard_dir, shard_name))
        category_entry["shards"].append({
            "file": shard_name,
            "tokens": int(arr.size),
            "checksum_sha256": _shard_checksum(arr),
        })
        shard_idx += 1

    for text, source in text_iter:
        ids = tokenizer.encode(text)
        if eos_between_docs and eos_id is not None:
            ids = ids + [eos_id]

        buffer.extend(ids)
        buffer_len += len(ids)
        total_written += len(ids)
        category_entry["per_source_doc_counts"][source] += 1
        category_entry["total_docs_written"] += 1

        while buffer_len >= shard_tokens:
            shard_ids, buffer = buffer[:shard_tokens], buffer[shard_tokens:]
            buffer_len = len(buffer)
            write_shard(shard_ids)

        if total_written >= target_tokens:
            break

    if buffer:
        write_shard(buffer)  # final, partial shard

    category_entry["total_tokens_written"] = total_written
    category_entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest = _load_or_init_manifest(dataset_root)
    manifest["categories"][category] = category_entry
    _save_manifest(dataset_root, manifest)

    print(
        f"Done. {total_written:,} tokens across {shard_idx} shard(s) "
        f"-> {shard_dir}"
    )
    return category_entry


def build_train_validation_shards(
    target_tokens: int,
    dataset_root: str,
    validation_ratio: float = 0.005,
    shard_tokens: int = 268_435_456,
    fineweb_ratio: float = 0.8,
    sequence_length: int = 2048,
    training_config: dict | None = None,
    eos_between_docs: bool = True,
    split_seed: int = 42,
):
    """Build deterministic, document-disjoint train and validation shards."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    train_target = round(target_tokens * (1.0 - validation_ratio))
    targets = {
        "general_train": train_target,
        "general_validation": target_tokens - train_target,
    }
    tokenizer = load_smollm2_tokenizer()
    eos_id = tokenizer.eos_token_id
    streams = get_source_streams()
    text_iter = mixed_text_iterator(streams, fineweb_ratio, seed=split_seed)
    buffers = {category: [] for category in targets}
    totals = {category: 0 for category in targets}
    shard_indices = {category: 0 for category in targets}
    entries = {}

    for category, split_target in targets.items():
        os.makedirs(os.path.join(dataset_root, category), exist_ok=True)
        entries[category] = {
            "source": "SmolLM-Corpus (fineweb-edu-dedup + cosmopedia-v2)",
            "split": "validation" if category.endswith("validation") else "train",
            "document_split": {
                "method": "sha256(seed + source + text)",
                "seed": split_seed,
                "validation_ratio": validation_ratio,
            },
            "fineweb_ratio": fineweb_ratio,
            "sequence_length": sequence_length,
            "training_config": training_config or {},
            "eos_boundary_handling": (
                "eos_token_id appended after every source document before "
                "concatenation; train and validation documents are disjoint"
            ),
            "shard_tokens_target": shard_tokens,
            "target_tokens": split_target,
            "shards": [],
            "per_source_doc_counts": {
                "fineweb-edu-dedup": 0,
                "cosmopedia-v2": 0,
            },
            "total_tokens_written": 0,
            "total_docs_written": 0,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def write_shard(category: str, token_ids: list[int]) -> None:
        arr = np.array(token_ids, dtype=np.uint16)
        shard_name = f"shard_{shard_indices[category]:05d}.bin"
        arr.tofile(os.path.join(dataset_root, category, shard_name))
        entries[category]["shards"].append({
            "file": shard_name,
            "tokens": int(arr.size),
            "checksum_sha256": _shard_checksum(arr),
        })
        shard_indices[category] += 1

    threshold = int(validation_ratio * (1 << 64))
    while any(totals[name] < target for name, target in targets.items()):
        try:
            text, source = next(text_iter)
        except StopIteration as error:
            raise RuntimeError("source streams ended before both split targets") from error
        digest = hashlib.sha256(
            f"{split_seed}\0{source}\0{text}".encode("utf-8")
        ).digest()
        category = (
            "general_validation"
            if int.from_bytes(digest[:8], "big") < threshold
            else "general_train"
        )
        if totals[category] >= targets[category]:
            continue

        ids = tokenizer.encode(text)
        if eos_between_docs and eos_id is not None:
            ids.append(eos_id)
        buffers[category].extend(ids)
        totals[category] += len(ids)
        entries[category]["per_source_doc_counts"][source] += 1
        entries[category]["total_docs_written"] += 1

        while len(buffers[category]) >= shard_tokens:
            write_shard(category, buffers[category][:shard_tokens])
            buffers[category] = buffers[category][shard_tokens:]

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    for category in targets:
        if buffers[category]:
            write_shard(category, buffers[category])
        entries[category]["total_tokens_written"] = totals[category]
        entries[category]["finished_at"] = finished_at

    manifest = _load_or_init_manifest(dataset_root)
    manifest["categories"].update(entries)
    _save_manifest(dataset_root, manifest)
    for category, entry in entries.items():
        print(
            f"Done. {entry['total_tokens_written']:,} tokens across "
            f"{len(entry['shards'])} shard(s) -> "
            f"{os.path.join(dataset_root, category)}"
        )
    return entries


# Validation -- read shards back and confirm integrity before calling the
# dataset "ready for pretraining"


def validate_shards(dataset_root: str, category: str = "general"):
    """
    Re-reads every shard file for `category` on disk and confirms, against
    this category's section of manifest.json:
      - every shard listed exists and has the right token count
      - every shard's checksum matches what was recorded at write time
      - no extra/unlisted shard files are sitting in the category folder
      - the sum of per-shard token counts matches total_tokens_written
    Raises AssertionError on any mismatch.
    """
    manifest = _load_or_init_manifest(dataset_root)
    assert category in manifest["categories"], f"No manifest entry for '{category}'"
    entry = manifest["categories"][category]

    shard_dir = os.path.join(dataset_root, category)
    disk_files = {
        f for f in os.listdir(shard_dir)
        if f.startswith("shard_") and f.endswith(".bin")
    }
    listed_files = {s["file"] for s in entry["shards"]}

    missing = listed_files - disk_files
    assert not missing, f"Shards listed in manifest but missing on disk: {missing}"

    extra = disk_files - listed_files
    assert not extra, f"Shard files on disk not recorded in manifest: {extra}"

    running_total = 0
    for shard in entry["shards"]:
        path = os.path.join(shard_dir, shard["file"])
        arr = np.fromfile(path, dtype=np.uint16)
        assert arr.size == shard["tokens"], (
            f"{shard['file']}: manifest says {shard['tokens']:,} tokens, "
            f"file actually has {arr.size:,}"
        )
        checksum = _shard_checksum(arr)
        assert checksum == shard["checksum_sha256"], (
            f"{shard['file']}: checksum mismatch -- file may be corrupted "
            f"or was modified after being written."
        )
        running_total += arr.size

    assert running_total == entry["total_tokens_written"], (
        f"Sum of shard tokens ({running_total:,}) != "
        f"total_tokens_written ({entry['total_tokens_written']:,})"
    )

    print(
        f"Validation OK: category='{category}', {len(entry['shards'])} shards, "
        f"{running_total:,} tokens, checksums verified."
    )
    return True



# Push to Hugging Face Hub


def push_to_hub(dataset_root: str, repo_id: str, private: bool = True):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=dataset_root,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add/update general pretraining shards (SmolLM-Corpus)",
    )
    print(f"Pushed {dataset_root} -> https://huggingface.co/datasets/{repo_id}")