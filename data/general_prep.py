"""
data/general_prep.py

Reusable logic for building the "general" pretraining slice (SmolLM-Corpus)
for the law-SLM project. Target per README: 5.0B tokens.

Called from scripts/prepare_general_data.py (thin entry point), per README
section 17: reusable implementation belongs in the appropriate package, not
in scripts/.

Shard/format spec:
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


KNOWN QUIRKS (documented, deliberately NOT fixed)
STATUS: the general shards were built with this code and pretraining is
already reading them. The behaviour of this file is therefore frozen so the
data stays reproducible from the code that produced it. The quirks below are
recorded instead of fixed here; the fixes go into the legal/SFT prep code.
Do not regenerate or modify dataset/general/ while training is running.

Affecting the data that was written:

 1. Mix ratio is per DOCUMENT, not per token. `fineweb_ratio=0.8` means 80%
    of the documents drawn come from fineweb-edu-dedup. If the two sources
    differ in average document length, the token mix is not 80/20. The
    manifest records per-source doc counts only, so the true token ratio
    has to be estimated (avg tokens/doc per source x doc counts).

 2. The final shard is partial. It holds whatever is left once the target
    is reached (target_tokens mod shard_tokens; ~168M tokens for a 5.0B
    target = 18 full shards + 1 partial). It does not give 512 optimizer
    steps and its length is not guaranteed to be a multiple of
    sequence_length. The training loader must handle or drop it.

 3. Documents are cut across shard boundaries. Shards are filled to exactly
    shard_tokens, so a document can start in one shard and finish in the
    next. Only the EOS token marks document ends.

 4. total_tokens_written overshoots target_tokens by up to one document,
    because the budget check runs after a whole document is added. The
    manifest records the true total.

 5. Manifest timestamps are LOCAL time with a misleading "Z" suffix.
    `started_at` / `finished_at` use time.strftime() without time.gmtime(),
    so they are not UTC despite the "Z".

 6. `sequence_length` in the manifest is metadata only. Nothing in this file
    uses it to shape the data; the training config decides the real value.
    The default here (2,048) is confirmed with Member B as the sequence
    length used in training.

Affecting how the run behaves (not the data):

 7. No resume. manifest.json is written once, at the very end. A crash
    mid-run leaves shard files with no manifest entry and a rerun starts
    from zero.

 8. High memory use. `buffer` is a Python list of ints, not a NumPy array:
    roughly 8-10 GB RAM near a full shard, and each slice copies it.

 9. Manifest writes are not concurrency-safe. Load -> modify -> save with
    no lock: if two prep scripts finish at the same moment, the last writer
    overwrites the other's section.

10. The shard_tokens < sequence_length check only prints a warning; the run
    continues.

11. The source name yielded by mixed_text_iterator is a NumPy string
    (np.str_), not a plain str. It works as a dict key, but is not a str.

Validation and upload:

12. validate_shards() proves the files match what was written, not that
    what was written is correct. It checks counts, checksums and token
    range, but does not decode any text. Its checks are `assert`s, which
    are skipped entirely if Python runs with -O.

13. push_to_hub() uploads the WHOLE dataset root (every category folder
    present), although its commit message says "general".

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
        "fineweb-edu-dedup", # real-world text
        split="train",
        streaming=True,
    )
    cosmopedia = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",  # flawless formating synthetic text
        split="train",
        streaming=True,
    )
    return {"fineweb-edu-dedup": fineweb, "cosmopedia-v2": cosmopedia}


def mixed_text_iterator(streams: dict, fineweb_ratio: float, seed: int = 42):
    """
    Interleaves the two source streams at a fixed ratio so the final mix
    roughly matches `fineweb_ratio` (e.g. 0.8 -> 80% fineweb / 20% cosmopedia).
    Yields (text, source_name) tuples until both streams run dry, but the caller stops it once
    the token budget is hit.

    Quirks 1 and 11 (see module docstring): the ratio is per document, and
    source_name is a NumPy string.
    """
    rng = np.random.default_rng(seed) #random number generator for interleaving

    # turning each dataset stream into an iterator and storing with name in a dict
    iters = {name: iter(ds) for name, ds in streams.items()}

    # setting weights that determine the ratio PER DOC between the 2 datasets
    weights = {"fineweb-edu-dedup": fineweb_ratio, "cosmopedia-v2": 1 - fineweb_ratio}

    names = list(weights.keys()) # ["fineweb-edu-dedup", "cosmopedia-v2"]

    probs = [weights[n] for n in names] # e.g. [0.8, 0.2] (default)

    exhausted = set()
    while len(exhausted) < len(names):
        chosen_ds = rng.choice(names, p=probs) #dataset chosen randomly (rng.choice is the NumPy method name)
        if chosen_ds in exhausted:
            continue
        try:
            example = next(iters[chosen_ds]) #extraxt next example from dataset
        except StopIteration: #if nothing more left to iterate
            exhausted.add(chosen_ds)
            continue
        text = example.get("text", "") #extract text from chosen dataset if available, otherwise ""
        if text:
            yield text, chosen_ds


# Manifest helpers (shared root manifest.json, one section per category)

def _load_or_init_manifest(dataset_root: str) -> dict:
    manifest_path = os.path.join(dataset_root, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as file:
            return json.load(file) #load previously written manifest
    return { #initialise new manifest if doesn't exist
        "manifest_version": MANIFEST_VERSION,
        "dtype": SHARD_DTYPE,
        "tokenizer_checkpoint": TOKENIZER_CHECKPOINT,
        "tokenizer_fingerprint": get_tokenizer_fingerprint(),
        "categories": {},
    }


def _save_manifest(dataset_root: str, manifest: dict):
    # Quirk 9: no lock -- not safe if two prep scripts save at the same time.
    manifest_path = os.path.join(dataset_root, "manifest.json")
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)


def _shard_checksum(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


# Shard writer

def build_shards(
    target_tokens: int,
    dataset_root: str,
    category: str = "general",
    shard_tokens: int = 268_435_456,   # 2^28, finalized project shard size
    fineweb_ratio: float = 0.8,
    sequence_length: int = 2048,       # Quirk 6: metadata only, see module docstring
    training_config: dict | None = None,
    eos_between_docs: bool = True, #End of Sequence
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
    os.makedirs(shard_dir, exist_ok=True) #True means no error if already exists

    # Quirk 10: warns only, does not stop the run.
    if shard_tokens < sequence_length:
        print(
            f"WARNING: shard_tokens ({shard_tokens:,}) is smaller than "
            f"sequence_length ({sequence_length:,}) -- a shard this size "
            f"couldn't hold even one full training sequence."
        ) # :, formats int with commas e.g. 268,435,456

    tokenizer = load_smollm2_tokenizer()
    eos_id = tokenizer.eos_token_id #unique numerical ID assigned to the End-of-Sequence token
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
        # Quirk 5: local time, despite the "Z" suffix.
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    buffer = []  #list that stores tokens for next shard (Quirk 8: Python list, memory-heavy)
    buffer_len = 0
    shard_idx = 0
    total_written = 0

    def write_shard(token_ids): #nested definition

        nonlocal shard_idx  #local to build_shards()
        arr = np.array(token_ids, dtype=np.uint16)  #unsigned 16-bit integers for compact storage
        shard_name = f"shard_{shard_idx:05d}.bin"  # 5-digit decimal number padded with zeros
        arr.tofile(os.path.join(shard_dir, shard_name))
        # writes raw bytes with no header. the file doesn't know its own dtype or length
        # so manifest has to record dtype and token count.

        #metadata dict entry in manifest about new shard
        category_entry["shards"].append({
            "file": shard_name,
            "tokens": int(arr.size), # defensive: guarantees a plain int for JSON
            "checksum_sha256": _shard_checksum(arr),
        })
        shard_idx += 1

    #iterating through docs in both dataset streams
    for text, source in text_iter:
        ids = tokenizer.encode(text)  #tokenizing the text from current dataset doc
        if eos_between_docs and eos_id is not None:
            ids = ids + [eos_id] #adding end-ofsequence token after each doc

        buffer.extend(ids) #add tokens from current doc to buffer
        buffer_len += len(ids)
        total_written += len(ids)
        category_entry["per_source_doc_counts"][source] += 1
        category_entry["total_docs_written"] += 1


        while buffer_len >= shard_tokens: #while number of tokens in buffer reaches shard size
            #move the first shard_tokens (2^28 = 268,435,456) tokens into a new shard
            #Quirk 3: this cuts whichever document straddles the boundary
            shard_ids, buffer = buffer[:shard_tokens], buffer[shard_tokens:]
            buffer_len = len(buffer)
            write_shard(shard_ids)
        #while is used because one document could theoretically cause the buffer
        # to contain more than one complete shard, requiring multiple shards to be written

        # Quirk 4: can exceed slightly, as no doc is taken incompletely
        if total_written >= target_tokens:
            break

    if buffer:
        write_shard(buffer)  # final, partial shard (Quirk 2)

    category_entry["total_tokens_written"] = total_written
    category_entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")  # Quirk 5

    # Quirk 7: this is the only point the manifest is written -- no resume.
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
      - every token ID is inside the vocab range
      - every shard's checksum matches what was recorded at write time
      - no extra/unlisted shard files are sitting in the category folder
      - the sum of per-shard token counts matches total_tokens_written
    Raises AssertionError on any mismatch.

    Quirk 12 (see module docstring): read-only, does not decode text, and
    the asserts are skipped under `python -O`.
    """
    manifest = _load_or_init_manifest(dataset_root)
    assert category in manifest["categories"], f"No manifest entry for '{category}'"
    entry = manifest["categories"][category] #metadata of specific category

    shard_dir = os.path.join(dataset_root, category)
    disk_files = { #set of all shard files from chosen data dir
        file for file in os.listdir(shard_dir)
        if file.startswith("shard_") and file.endswith(".bin")
    }
    listed_files = {s["file"] for s in entry["shards"]} #set of all shard files in manifest

    #ensuring above two sets are same, and raising error with suitable message if not:

    missing = listed_files - disk_files #set diff
    assert not missing, f"Shards listed in manifest but missing on disk: {missing}"

    extra = disk_files - listed_files #set diff
    assert not extra, f"Shard files on disk not recorded in manifest: {extra}"

    running_total = 0
    #iterating through each shard's metadata dictionary in manifest
    for shard in entry["shards"]:
        path = os.path.join(shard_dir, shard["file"])
        arr = np.fromfile(path, dtype=np.uint16)
        #ensure all tokens are within vocab range
        assert arr.max() < 49152, (
            f"{shard['file']}: token ID {int(arr.max())} is outside the "
            f"vocab range (must be < 49,152)"
        )
        #compare actual token count with recorded token count
        assert arr.size == shard["tokens"], (
            f"{shard['file']}: manifest says {shard['tokens']:,} tokens, "
            f"file actually has {arr.size:,}"
        )
        checksum = _shard_checksum(arr)
        #compare newly calculated checksum with recorded checksum
        assert checksum == shard["checksum_sha256"], (
            f"{shard['file']}: checksum mismatch -- file may be corrupted "
            f"or was modified after being written."
        )
        running_total += arr.size

    # compare token count with recorded token count
    assert running_total == entry["total_tokens_written"], (
        f"Sum of shard tokens ({running_total:,}) != "
        f"total_tokens_written ({entry['total_tokens_written']:,})"
    )

    print(
        f"Validation OK: category='{category}', {len(entry['shards'])} shards, "
        f"{running_total:,} tokens, checksums verified."
    )
    return True


# Push to Hugging Face Hub (not in use for now)

def push_to_hub(dataset_root: str, repo_id: str, private: bool = True):
    # Quirk 13: uploads the whole dataset root, not just general/.
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