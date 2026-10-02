"""
data/shard_dataset.py

The missing piece between data/general_prep.py's shard output and
training/dataloader.py's create_dataloader(): a torch.utils.data.Dataset
that reads pretokenized uint16 .bin shards and yields fixed-length causal-LM
training examples.

Why this exists: training/dataloader.py (create_dataloader) wraps any
Dataset with a DataLoader, but no Dataset reading our shard format existed
yet.

Output contract (must match training/batch.py + training/loss.py):
    {"input_ids": LongTensor[sequence_length], "labels": LongTensor[sequence_length]}
  where labels = input_ids shifted by one position. causal_language_model_loss
  does NOT shift internally -- it expects logits and labels already aligned --
  so the shift has to happen here, once, at data-loading time.
  DataLoader's default collate stacks these into
  {"input_ids": [B, T], "labels": [B, T]}, satisfying README section 14's
  shared batch format once cast to torch.long (done below).

Memory behavior: shards are opened with np.memmap, never loaded fully into
RAM, per the project data spec ("use memory mapping and prefetching during
training; do not load an entire shard into GPU memory"). Each __getitem__
call reads only the sequence_length+1 tokens it needs.

Windowing: samples are drawn from non-overlapping windows within each shard
(standard nanoGPT-style packing). A window may span multiple source
documents/EOS markers -- see manifest["categories"][category]["eos_boundary_handling"]
for how document boundaries are marked (EOS token only, no hard packing
boundary). Leftover tokens at the end of a shard that don't fill one full
window are dropped; this is a few thousand tokens per shard at most and not
worth the added complexity of cross-shard windows.

Shuffling: this class does NOT shuffle itself. Pass shuffle=True to
create_dataloader (already its default) -- DataLoader shuffles the index
order each epoch, which is sample-level shuffling across the whole dataset,
not just within a shard.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedTokenDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        category: str,
        sequence_length: int = 2048,
    ):
        self.dataset_root = dataset_root
        self.category = category
        self.sequence_length = sequence_length
        self._window = sequence_length + 1  # +1 so labels = input shifted by one

        manifest_path = os.path.join(dataset_root, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        if category not in manifest["categories"]:
            raise KeyError(
                f"No manifest entry for category '{category}' in {manifest_path}. "
                f"Available categories: {list(manifest['categories'].keys())}"
            )

        self.entry = manifest["categories"][category]
        shard_dir = os.path.join(dataset_root, category)

        # Memory-map every shard (does not load into RAM).
        self._shards = []
        for shard in self.entry["shards"]:
            path = os.path.join(shard_dir, shard["file"])
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            if arr.size != shard["tokens"]:
                raise ValueError(
                    f"{shard['file']}: manifest says {shard['tokens']:,} tokens, "
                    f"file on disk has {arr.size:,}. Re-run validate_shards() "
                    f"before training on this dataset."
                )
            self._shards.append(arr)

        # Flat index of (shard_idx, start_offset) for every non-overlapping
        # window, so __getitem__ is O(1) and shuffling is just index
        # shuffling (handled by DataLoader, not here).
        self._index = []
        for shard_idx, arr in enumerate(self._shards):
            n_windows = arr.size // self._window
            for w in range(n_windows):
                self._index.append((shard_idx, w * self._window))

        if not self._index:
            raise ValueError(
                f"No shard in category '{category}' holds a full "
                f"sequence_length={sequence_length} example (window needs "
                f"{self._window} tokens). Check shard_tokens vs "
                f"sequence_length in the manifest."
            )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        shard_idx, offset = self._index[idx]
        arr = self._shards[shard_idx]
        window = arr[offset: offset + self._window]  # uint16, len = sequence_length+1

        # int64 because torch has no uint16 tensor type, and the shared
        # batch contract requires torch.long.
        window = torch.from_numpy(window.astype(np.int64))

        input_ids = window[:-1]
        labels = window[1:]
        return {"input_ids": input_ids, "labels": labels}