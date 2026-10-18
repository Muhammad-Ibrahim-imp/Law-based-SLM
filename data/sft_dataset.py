"""PyTorch Dataset over the packed SFT shards written by ``data/sft_prep.py``.

Each item is one packed row of ``sequence_length`` tokens::

    {"input_ids": LongTensor[T], "labels": LongTensor[T]}

with ``labels[t] = input_ids[t+1]`` where ``mask[t+1] == 1`` and ``-100``
everywhere else (prompt, role headers, padding, and the last position). This
matches the shared batch contract and ``cross_entropy``'s default
``ignore_index=-100``; check that ``training/loss.py`` does not override it.

Unlike ``ShardedTokenDataset`` there is no window stride: rows are stored
already aligned, so every stored row is one example.

Several sources (e.g. smol-smoltalk and the law SFT data) can be combined by
passing several ``(source, split)`` pairs; their blend is set at prep time via
each source's ``max_tokens``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.shard_writer import read_manifest

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    def __init__(self, dataset_root, sources: Sequence[tuple[str, str]],
                 sequence_length: int | None = None):
        """``sequence_length=None`` takes it from the manifest (all sources
        must have been packed at the same length); pass a value to assert it
        matches the training config."""
        if not sources:
            raise ValueError("sources is empty")
        root = Path(dataset_root)
        sft = (read_manifest(root).get("categories", {}).get("sft", {})
               .get("sources", {}))
        self._shards: list[tuple[np.memmap, np.memmap]] = []
        self._row_offsets = [0]
        for source, split in sources:
            entry = sft.get(source, {}).get(split)
            if not entry:
                raise ValueError(f"no manifest entry for sft/{source}/{split}")
            if entry.get("status") != "complete":
                raise ValueError(f"sft/{source}/{split} is {entry.get('status')!r}, not complete")
            if sequence_length is None:
                sequence_length = entry["sequence_length"]
            if entry["sequence_length"] != sequence_length:
                raise ValueError(f"sft/{source}/{split} was packed at "
                                 f"{entry['sequence_length']}, not {sequence_length}")
            self.sequence_length = sequence_length
            d = root / "sft" / source / split
            for s in entry["shards"]:
                n = s["num_tokens"]
                tok_p, mask_p = d / s["files"]["tokens"], d / s["files"]["mask"]
                # Cheap size check only; full validation is validate_sft_split().
                if tok_p.stat().st_size != 2 * n or mask_p.stat().st_size != n:
                    raise ValueError(f"size mismatch in {tok_p.name} / {mask_p.name}")
                if n % sequence_length:
                    raise ValueError(f"{tok_p} is not a whole number of rows")
                self._shards.append((np.memmap(tok_p, dtype=np.uint16, mode="r"),
                                     np.memmap(mask_p, dtype=np.uint8, mode="r")))
                self._row_offsets.append(self._row_offsets[-1] + n // sequence_length)

    def __len__(self) -> int:
        return self._row_offsets[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        k = int(np.searchsorted(self._row_offsets, idx, side="right")) - 1
        r = idx - self._row_offsets[k]
        L = self.sequence_length
        tok, mask = self._shards[k]
        ids = torch.from_numpy(tok[r * L:(r + 1) * L].astype(np.int64))
        m = torch.from_numpy(mask[r * L:(r + 1) * L].astype(np.bool_))
        labels = torch.full((L,), IGNORE_INDEX, dtype=torch.long)
        labels[:-1] = torch.where(m[1:], ids[1:], torch.tensor(IGNORE_INDEX))
        return {"input_ids": ids, "labels": labels}
