"""Deterministic data-order helpers with resumable sampler state."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class StatefulBatchSampler(Sampler[list[int]]):
    """Seeded epoch permutations that can resume at an optimizer boundary.

    Training uses ``num_workers=0`` so DataLoader prefetch cannot advance this
    state beyond the batch that the model actually consumed.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        drop_last: bool = True,
    ):
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.batch_index = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        batches = [
            indices[start : start + self.batch_size]
            for start in range(0, self.dataset_size, self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()

        for index in range(self.batch_index, len(batches)):
            self.batch_index = index + 1
            yield batches[index]

        self.epoch += 1
        self.batch_index = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_index": self.batch_index}

    def load_state_dict(self, state: dict[str, int]) -> None:
        epoch = int(state["epoch"])
        batch_index = int(state["batch_index"])
        if epoch < 0 or not 0 <= batch_index <= len(self):
            raise ValueError("invalid sampler state")
        self.epoch = epoch
        self.batch_index = batch_index
