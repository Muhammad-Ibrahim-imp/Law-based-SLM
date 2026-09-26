import torch
from torch.utils.data import DataLoader

def create_dataloader(dataset,
               batch_size,
               shuffle=True,  # Randomizes sample order each epoch to prevent memorization
               num_workers=0,  # Number of CPU subprocesses for parallel data loading
               pin_memory=False,  # Speeds up CPU-to-GPU data transfers via pinned RAM
               drop_last=True,  # Drops the incomplete final batch to keep batch size uniform
               ):

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      drop_last=drop_last,)