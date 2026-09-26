import torch
from torch.utils.data import DataLoader, Dataset

from training.dataloader import DataLoader

class TinyDataset(Dataset):
    def __init__(self, size=8, seq_len=16):
        self.input_ids = torch.randint(0, 100, (size, seq_len))
        self.labels = torch.randint(0, 100, (size, seq_len))

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }
def test_dataloader_return_expected_batch():
    dataset = TinyDataset(size=8, seq_len=16)
    dataloader = DataLoader(dataset,
                            batch_size=4,)
    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (4, 16)
    assert batch["labels"].shape == (4, 16)

def test_dataloader_batch_count():
    dataset = TinyDataset(size=8, seq_len=16)
    dataloader = DataLoader(dataset,batch_size=4)

    assert len(dataloader) == 2