import torch
import torch.nn as nn
from torch.utils.data import Dataset

from training.dataloader import create_dataloader
from training.trainer import Trainer


class TinyDataset(Dataset):
    def __init__(self, size=8, seq_len=16, vocab_size=20):
        self.input_ids = torch.randint(
            0,
            vocab_size,
            (size, seq_len),
        )

        self.labels = torch.randint(
            0,
            vocab_size,
            (size, seq_len),
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }


class TinyLM(nn.Module):
    def __init__(self, vocab_size=20, d_model=16):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.lm_head(x)


class DummyScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1

    def state_dict(self):
        return {
            "step_calls": self.step_calls,
        }

    def load_state_dict(self, state):
        self.step_calls = state["step_calls"]


def test_full_training_pipeline():
    torch.manual_seed(42)

    dataset = TinyDataset(
        size=8,
        seq_len=16,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
    )

    for batch in dataloader:
        result = trainer.train_step(batch)

        assert "loss" in result
        assert "global_step" in result
        assert "micro_step" in result
        assert "optimizer_step" in result

    # 8 examples / batch size 2 = 4 microbatches.
    assert trainer.micro_step == 4

    # 4 microbatches / accumulation 2 = 2 optimizer steps.
    assert trainer.global_step == 2

    # Scheduler advances once per optimizer step.
    assert scheduler.step_calls == 2


def test_training_pipeline_updates_model():
    torch.manual_seed(42)

    dataset = TinyDataset(
        size=4,
        seq_len=16,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    model = TinyLM()

    initial_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        gradient_accumulation_steps=1,
    )

    for batch in dataloader:
        trainer.train_step(batch)

    changed = any(
        not torch.equal(
            parameter,
            initial_state[name],
        )
        for name, parameter in model.named_parameters()
    )

    assert changed