import torch
import torch.nn as nn

from training.checkpoint import save_checkpoint


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
        self.value = 123

    def state_dict(self):
        return {
            "value": self.value,
        }


def test_checkpoint_is_created(tmp_path):
    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    model_config = {
        "vocab_size": 20,
        "d_model": 16,
    }

    training_config = {
        "learning_rate": 1e-3,
    }

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=10,
        micro_step=20,
        model_config=model_config,
        training_config=training_config,
    )

    assert path.exists()


def test_checkpoint_contains_required_fields(tmp_path):
    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=10,
        micro_step=20,
        model_config={"d_model": 16},
        training_config={"learning_rate": 1e-3},
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    expected_keys = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "global_step",
        "micro_step",
        "model_config",
        "training_config",
        "metadata",
    }

    assert expected_keys.issubset(
        checkpoint.keys()
    )


def test_checkpoint_preserves_steps(tmp_path):
    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=123,
        micro_step=456,
        model_config={},
        training_config={},
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    assert checkpoint["global_step"] == 123
    assert checkpoint["micro_step"] == 456


def test_checkpoint_preserves_model_weights(tmp_path):
    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    original_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=1,
        micro_step=1,
        model_config={},
        training_config={},
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    saved_state = checkpoint["model_state"]

    for key in original_state:
        assert torch.equal(
            original_state[key],
            saved_state[key],
        )