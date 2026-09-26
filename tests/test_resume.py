import torch
import torch.nn as nn

from training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(20, 16)
        self.lm_head = nn.Linear(16, 20)

    def forward(self, input_ids):
        return self.lm_head(
            self.embedding(input_ids)
        )


class DummyScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def state_dict(self):
        return {
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        self.step_count = state["step_count"]


def test_resume_restores_model_state(tmp_path):
    torch.manual_seed(42)

    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    # Change model weights so we know exactly what is being saved.
    input_ids = torch.randint(
        0,
        20,
        (2, 8),
    )

    output = model(input_ids)
    loss = output.mean()

    loss.backward()
    optimizer.step()
    scheduler.step()

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
        global_step=10,
        micro_step=20,
        model_config={"d_model": 16},
        training_config={"learning_rate": 1e-3},
    )

    # Create completely new objects.
    new_model = TinyLM()

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=1e-3,
    )

    new_scheduler = DummyScheduler()

    checkpoint = load_checkpoint(
        path=path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        device=torch.device("cpu"),
    )

    for key, value in new_model.state_dict().items():
        assert torch.equal(
            value,
            original_state[key],
        )

    assert checkpoint["global_step"] == 10
    assert checkpoint["micro_step"] == 20
    assert new_scheduler.step_count == 1


def test_resume_restores_optimizer_state(tmp_path):
    torch.manual_seed(42)

    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = DummyScheduler()

    input_ids = torch.randint(
        0,
        20,
        (2, 8),
    )

    loss = model(input_ids).mean()
    loss.backward()
    optimizer.step()

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

    new_model = TinyLM()

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=1e-3,
    )

    new_scheduler = DummyScheduler()

    load_checkpoint(
        path=path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        device=torch.device("cpu"),
    )

    assert len(new_optimizer.state) == len(
        optimizer.state
    )