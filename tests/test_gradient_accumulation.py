import torch
import torch.nn as nn

from training.trainer import Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size=20, d_model=16):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.lm_head(x)


class SpyScheduler:
    """Minimal scheduler used to verify scheduler.step() calls."""

    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def make_trainer(accumulation_steps=2):
    model = TinyLM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    scheduler = SpyScheduler()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
        gradient_accumulation_steps=accumulation_steps,
    )

    return trainer, model, optimizer, scheduler


def make_batch():
    input_ids = torch.randint(
        0,
        20,
        (2, 8),
    )

    labels = torch.randint(
        0,
        20,
        (2, 8),
    )

    return input_ids, labels


def test_no_optimizer_step_before_accumulation_boundary():
    trainer, model, optimizer, scheduler = make_trainer(
        accumulation_steps=2
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    input_ids, labels = make_batch()

    result = trainer.train_step(input_ids, labels)

    assert result["micro_step"] == 1
    assert result["global_step"] == 0
    assert result["optimizer_step"] is False
    assert result["grad_norm"] is None
    assert scheduler.step_calls == 0

    # Gradients should exist because backward() already happened.
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    # Parameters must not have changed yet.
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_optimizer_step_occurs_at_accumulation_boundary():
    trainer, model, optimizer, scheduler = make_trainer(
        accumulation_steps=2
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    input_ids, labels = make_batch()

    trainer.train_step(input_ids, labels)

    result = trainer.train_step(input_ids, labels)

    assert result["micro_step"] == 2
    assert result["global_step"] == 1
    assert result["optimizer_step"] is True

    assert result["grad_norm"] is not None
    assert scheduler.step_calls == 1

    # At least one parameter should have changed.
    changed = any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )

    assert changed


def test_gradients_are_cleared_after_optimizer_step():
    trainer, model, optimizer, scheduler = make_trainer(
        accumulation_steps=2
    )

    input_ids, labels = make_batch()

    trainer.train_step(input_ids, labels)

    # Gradients exist during accumulation.
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    trainer.train_step(input_ids, labels)

    # Trainer calls zero_grad(set_to_none=True)
    # after the optimizer update.
    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_scheduler_steps_once_per_optimizer_step():
    trainer, model, optimizer, scheduler = make_trainer(
        accumulation_steps=2
    )

    for _ in range(4):
        input_ids, labels = make_batch()
        trainer.train_step(input_ids, labels)

    # 4 microsteps / 2 accumulation steps = 2 optimizer steps.
    assert trainer.micro_step == 4
    assert trainer.global_step == 2
    assert scheduler.step_calls == 2