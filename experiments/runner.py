"""Training, validation, smoke testing, logging, and exact resume."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.shard_dataset import ShardedTokenDataset
from evaluation.validation import evaluate_validation
from experiments.config import ExperimentConfig
from experiments.data import StatefulBatchSampler
from experiments.models import build_experiment_model
from training.checkpoint import load_checkpoint
from training.optimizer import adamw
from training.scheduler import build_scheduler_cosine
from training.trainer import Trainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_smoke_test(
    config: ExperimentConfig,
    device: torch.device,
    steps: int = 3,
    sequence_length: int = 64,
    batch_size: int = 2,
    compile_model: bool = False,
    warmup_steps: int = 0,
) -> dict[str, float | int | str | bool]:
    """Train repeatedly on one deterministic batch and require finite descent."""
    seed_everything(config.seed)
    base_model = build_experiment_model(config).to(device)
    model = torch.compile(base_model) if compile_model else base_model
    optimizer = adamw(
        model,
        learning_rate=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = build_scheduler_cosine(
        optimizer,
        total_steps=max(steps + warmup_steps, 2),
        warmup_steps=1,
        min_lr_ratio=config.min_lr_ratio,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_grad_norm=config.max_grad_norm,
    )
    generator = torch.Generator().manual_seed(config.seed)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, sequence_length + 1),
        generator=generator,
    )
    batch = {
        "input_ids": tokens[:, :-1].contiguous(),
        "labels": tokens[:, 1:].contiguous(),
    }

    for _ in range(warmup_steps):
        trainer.train_step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    losses = []
    started = time.perf_counter()
    for _ in range(steps):
        result = trainer.train_step(batch)
        loss = float(result["loss"])
        if not np.isfinite(loss):
            raise RuntimeError(f"{config.name} produced non-finite smoke loss")
        losses.append(loss)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    decreased = losses[-1] < losses[0]
    if not decreased:
        raise RuntimeError(
            f"{config.name} smoke loss did not decrease: "
            f"{losses[0]:.6f} -> {losses[-1]:.6f}"
        )
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    result = {
        "name": config.name,
        "architecture": config.architecture,
        "size": config.size,
        "parameter_count": base_model.parameter_count,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "compiled": compile_model,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": decreased,
        "elapsed_seconds": elapsed,
        "tokens_per_second": steps * batch_size * sequence_length / elapsed,
        "peak_memory_bytes": peak_memory,
    }
    del trainer, optimizer, model, base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_training(
    config: ExperimentConfig,
    dataset_root: str | Path,
    output_root: str | Path,
    device: torch.device,
    resume: bool = True,
    compile_model: bool = False,
    checkpoint_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    seed_everything(config.seed)
    output_dir = Path(output_root) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"

    train_dataset = ShardedTokenDataset(
        str(dataset_root), "general_train", config.sequence_length
    )
    validation_dataset = ShardedTokenDataset(
        str(dataset_root), "general_validation", config.sequence_length
    )
    train_sampler = StatefulBatchSampler(
        len(train_dataset),
        config.micro_batch_size,
        config.seed,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    base_model = build_experiment_model(config).to(device)
    model = base_model
    if compile_model:
        model = torch.compile(model)
    optimizer = adamw(
        model,
        learning_rate=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = build_scheduler_cosine(
        optimizer,
        config.total_optimizer_steps,
        config.warmup_steps,
        config.min_lr_ratio,
    )
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        device,
        config.max_grad_norm,
        config.gradient_accumulation_steps,
    )
    prior_train_seconds = 0.0
    if resume and checkpoint_path.exists():
        checkpoint = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        trainer.global_step = checkpoint["global_step"]
        trainer.micro_step = checkpoint["micro_step"]
        runtime = checkpoint.get("runtime_state", {})
        if runtime:
            train_sampler.load_state_dict(runtime["sampler"])
            random.setstate(runtime["python_rng"])
            np.random.set_state(runtime["numpy_rng"])
            torch.set_rng_state(runtime["torch_rng"])
            if device.type == "cuda" and runtime.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(runtime["cuda_rng"])
            prior_train_seconds = float(runtime.get("train_seconds", 0.0))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(train_loader)
    train_seconds = prior_train_seconds
    running_loss = 0.0
    running_micro_steps = 0

    def write_event(event: dict[str, object]) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    def save() -> None:
        runtime_state = {
            "sampler": train_sampler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "train_seconds": train_seconds,
        }
        trainer.save_checkpoint(
            checkpoint_path,
            base_model.config,
            asdict(config),
            metadata={"experiment": config.name},
            runtime_state=runtime_state,
        )
        if checkpoint_callback is not None:
            checkpoint_callback()

    def validate() -> dict[str, float | int]:
        metrics = evaluate_validation(
            model,
            validation_loader,
            device,
            max_batches=config.validation_batches,
        )
        event = {
            "event": "validation",
            "optimizer_step": trainer.global_step,
            "tokens_seen": trainer.global_step * config.tokens_per_optimizer_step,
            **metrics,
        }
        write_event(event)
        return metrics

    if trainer.global_step == 0:
        validate()

    while trainer.global_step < config.total_optimizer_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        started = time.perf_counter()
        result = trainer.train_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - started
        loss = float(result["loss"])
        if not np.isfinite(loss):
            save()
            raise RuntimeError(
                f"non-finite loss at optimizer step {trainer.global_step}"
            )
        running_loss += loss
        running_micro_steps += 1
        if not result["optimizer_step"]:
            continue

        step = trainer.global_step
        if step % config.log_every_steps == 0 or step == 1:
            tokens_seen = step * config.tokens_per_optimizer_step
            event = {
                "event": "train",
                "optimizer_step": step,
                "micro_step": trainer.micro_step,
                "tokens_seen": tokens_seen,
                "loss": running_loss / running_micro_steps,
                "grad_norm": float(result["grad_norm"]),
                "learning_rate": scheduler.get_last_lr()[0],
                "train_seconds": train_seconds,
                "tokens_per_second": tokens_seen / train_seconds,
            }
            write_event(event)
            running_loss = 0.0
            running_micro_steps = 0
        if step % config.eval_every_steps == 0:
            validate()
        if step % config.checkpoint_every_steps == 0:
            save()

    final_validation = validate()
    save()
    tokens_seen = trainer.global_step * config.tokens_per_optimizer_step
    summary = {
        "name": config.name,
        "architecture": config.architecture,
        "size": config.size,
        "parameter_count": base_model.parameter_count,
        "target_train_tokens": config.target_train_tokens,
        "actual_train_tokens": tokens_seen,
        "optimizer_steps": trainer.global_step,
        "train_seconds": train_seconds,
        "tokens_per_second": tokens_seen / train_seconds,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        **final_validation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
