"""Validated JSON configuration for one architecture/size experiment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    architecture: str
    size: str
    expected_parameters: int
    model: dict[str, Any]
    seed: int
    target_train_tokens: int
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    max_grad_norm: float
    log_every_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int
    validation_batches: int | None
    copy_mechanism: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in {"gqa", "lfm25", "qwen3", "gemma3"}:
            raise ValueError(f"unsupported architecture: {self.architecture!r}")
        if self.size not in {"25m", "50m", "100m"}:
            raise ValueError(f"unsupported size: {self.size!r}")
        positive = {
            "expected_parameters": self.expected_parameters,
            "seed": self.seed,
            "target_train_tokens": self.target_train_tokens,
            "sequence_length": self.sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "eps": self.eps,
            "max_grad_norm": self.max_grad_norm,
            "log_every_steps": self.log_every_steps,
            "eval_every_steps": self.eval_every_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"values must be positive: {', '.join(invalid)}")
        if self.sequence_length > 8192:
            raise ValueError("sequence_length cannot exceed 8192")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.validation_batches is not None and self.validation_batches <= 0:
            raise ValueError("validation_batches must be positive or null")

    @property
    def tokens_per_micro_step(self) -> int:
        return self.sequence_length * self.micro_batch_size

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.tokens_per_micro_step * self.gradient_accumulation_steps

    @property
    def total_optimizer_steps(self) -> int:
        return math.ceil(self.target_train_tokens / self.tokens_per_optimizer_step)

    @property
    def actual_train_tokens(self) -> int:
        return self.total_optimizer_steps * self.tokens_per_optimizer_step

    @property
    def warmup_steps(self) -> int:
        return max(1, round(self.total_optimizer_steps * self.warmup_ratio))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["betas"] = tuple(payload["betas"])
    config = ExperimentConfig(**payload)
    if config.name != config_path.stem:
        raise ValueError(
            f"config name {config.name!r} does not match filename {config_path.stem!r}"
        )
    return config
