"""Held-out causal-language-model loss and perplexity."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    completed_batches = 0

    for batch in batches:
        if max_batches is not None and completed_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type in {"cuda", "cpu"},
        ):
            logits = model(input_ids)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
        token_count = labels.numel()
        total_loss += float(loss_sum)
        total_tokens += token_count
        completed_batches += 1

    if total_tokens == 0:
        raise ValueError("validation loader produced no tokens")
    mean_loss = total_loss / total_tokens
    return {
        "validation_loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 100 else math.inf,
        "validation_tokens": total_tokens,
        "validation_batches": completed_batches,
    }
