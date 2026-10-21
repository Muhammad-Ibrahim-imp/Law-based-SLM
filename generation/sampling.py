"""Choose the next token from one row of vocabulary logits."""

from __future__ import annotations

import torch

from common.interfaces import GenerationConfig


def sample_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row.

    ``logits`` has shape ``[batch, vocab]``. The result has shape ``[batch, 1]``.
    Temperature 0 is greedy argmax. A positive temperature divides the logits,
    then optional top-k and top-p filters, then samples once.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if config.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if config.top_k is not None and config.top_k <= 0:
        raise ValueError("top_k must be positive")
    if config.top_p is not None and not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if config.temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = logits / config.temperature
    filtered = _apply_top_k(filtered, config.top_k)
    filtered = _apply_top_p(filtered, config.top_p)
    probabilities = torch.softmax(filtered, dim=-1)
    if generator is not None and probabilities.device.type != generator.device.type:
        chosen = torch.multinomial(probabilities.cpu(), num_samples=1, generator=generator)
        return chosen.to(device=probabilities.device)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def _apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(ordered, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    ordered = ordered.masked_fill(remove, float("-inf"))
    return torch.gather(ordered, -1, torch.argsort(indices, dim=-1))
