"""Continue a token sequence one new token at a time when a cache is available."""

from __future__ import annotations

import torch

from common.interfaces import GenerationConfig
from generation.sampling import sample_token


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    generation_config: GenerationConfig,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append tokens until the limit, the context limit, or EOS.

    Models that implement ``build_cache`` run the prompt once, store keys,
    values, and any convolution state, then run each new token by itself.
    Other models recompute the whole prefix on every token.

    Returns a tensor of shape ``[batch, prompt_length + new_tokens]``.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[1] == 0:
        raise ValueError("prompt is empty")
    if generation_config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    max_seq_len = model.config.max_seq_len
    prompt_length = input_ids.shape[1]
    if prompt_length >= max_seq_len:
        raise ValueError("prompt already fills max_seq_len")

    room = max_seq_len - prompt_length
    steps = min(generation_config.max_new_tokens, room)
    sequences = input_ids
    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )
    cache = model.build_cache(input_ids.shape[0]) if hasattr(model, "build_cache") else None
    if cache is None:
        pending = None
    else:
        pending = model(input_ids, cache=cache)

    for step in range(steps):
        if pending is None:
            logits = model(sequences)[:, -1, :]
        else:
            logits = pending[:, -1, :]
            pending = None
        next_id = sample_token(logits, generation_config, generator)
        if eos_id is not None:
            eos = torch.full_like(next_id, eos_id)
            next_id = torch.where(finished.unsqueeze(1), eos, next_id)
            finished = finished | next_id.squeeze(1).eq(eos_id)
        sequences = torch.cat([sequences, next_id], dim=1)
        if eos_id is not None and bool(finished.all()):
            break
        if step + 1 == steps or sequences.shape[1] >= max_seq_len:
            break
        if cache is not None:
            pending = model(next_id, cache=cache)
    return sequences
