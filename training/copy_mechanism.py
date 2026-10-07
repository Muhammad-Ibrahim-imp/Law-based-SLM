"""Pointer-generator mixture shared by every architecture.

The architecture still predicts a distribution over the vocabulary. A second
distribution copies a token from the prefix already seen at that position.
The two distributions are mixed with a learned gate. The gate starts near
"generate", so training begins close to ordinary next-token loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CopyLM(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        d_model = model.config.d_model
        self.generate_gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.generate_gate.weight)
        nn.init.constant_(self.generate_gate.bias, 2.0)

    @property
    def config(self):
        return self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        labels: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        logits, hidden = self._encode(input_ids, position_offset)
        if labels is None:
            return logits
        return mixture_nll(
            logits,
            hidden,
            input_ids,
            labels,
            self.generate_gate,
            reduction=reduction,
        )

    def _encode(
        self, input_ids: torch.Tensor, position_offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, position_offset=position_offset)
        hidden = self.model.final_norm(hidden)
        return self.model.lm_head(hidden), hidden


def mixture_nll(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    context_ids: torch.Tensor,
    labels: torch.Tensor,
    generate_gate: nn.Linear,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Negative log of p_gen * P_vocab + (1 - p_gen) * P_copy."""
    log_vocab = F.log_softmax(logits.float(), dim=-1)
    safe_labels = labels.clamp_min(0)
    token_log_prob = log_vocab.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    p_generate = torch.sigmoid(generate_gate(hidden.float())).squeeze(-1)

    states = hidden.float()
    scores = torch.matmul(states, states.transpose(-1, -2))
    scores = scores * (states.shape[-1] ** -0.5)
    length = scores.shape[-1]
    causal = torch.ones(
        length, length, device=scores.device, dtype=torch.bool
    ).tril()
    attention = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    matches = context_ids.unsqueeze(1).eq(safe_labels.unsqueeze(2)) & causal
    copy_prob = (attention * matches.to(attention.dtype)).sum(dim=-1)

    generate_term = torch.log(p_generate.clamp_min(1e-12)) + token_log_prob
    copy_term = torch.where(
        copy_prob > 0,
        torch.log((1.0 - p_generate).clamp_min(1e-12))
        + torch.log(copy_prob.clamp_min(1e-12)),
        torch.full_like(copy_prob, float("-inf")),
    )
    nll = -torch.logaddexp(generate_term, copy_term)
    valid = labels.ne(ignore_index)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.sum() / valid.sum().clamp_min(1)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def is_copy_model(model: nn.Module) -> bool:
    raw = getattr(model, "_orig_mod", model)
    return isinstance(raw, CopyLM)
