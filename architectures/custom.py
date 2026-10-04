"""Placeholder for a future custom decoder.

Do not put a real design here yet. `build_model("custom")` fails on purpose
until `DecoderLM` is implemented.

Expected interface, matching the live GQA decoder:

- Constructor: `DecoderLM(config)` where `config` is `CustomConfig`.
- `forward(input_ids, position_offset=0) -> logits`
  - `input_ids`: `torch.long` tensor of shape `[batch, sequence]`
  - `logits`: `torch.float` tensor of shape `[batch, sequence, vocab]`
- `parameter_count`: total trainable parameters, with tied embeddings counted once.

TODO: choose depth, hidden size, heads, KV heads, and the FFN.
TODO: define the block, attention, norm, and positional encoding.
TODO: keep this file self-contained. Do not import `model/` or the other
architecture files.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CustomConfig:
    """Stub config. Replace the TODO fields when the design is chosen."""

    vocab_size: int = 49_152
    max_seq_len: int = 8192
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    tie_embeddings: bool = True
    # TODO: add the architecture fields (n_layers, d_model, n_heads, ...).


_NOT_READY = (
    "Custom architecture is not implemented. "
    "Add the design in architectures/custom.py. "
    "DecoderLM must take a CustomConfig and implement "
    "forward(input_ids, position_offset=0) -> logits [batch, sequence, vocab] "
    "plus parameter_count."
)


class DecoderLM(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.config = config
        # TODO: build embeddings, blocks, and the tied output head here.
        raise NotImplementedError(_NOT_READY)

    def forward(
        self, input_ids: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        # TODO: run the custom stack and return logits.
        raise NotImplementedError(_NOT_READY)

    @property
    def parameter_count(self) -> int:
        # TODO: return the number of trainable parameters.
        raise NotImplementedError(_NOT_READY)
