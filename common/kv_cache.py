"""Fixed-size inference caches for attention and short convolution.

Training calls ``forward`` without a cache. Generation allocates one cache,
fills it with the prompt, then writes one new token into the next free slot.
"""

from __future__ import annotations

import torch


class AttentionSlot:
    """Preallocated keys and values for one attention layer.

    Storage is allocated on the first append, using that tensor's device and
    dtype, so a BF16 decode does not keep an FP32 copy. ``window`` limits
    which stored keys a later token may attend to. The stored keys themselves
    stay in position order up to ``max_len``.
    """

    def __init__(
        self,
        max_len: int,
        n_kv_heads: int,
        head_dim: int,
        window: int | None,
    ):
        self.max_len = max_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window = window
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.length = 0

    def append(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy ``[batch, kv_heads, time, dim]`` keys and return the visible span."""
        keys = keys.detach()
        values = values.detach()
        if keys.ndim != 4 or keys.shape[1:] != (
            self.n_kv_heads,
            keys.shape[2],
            self.head_dim,
        ):
            raise ValueError("keys must have shape [batch, kv_heads, time, head_dim]")
        batch, _, steps, _ = keys.shape
        if self.k is None:
            self.k = keys.new_empty(batch, self.n_kv_heads, self.max_len, self.head_dim)
            self.v = keys.new_empty(batch, self.n_kv_heads, self.max_len, self.head_dim)
        elif self.k.shape[0] != batch:
            raise ValueError("cache batch size does not match the input")
        end = self.length + steps
        if end > self.max_len:
            raise ValueError("input sequence exceeds max_seq_len")
        self.k[:, :, self.length : end] = keys
        self.v[:, :, self.length : end] = values
        self.length = end
        start = 0 if self.window is None else max(0, self.length - self.window)
        return self.k[:, :, start : self.length], self.v[:, :, start : self.length]


class ConvSlot:
    """Last ``kernel - 1`` convolution inputs for one LFM short-conv layer."""

    def __init__(self, kernel: int):
        if kernel < 1:
            raise ValueError("kernel must be positive")
        self.kernel = kernel
        self.state: torch.Tensor | None = None
