import torch
import torch.nn as nn

from model.config import ModelConfig

class RoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.head_dim
        if self.dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        self.max_seq_len = config.max_seq_len
        self.theta = config.rope_theta
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    # Applies rope to a given tensor of embedding vector,
    # having shape [Batch, Sequence, Heads, Dimensions]
    def forward(self, x: torch.Tensor, position_offset: int = 0):
        seq_len = x.shape[1]
        position_end = position_offset + seq_len
        if position_end > self.max_seq_len:
            raise ValueError("Position end exceeds max sequence length")
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        cos = self.cos_cached[position_offset:position_end].to(
            device=x.device, dtype=x.dtype
        )[None, :, None, :]
        sin = self.sin_cached[position_offset:position_end].to(
            device=x.device, dtype=x.dtype
        )[None, :, None, :]
        even_rotated = x_even * cos - x_odd * sin
        odd_rotated = x_even * sin + x_odd * cos
        output = torch.empty_like(x)
        output[...,0::2] = even_rotated
        output[...,1::2] = odd_rotated
        return output


