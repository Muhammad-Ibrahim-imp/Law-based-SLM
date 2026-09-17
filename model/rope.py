import torch
import torch.nn as nn

from model.config import ModelConfig

class RoPE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dim = config.d_model // config.n_heads
        if self.dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        self.max_seq_len = config.max_seq_len
        self.theta = config.RoPE_theta
        inv_freq = 1 / (self.theta ** (torch.arange(0,self.dim,2,dtype = torch.float32,) / self.dim))
        # Not stored in build_cache as its independent of length of sequence and is fixed(based on dimensions)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(self.max_seq_len,
                          device=inv_freq.device)

    # Used to store cos and sin values in cache, depends on length of sequence
    def _build_cache(self, seq_len: int, device: torch.device):
        positions = torch.arange(seq_len,device = device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    # Applies rope to a given tensor of embedding vector,
    # having shape [Batch, Sequence, Heads, Dimensions]
    def forward(self, x: torch.Tensor, position_offset: int = 0):
        seq_len = x.shape[1]
        position_end = position_offset + seq_len
        if position_end > self.max_seq_len:
            raise ValueError("Position end exceeds max sequence length")
        x_even = x[...,0::2]
        x_odd = x[...,1::2]
        cos = self.cos_cached[position_offset:position_end]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[position_offset:position_end]
        sin = sin.unsqueeze(0).unsqueeze(2)
        even_rotated = x_even * cos - x_odd * sin
        odd_rotated = x_even * sin + x_odd * cos
        output = torch.empty_like(x)
        output[...,0::2] = even_rotated
        output[...,1::2] = odd_rotated
        return output


