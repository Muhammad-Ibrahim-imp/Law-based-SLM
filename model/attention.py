import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from model.config import ModelConfig
from model.rope import RoPE

class GQAttention(nn.Module):

    def __init__(self, config: ModelConfig):

        super().__init__()

        self.dim = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // self.n_heads

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.rope = RoPE(config)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:

        B, T, _ = x.shape # B = Number of batches, T = tokens per batch

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        q = self.rope(q,position_offset)
        k = self.rope(k,position_offset)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        sdpa_args = {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "enable_gqa": True,
        }
        if x.is_cuda:
            # Force PyTorch's FlashAttention-2-compatible fused SDPA backend.
            # Failing loudly here prevents accidental benchmark comparisons
            # against the slower memory-hungry math implementation.
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                output = F.scaled_dot_product_attention(q, k, v, **sdpa_args)
        else:
            output = F.scaled_dot_product_attention(q, k, v, **sdpa_args)

        output = output.transpose(1, 2)
        output = output.contiguous().view(B, T, self.n_heads * self.head_dim)
        output = self.o_proj(output)
        return output

