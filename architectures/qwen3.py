"""Qwen3 decoder sized near 200M parameters.

Source: Hugging Face `modeling_qwen3.py` and `Qwen/Qwen3-0.6B` config.json.
Kept from that model: GQA, per-head RMSNorm on q and k before RoPE, RoPE on
the full head dim (half-split), SwiGLU with SiLU, no QKV bias, pre-norm
residuals, RMSNorm epsilon 1e-6, RoPE theta 1_000_000.
Qwen3-0.6B does not use a sliding window (`use_sliding_window` is false).

`head_dim` stays 128, so query width is `n_heads * 128`, not `d_model`.
Depth and width are reduced from the 0.6B checkpoint so vocab 49,152 is near
200M. Weights are float32. This file imports neither `model/` nor any other
architecture. There is no KV cache; `position_offset` affects RoPE only.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 49_152
    max_seq_len: int = 8192
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    d_model: int = 768
    n_layers: int = 18
    n_heads: int = 12
    n_kv_heads: int = 6
    head_dim: int = 128
    d_ff: int = 2304

    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0 or self.max_seq_len > 8192:
            raise ValueError("max_seq_len must be in 1..8192")
        if min(self.d_model, self.n_layers, self.n_heads, self.n_kv_heads, self.d_ff) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not self.tie_embeddings:
            raise ValueError("this architecture requires tied embeddings")


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_offset: int) -> torch.Tensor:
        seq_len = x.shape[1]
        end = position_offset + seq_len
        cos = self.cos_cached[position_offset:end].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[position_offset:end].to(device=x.device, dtype=x.dtype)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return x * cos + _rotate_half(x) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    args = {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": True,
        "enable_gqa": True,
    }
    if q.is_cuda:
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, **args)
    return F.scaled_dot_product_attention(q, k, v, **args)


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_eps)
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor, position_offset: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = self.rope(q, position_offset)
        k = self.rope(k, position_offset)
        out = _causal_sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model, config.rms_eps)
        self.attention = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.rms_eps)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor, position_offset: int) -> torch.Tensor:
        x = x + self.attention(self.input_layernorm(x), position_offset)
        x = x + self.feed_forward(self.post_attention_layernorm(x))
        return x


class TiedHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class DecoderLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = TiedHead(self.embedding)
        _initialize(self, config.n_layers)

    def forward(
        self, input_ids: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if position_offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, position_offset=position_offset)
        return self.lm_head(self.final_norm(x))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def embedding_parameter_count(self) -> int:
        return self.embedding.weight.numel()


def _initialize(model: DecoderLM, n_layers: int) -> None:
    std = 0.02
    scaled_std = std / math.sqrt(2 * n_layers)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    for layer in model.layers:
        nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=scaled_std)
        nn.init.normal_(layer.feed_forward.down_proj.weight, mean=0.0, std=scaled_std)
