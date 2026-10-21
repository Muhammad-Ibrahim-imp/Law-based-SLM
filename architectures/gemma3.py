"""Gemma 3 text decoder sized near 200M parameters.

Sources:
- Hugging Face `modeling_gemma3.py`: RMSNorm is `(1 + weight) * rms` with
  weight initialized at 0, GeGLU uses the tanh GELU approximation, input
  embeddings are multiplied by sqrt(hidden size), QK-norm is applied before
  RoPE, and each block has pre-norm and post-norm around attention and the MLP.
- Gemma 3 technical report: 5 local sliding-window layers per 1 global layer.
- Gemma 3 270M config (`gemma.gm.nn.Gemma3_270M`): head dim 256, 4 query heads,
  1 KV head, local window 512, local RoPE base 10_000, global RoPE base
  1_000_000, no attention or final-logit soft cap.

Local layers use a sliding-window causal mask. That mask cannot go through the
flash-attention-only path the GQA model uses, so local layers use ordinary
SDPA. Global layers still force flash attention on CUDA.
The 270M depth pattern is kept (18 layers = three repeats of 5 local + 1
global). Hidden size and FFN width are scaled so vocab 49,152 is near 200M.
`head_dim` is not `d_model / n_heads`. Weights are float32.
Passing a cache makes generation reuse keys and values. Training calls
``forward`` without one, and that path is unchanged.
This file imports neither `model/` nor any other architecture.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.kv_cache import AttentionSlot


def gemma3_layer_types(n_layers: int, local_per_global: int = 5) -> tuple[str, ...]:
    block = local_per_global + 1
    if n_layers % block != 0:
        raise ValueError(
            f"n_layers must be a multiple of {block} so the 5 local : 1 global "
            "pattern repeats exactly"
        )
    pattern = ("sliding_attention",) * local_per_global + ("full_attention",)
    return pattern * (n_layers // block)


@dataclass(frozen=True)
class Gemma3Config:
    vocab_size: int = 49_152
    max_seq_len: int = 8192
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    d_model: int = 832
    n_layers: int = 18
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 256
    d_ff: int = 2688
    sliding_window: int = 512
    # Official 270M sets this equal to head_dim. SDPA then scales by 1/sqrt(head_dim).
    query_pre_attn_scalar: int = 256

    rms_eps: float = 1e-6
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0 or self.max_seq_len > 8192:
            raise ValueError("max_seq_len must be in 1..8192")
        if min(
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.d_ff,
            self.sliding_window,
            self.query_pre_attn_scalar,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.n_layers % 6 != 0:
            raise ValueError("n_layers must be a multiple of 6 (5 local + 1 global)")
        if not self.tie_embeddings:
            raise ValueError("this architecture requires tied embeddings")

    @property
    def layer_types(self) -> tuple[str, ...]:
        return gemma3_layer_types(self.n_layers)


class RMSNorm(nn.Module):
    """Gemma RMSNorm: weight starts at 0 so the scale is (1 + weight)."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)


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


def _sliding_causal_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    # True means the query may attend to that key. PyTorch SDPA uses this
    # convention for boolean masks.
    positions = torch.arange(seq_len, device=device)
    query = positions[:, None]
    key = positions[None, :]
    return (key <= query) & ((query - key) < window)


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sliding_window: int | None,
    scale: float,
) -> torch.Tensor:
    if sliding_window is None:
        args = {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "enable_gqa": True,
            "scale": scale,
        }
        if q.is_cuda:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, **args)
        return F.scaled_dot_product_attention(q, k, v, **args)
    mask = _sliding_causal_mask(q.shape[-2], sliding_window, q.device)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
        scale=scale,
    )


class Attention(nn.Module):
    def __init__(self, config: Gemma3Config, sliding: bool):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window if sliding else None
        self.scale = config.query_pre_attn_scalar ** -0.5
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_eps)
        theta = config.rope_theta_local if sliding else config.rope_theta_global
        self.rope = RoPE(self.head_dim, config.max_seq_len, theta)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int,
        slot: AttentionSlot | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, position_offset)
        k = self.rope(k, position_offset)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if slot is None or slot.length == 0:
            out = _attend(q, k, v, self.sliding_window, self.scale)
            if slot is not None:
                slot.append(k, v)
        elif seq_len == 1:
            cached_k, cached_v = slot.append(k, v)
            out = F.scaled_dot_product_attention(
                q,
                cached_k,
                cached_v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
                scale=self.scale,
            )
        else:
            raise ValueError("cached decode accepts one new token at a time")
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class GeGLU(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config: Gemma3Config, kind: str):
        super().__init__()
        sliding = kind == "sliding_attention"
        self.input_layernorm = RMSNorm(config.d_model, config.rms_eps)
        self.attention = Attention(config, sliding=sliding)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.rms_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.d_model, config.rms_eps)
        self.feed_forward = GeGLU(config)
        self.post_feedforward_layernorm = RMSNorm(config.d_model, config.rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int,
        slot: AttentionSlot | None = None,
    ) -> torch.Tensor:
        attn = self.post_attention_layernorm(
            self.attention(self.input_layernorm(x), position_offset, slot)
        )
        x = x + attn
        mlp = self.post_feedforward_layernorm(self.feed_forward(self.pre_feedforward_layernorm(x)))
        return x + mlp


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        scale = embedding_dim ** 0.5
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(dtype=self.weight.dtype)


class TiedHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class GemmaCache:
    """One preallocated attention slot per Gemma layer."""

    def __init__(self, config: Gemma3Config):
        self.seen = 0
        self.slots = [
            AttentionSlot(
                config.max_seq_len,
                config.n_kv_heads,
                config.head_dim,
                config.sliding_window if kind == "sliding_attention" else None,
            )
            for kind in config.layer_types
        ]


class DecoderLM(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.embedding = ScaledEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            Block(config, kind) for kind in config.layer_types
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = TiedHead(self.embedding)
        _initialize(self, config.n_layers)

    def build_cache(self, batch_size: int) -> GemmaCache:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return GemmaCache(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        cache: GemmaCache | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        offset = position_offset if cache is None else cache.seen
        if offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        x = self.embedding(input_ids)
        slots = (None,) * len(self.layers) if cache is None else cache.slots
        for layer, slot in zip(self.layers, slots):
            x = layer(x, offset, slot)
        if cache is not None:
            cache.seen += input_ids.shape[1]
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
        elif isinstance(module, RMSNorm):
            nn.init.zeros_(module.weight)
    for layer in model.layers:
        nn.init.normal_(layer.attention.o_proj.weight, mean=0.0, std=scaled_std)
        nn.init.normal_(layer.feed_forward.down_proj.weight, mean=0.0, std=scaled_std)
