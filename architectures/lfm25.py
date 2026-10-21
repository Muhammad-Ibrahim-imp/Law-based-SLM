"""LFM2.5 decoder sized near 200M parameters.

Sources, in this order when they disagree:
- Hugging Face `modeling_lfm2.py` (the executable definition of the released
  LFM2 / LFM2.5 stack): double-gated short convolution, SwiGLU with the
  published 2/3 FFN adjustment, pre-norm residuals, QK RMSNorm before RoPE.
- `LiquidAI/LFM2.5-350M` config.json for the 16-layer conv/attention order,
  kernel length 3, RoPE theta 1_000_000, and RMSNorm epsilon 1e-5.

This file does not import any other architecture or `model/`.
The width is narrower than LFM2.5-350M so the SmolLM2 vocabulary of 49,152
lands near 200M parameters. The 16-layer pattern is unchanged.
Weights are float32. Training bf16 happens only in the existing autocast.
Generation can pass a cache: attention layers reuse keys and values, and each
short convolution keeps its last ``kernel - 1`` inputs. Training calls
``forward`` without a cache. `position_offset` still affects RoPE only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from dataclasses import dataclass

from common.kv_cache import AttentionSlot, ConvSlot


OFFICIAL_LAYER_TYPES = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)


def adjusted_ffn_size(
    intermediate_size: int,
    auto_adjust: bool,
    multiplier: float,
    multiple_of: int,
) -> int:
    """Match `Lfm2MLP`: optional 2/3 shrink, multiplier, then round up."""
    size = intermediate_size
    if auto_adjust:
        size = int(2 * size / 3)
        if multiplier is not None:
            size = int(multiplier * size)
        size = multiple_of * ((size + multiple_of - 1) // multiple_of)
    return size


@dataclass(frozen=True)
class LFM25Config:
    vocab_size: int = 49_152
    max_seq_len: int = 8192
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 6
    # Raw FFN width from the 350M ratio 6656/1024 = 6.5. The MLP uses the
    # adjusted width (3328 at this hidden size).
    intermediate_size: int = 4992
    block_auto_adjust_ff_dim: bool = True
    block_ffn_dim_multiplier: float = 1.0
    block_multiple_of: int = 256
    conv_kernel: int = 3
    conv_bias: bool = False

    rms_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    tie_embeddings: bool = True
    layer_types: tuple[str, ...] = OFFICIAL_LAYER_TYPES

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0 or self.max_seq_len > 8192:
            raise ValueError("max_seq_len must be in 1..8192")
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("d_model, n_heads, and n_kv_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if self.conv_kernel < 1:
            raise ValueError("conv_kernel must be positive")
        if self.conv_bias:
            raise ValueError("LFM2.5-350M uses conv_bias=False")
        if not self.tie_embeddings:
            raise ValueError("this architecture requires tied embeddings")
        if len(self.layer_types) == 0:
            raise ValueError("layer_types must not be empty")
        if any(kind not in ("conv", "full_attention") for kind in self.layer_types):
            raise ValueError("layer_types entries must be 'conv' or 'full_attention'")
        if self.block_multiple_of <= 0:
            raise ValueError("block_multiple_of must be positive")

    @property
    def n_layers(self) -> int:
        return len(self.layer_types)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        return adjusted_ffn_size(
            self.intermediate_size,
            self.block_auto_adjust_ff_dim,
            self.block_ffn_dim_multiplier,
            self.block_multiple_of,
        )


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
    """Half-split RoPE used by Hugging Face LFM2, over the full head dim."""

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
        # x: [B, T, heads, dim]
        seq_len = x.shape[1]
        end = position_offset + seq_len
        cos = self.cos_cached[position_offset:end].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[position_offset:end].to(device=x.device, dtype=x.dtype)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        rotated = _rotate_half(x)
        return x * cos + rotated * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return torch.cat((-second, first), dim=-1)


def _causal_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q, k, v: [B, heads, T, dim]
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
    def __init__(self, config: LFM25Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, config.rms_eps)
        self.k_layernorm = RMSNorm(self.head_dim, config.rms_eps)
        self.rope = RoPE(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int,
        slot: AttentionSlot | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_layernorm(self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        q = self.rope(q, position_offset)
        k = self.rope(k, position_offset)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if slot is None or slot.length == 0:
            out = _causal_sdpa(q, k, v)
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
            )
        else:
            raise ValueError("cached decode accepts one new token at a time")
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.out_proj(out)


class ShortConv(nn.Module):
    """Gated depthwise conv: y = out_proj(C * conv(B * x)), kernel causal."""

    def __init__(self, config: LFM25Config):
        super().__init__()
        hidden = config.d_model
        kernel = config.conv_kernel
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel,
            groups=hidden,
            bias=config.conv_bias,
            padding=kernel - 1,
        )
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, slot: ConvSlot | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        b_gate, c_gate, values = self.in_proj(x).chunk(3, dim=-1)
        mixed = (b_gate * values).transpose(1, 2)
        if slot is None or slot.state is None:
            convolved = self.conv(mixed)[:, :, :seq_len]
            if slot is not None:
                slot.state = _conv_state(mixed, self.conv.kernel_size[0])
        else:
            convolved, slot.state = _conv_step(self.conv, slot.state, mixed)
        y = c_gate * convolved.transpose(1, 2)
        return self.out_proj(y)


def _conv_state(mixed: torch.Tensor, kernel: int) -> torch.Tensor:
    keep = kernel - 1
    if keep == 0:
        return mixed.new_empty(mixed.shape[0], mixed.shape[1], 0)
    if mixed.shape[-1] >= keep:
        return mixed[:, :, -keep:].detach().contiguous()
    pad = mixed.new_zeros(mixed.shape[0], mixed.shape[1], keep - mixed.shape[-1])
    return torch.cat([pad, mixed], dim=-1).detach()


def _conv_step(
    conv: nn.Conv1d, state: torch.Tensor, mixed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    incoming = torch.cat([state, mixed], dim=-1)
    convolved = F.conv1d(incoming, conv.weight, conv.bias, groups=conv.groups)
    keep = conv.kernel_size[0] - 1
    if keep == 0:
        new_state = incoming.new_empty(incoming.shape[0], incoming.shape[1], 0)
    else:
        new_state = incoming[:, :, -keep:].detach().contiguous()
    return convolved, new_state


class SwiGLU(nn.Module):
    def __init__(self, config: LFM25Config):
        super().__init__()
        hidden = config.d_model
        inner = config.d_ff
        self.w1 = nn.Linear(hidden, inner, bias=False)
        self.w3 = nn.Linear(hidden, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, config: LFM25Config, kind: str):
        super().__init__()
        self.is_attention_layer = kind == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Attention(config)
        else:
            self.conv = ShortConv(config)
        self.feed_forward = SwiGLU(config)
        self.operator_norm = RMSNorm(config.d_model, config.rms_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int,
        slot: AttentionSlot | ConvSlot | None = None,
    ) -> torch.Tensor:
        if self.is_attention_layer:
            mixed = self.self_attn(self.operator_norm(x), position_offset, slot)
        else:
            mixed = self.conv(self.operator_norm(x), slot)
        x = x + mixed
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class TiedHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class LFMCache:
    """Attention slots and convolution states, one per LFM layer."""

    def __init__(self, config: LFM25Config):
        self.seen = 0
        self.slots = []
        for kind in config.layer_types:
            if kind == "full_attention":
                self.slots.append(
                    AttentionSlot(
                        config.max_seq_len,
                        config.n_kv_heads,
                        config.head_dim,
                        window=None,
                    )
                )
            else:
                self.slots.append(ConvSlot(config.conv_kernel))


class DecoderLM(nn.Module):
    def __init__(self, config: LFM25Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            Block(config, kind) for kind in config.layer_types
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = TiedHead(self.embedding)
        _initialize(self, config.n_layers)

    def build_cache(self, batch_size: int) -> LFMCache:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return LFMCache(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_offset: int = 0,
        cache: LFMCache | None = None,
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
    # Same scheme as the GQA decoder so architecture comparisons are not
    # mixed with LFM2.5-350M's Xavier init.
    std = 0.02
    scaled_std = std / math.sqrt(2 * n_layers)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    for layer in model.layers:
        if layer.is_attention_layer:
            nn.init.normal_(layer.self_attn.out_proj.weight, mean=0.0, std=scaled_std)
        else:
            nn.init.normal_(layer.conv.out_proj.weight, mean=0.0, std=scaled_std)
        nn.init.normal_(layer.feed_forward.w2.weight, mean=0.0, std=scaled_std)
