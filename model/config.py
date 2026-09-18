from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ModelConfig:
    # Tokenizer
    vocab_size: int = 32000
    max_seq_len: int = 8192

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Transformer
    d_model: int = 1280
    n_layers: int = 24
    n_heads: int = 20
    n_kv_heads: int = 5

    # Feed Forward
    d_ff: int = 3840

    # Normalization/ Positional encoding
    RMS_eps: float = 1e-5
    RoPE_theta: int = 500000

    # Architecture
    bias: bool = False
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")

        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")

        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")

        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                "n_heads must be divisible by n_kv_heads"
            )

        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")


    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim
    @property
    def n_rep(self) -> int:
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return asdict(self)

cfg = ModelConfig()
print(cfg.head_dim)
print(cfg.kv_dim)
print(cfg.n_rep)