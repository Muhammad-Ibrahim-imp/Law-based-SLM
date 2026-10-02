from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class ModelConfig:
    # Tokenizer
    vocab_size: int = 49_152
    max_seq_len: int = 8192

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Transformer
    d_model: int = 768
    n_layers: int = 24
    n_heads: int = 12
    n_kv_heads: int = 4

    # Feed Forward
    d_ff: int = 2_240

    # Normalization/ Positional encoding
    rms_eps: float = 1e-5
    rope_theta: float = 500_000.0

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

        if self.max_seq_len > 8192:
            raise ValueError("max_seq_len cannot exceed 8192")
        if self.bias:
            raise ValueError("this architecture does not use linear biases")
        if self.dropout != 0.0:
            raise ValueError("this architecture requires dropout=0.0")
        if not self.tie_embeddings:
            raise ValueError("this architecture requires tied embeddings")

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