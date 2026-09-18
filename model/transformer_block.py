import torch

from model.config import ModelConfig
from model.attention import GQAttention
from model.rmsnorm import RMSNorm
from model.swiglu import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.att_norm = RMSNorm(config)
        self.attention = GQAttention(config)
        self.ff_norm = RMSNorm(config)
        self.feed_forward = SwiGLU(config)
    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        x = x + self.attention(self.att_norm(x), position_offset = position_offset)
        x = x + self.feed_forward(self.ff_norm(x))
        return x
