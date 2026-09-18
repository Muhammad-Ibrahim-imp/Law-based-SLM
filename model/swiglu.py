import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias = config.bias)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias = config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias = config.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        down = self.down_proj(hidden)
        return down