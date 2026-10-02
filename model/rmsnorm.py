import torch
import torch.nn as nn

from model.config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.d_model))
        self.epsilon = config.rms_eps
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(
            x.pow(2).mean(
                dim = -1,
                keepdim = True
            ) + self.epsilon
        )
        output = x * rms
        output = output * self.weight
        return output.to(input_dtype)

