import math
import torch.nn as nn

from model.config import ModelConfig

def initialize_model(
    model: nn.Module, config: ModelConfig
) -> None:

    std = 0.02
    scaled_std = std / math.sqrt(2 * config.n_layers)

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    for layer in model.layers:

        nn.init.normal_(layer.attention.o_proj.weight, mean=0, std=scaled_std)

        nn.init.normal_(layer.feed_forward.down_proj.weight, mean=0, std=scaled_std)
