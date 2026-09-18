import torch
import torch.nn as nn

from model.config import ModelConfig
from model.embeddings import LanguageModelHead, TokenEmbedding
from model.rmsnorm import RMSNorm
from model.transformer_block import TransformerBlock

class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.finalnorm = RMSNorm(config)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        if config.tie_embeddings:
            self.lm_head = LanguageModelHead(self.embedding.embedding)
        else:
            self.lm_head = LanguageModelHead(
                nn.Linear(
                    config.d_model, config.vocab_size, bias = config.bias
                )
            )
    def forward(self, input_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x, position_offset = position_offset)
        x = self.finalnorm(x)
        logits = self.lm_head(x)
        return logits

