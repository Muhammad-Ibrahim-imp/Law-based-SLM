import torch
import torch.nn as nn

from model.config import ModelConfig
from model.embeddings import LanguageModelHead, TokenEmbedding
from model.rmsnorm import RMSNorm
from model.transformer_block import TransformerBlock
from model.initialization import initialize_model

class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config)
        self.lm_head = LanguageModelHead(self.embedding.embedding)

        initialize_model(self, config)

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

