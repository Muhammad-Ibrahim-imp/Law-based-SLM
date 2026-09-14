import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import supported_ops

from model.config import ModelConfig

# For making embedding matrix
class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        #Makes embedding matrix
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )
    # returns embedding vectors when input token ids are given
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

class LanguageModelHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()

        self.embedding = embedding

    # Returns dot product of hidden vector(produced at last for each word)
    # with embedding weights to produce logits(scores for next word)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.embedding.weight)
