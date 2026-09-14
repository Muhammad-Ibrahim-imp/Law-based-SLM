import torch

from model.config import ModelConfig
from model.embeddings import TokenEmbedding, LanguageModelHead
# tests whether TokenEmbedding is actually making valid embedding matrix
def test_token_embedding_shape():
    config = ModelConfig()
    embedding = TokenEmbedding(config)
    input_ids = torch.randint(0, config.d_model, (2,16), dtype=torch.long)
    hidden_states = embedding(input_ids)
    assert (hidden_states.shape == (2,16,config.d_model))
# tests whether the concept kf weight tying is being implemented correctly
def test_weight_tying():
    config = ModelConfig()
    embedding = TokenEmbedding(config)
    lm_head = LanguageModelHead(embedding.embedding)
    assert (embedding.embedding.weight is lm_head.embedding.weight)
# tests whether the shape of logits produced by LanguageModelHead is valid or not
def test_lm_head_shape():
    config = ModelConfig()
    embedding = TokenEmbedding(config)
    lm_head = LanguageModelHead(embedding.embedding)
    hidden_states = lm_head(torch.randn(2,16,config.d_model))
    assert hidden_states.shape == (2,16,config.vocab_size)