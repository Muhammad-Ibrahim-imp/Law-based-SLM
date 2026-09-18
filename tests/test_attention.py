import torch

from model.config import ModelConfig
from model.attention import GQAttention

def test_attention_shape():
    config = ModelConfig()
    attention = GQAttention(config)

    x = torch.randn(2, 128, config.d_model, dtype=torch.float)
    y = attention(x)

    assert y.shape == x.shape

def test_attention_gradients():
    config = ModelConfig()
    attention = GQAttention(config)

    x = torch.randn(2, 128, config.d_model, dtype=torch.float, requires_grad=True)
    y = attention(x)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None

    for parameters in attention.parameters():
        assert parameters.grad is not None

def test_gqa_configuration():
    config = ModelConfig()
    attention = GQAttention(config)

    assert attention.n_heads == config.n_heads
    assert attention.n_kv_heads == config.n_kv_heads
    assert attention.head_dim == config.d_model // config.n_heads
    assert attention.kv_repeats == config.n_heads // config.n_kv_heads