import torch

from model.config import ModelConfig
from model.transformer_block import TransformerBlock

def test_transformer_block_shape():
    config = ModelConfig()
    transformer = TransformerBlock(config)
    x = torch.randn(2, 128, config.d_model)
    output = transformer(x)
    assert output.shape == x.shape

def test_transformer_block_gradients():
    config = ModelConfig()
    transformer = TransformerBlock(config)
    x = torch.randn(2, 128, config.d_model, requires_grad=True)
    output = transformer(x)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None
    for param in transformer.parameters():
        assert param.grad is not None