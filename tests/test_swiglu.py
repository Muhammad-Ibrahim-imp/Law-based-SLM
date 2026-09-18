import torch

from model.config import ModelConfig
from model.swiglu import SwiGLU

def test_swiglu_shape():
    config = ModelConfig()
    mlp = SwiGLU(config)
    x = torch.randn(2,128,config.d_model, requires_grad=True)
    output = mlp(x)
    assert output.shape == x.shape

def test_swiglu_gradient():
    config = ModelConfig()
    mlp = SwiGLU(config)
    x = torch.randn(2,128,config.d_model, requires_grad=True)
    output = mlp(x)
    loss = output.mean()
    loss.backward()
    assert x.grad is not None
    for param in mlp.parameters():
        assert param.grad is not None