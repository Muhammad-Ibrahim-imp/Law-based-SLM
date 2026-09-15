import torch

from model.config import ModelConfig
from model.rmsnorm import RMSNorm

def test_rmsnorm_shape():
    config = ModelConfig()
    norm = RMSNorm(config)
    x = torch.randn(2,8,config.d_model)
    y = norm(x)
    assert y.shape == x.shape

def test_rmsnorm_has_one_parameter():
    config = ModelConfig()
    norm = RMSNorm(config)
    parameters = list(norm.parameters())
    assert len(parameters) == 1
    assert parameters[0].shape == (config.d_model,)

def test_rmsnorm_unit_rms():
    config = ModelConfig()
    norm = RMSNorm(config)
    x = torch.randn(2,8,config.d_model)
    y = norm(x)
    y = y.pow(2).mean( dim=-1,)
    assert torch.allclose(y, torch.ones_like(y), atol=1e-5)

def test_rmsnorm_gradient():
    config = ModelConfig()
    norm = RMSNorm(config)
    x = torch.randn(2,8,config.d_model, requires_grad= True)
    y = norm(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None