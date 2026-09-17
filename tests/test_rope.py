import torch

from model.config import ModelConfig
from model.rope import RoPE

def test_rope_shape():
    config = ModelConfig()
    rope = RoPE(config)
    x = torch.randn(2,16,config.n_heads, config.d_model // config.n_heads)
    y = rope(x)
    assert y.shape == x.shape

def test_rope_head_dim():
    config = ModelConfig()
    rope = RoPE(config)
    assert rope.dim == config.d_model // config.n_heads

def test_rope_does_not_change_norm():
    config = ModelConfig()
    rope = RoPE(config)
    x = torch.randn(2,16,config.n_heads, config.d_model // config.n_heads)
    y = rope(x)
    x_norm = torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True)
    y_norm = torch.linalg.vector_norm(y.float(), dim=-1, keepdim=True)
    assert torch.allclose(
        x_norm,
        y_norm,
        atol=1e-6,
        rtol=1e-6,
    )

def test_rope_changes_position_dependent_values():
    config = ModelConfig()
    rope = RoPE(config)
    x = torch.ones(1,8,config.n_heads, config.d_model // config.n_heads)
    y = rope(x)
    # Verify that the output embeddings for the first token (position 0)
    # and the second token (position 1) are distinct. This element-wise
    # comparison checks across all batches, all heads, and all head dimensions
    # to ensure position-dependent rotations were applied everywhere.
    #It passes (True for the assert) if even a single element is far.
    assert not torch.allclose(
        y[:,0],
        y[:,1],
    )