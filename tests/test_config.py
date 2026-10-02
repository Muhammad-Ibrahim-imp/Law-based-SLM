import pytest

from model.config import ModelConfig
from model.parameter_count import estimate_parameter_count

def test_default_config():
    cfg = ModelConfig()
    assert cfg.vocab_size == 49_152
    assert cfg.d_model == 768
    assert cfg.n_layers == 24
    assert cfg.n_heads == 12
    assert cfg.n_kv_heads == 4
    assert cfg.head_dim == 64
    assert cfg.kv_dim == 256
    assert cfg.n_rep == 3
    assert cfg.d_ff == 2_240
    assert cfg.rope_theta == 500_000.0

def invalid_head_division():
    with pytest.raises(ValueError):
        ModelConfig(
            d_model=1281,
            n_heads=20
        )

def test_invalid_gqa():
    with pytest.raises(ValueError):
        ModelConfig(
            n_heads = 20,
            n_kv_heads=6,
        )

def test_params_count():
    cfg = ModelConfig()
    parameters = estimate_parameter_count(cfg)["total"]
    assert parameters == 199_398_144