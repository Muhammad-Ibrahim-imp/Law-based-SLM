import pytest

from model.config import ModelConfig
from model.parameter_count import estimate_parameter_count

def test_default_config():
    cfg = ModelConfig()
    assert cfg.kv_dim == 320
    assert cfg.head_dim == 64
    assert cfg.n_rep == 4

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
    assert 480000000 <= parameters <= 520000000