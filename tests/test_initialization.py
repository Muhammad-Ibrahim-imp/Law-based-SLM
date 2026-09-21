import torch

from model.config import ModelConfig
from model.model import DecoderLM

def test_parameters_are_finite():
    config = ModelConfig()
    model = DecoderLM(config)

    for param in model.parameters():
        assert torch.isfinite(param).all()

def test_weight_tying_after_initialization():
    config = ModelConfig()
    model = DecoderLM(config)

    assert (model.embedding.embedding.weight
            is model.lm_head.embedding.weight)