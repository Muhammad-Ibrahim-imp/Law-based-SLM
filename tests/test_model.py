import torch

from model.config import ModelConfig
from model.model import DecoderLM

def test_model_shape():
    config = ModelConfig()
    model = DecoderLM(config)

    x = torch.randint(0, config.vocab_size, (1,32), dtype=torch.long)

    logits = model(x)

    assert logits.shape == (1,32,config.vocab_size)

def test_weight_tying():
    config = ModelConfig()
    model = DecoderLM(config)

    assert (model.embedding.embedding.weight is model.lm_head.embedding.weight)

def test_model_gradients():
    config = ModelConfig()
    model = DecoderLM(config)

    x = torch.randint(0, config.vocab_size, (1,32), dtype=torch.long)
    logits = model(x)

    loss = logits.mean()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
