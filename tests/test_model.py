import torch

from model.config import ModelConfig
from model.model import DecoderLM

def tiny_config():
    return ModelConfig(
        vocab_size=128,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )


def test_model_shape():
    config = tiny_config()
    model = DecoderLM(config)

    x = torch.randint(0, config.vocab_size, (1,32), dtype=torch.long)

    logits = model(x)

    assert logits.shape == (1,32,config.vocab_size)

def test_weight_tying():
    config = tiny_config()
    model = DecoderLM(config)

    assert (model.embedding.embedding.weight is model.lm_head.embedding.weight)

def test_model_gradients():
    config = tiny_config()
    model = DecoderLM(config)

    x = torch.randint(0, config.vocab_size, (1,32), dtype=torch.long)
    logits = model(x)

    loss = logits.mean()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
