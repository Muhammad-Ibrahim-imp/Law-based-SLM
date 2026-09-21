import torch

from model.config import ModelConfig
from model.model import DecoderLM
from training.loss import causal_language_model_loss

def test_loss_is_scalar():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=384,
    )
    model = DecoderLM(config)
    inputs_ids = torch.randint(0,config.vocab_size,(2,16))
    logits = model(inputs_ids)
    labels = torch.randint(0,config.vocab_size,(2,16))
    loss = causal_language_model_loss(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_loss_backward():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=384,
    )
    model = DecoderLM(config)
    inputs_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(inputs_ids)
    labels = torch.randint(0, config.vocab_size, (2, 16))
    loss = causal_language_model_loss(logits, labels)

    loss.backward()
    assert model.embedding.embedding.weight.grad is not None