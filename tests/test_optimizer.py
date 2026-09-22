import torch

from model.config import ModelConfig
from model.model import DecoderLM
from training.optimizer import adamw
from training.loss import causal_language_model_loss

def test_optimizer_updates_gradients():

    config = ModelConfig(
        vocab_size=100,
        max_seq_len=32,
        d_model=128,
        d_ff=328,
        n_heads=4,
        n_layers=2,
        n_kv_heads=2
    )

    model = DecoderLM(config)

    optimizer = adamw(model)

    input_ids = torch.randint(0,config.vocab_size,(2,16))
    labels = torch.randint(0,config.vocab_size,(2,16))

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)

    loss = causal_language_model_loss(logits, labels)

    loss.backward()

    old_weight = (
        model.embedding.embedding.weight
        .detach()
        .clone()
    )

    optimizer.step()

    new_weight = (
        model.embedding.embedding.weight
        .detach()
    )

    assert not torch.equal(
        old_weight,
        new_weight,
    )

