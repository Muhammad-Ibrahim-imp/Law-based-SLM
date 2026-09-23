import torch

from model.config import ModelConfig
from model.model import DecoderLM
from training.gradient import clip_gradients
from training.loss import causal_language_model_loss

def test_gradient_clipping():
    config = ModelConfig(vocab_size=100,
                         max_seq_len=32,
                         d_model=128,
                         d_ff=384,
                         n_layers=2,
                         n_heads=4,
                         n_kv_heads=2)
    model = DecoderLM(config)

    input_ids = torch.randint(0,config.vocab_size, (2,16))
    logits = model(input_ids)

    labels = torch.randint(0,config.vocab_size, (2,16))

    loss = causal_language_model_loss(logits, labels)

    loss = loss.square().mean()

    loss.backward()

    grad_norm = clip_gradients(model, max_norm=1)

    assert torch.isfinite(grad_norm)

    # After clipping, global norm should be <= 1
    final_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    assert final_norm <= 1.0 + 1e-6