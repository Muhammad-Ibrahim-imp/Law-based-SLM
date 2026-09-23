import torch

from model.config import ModelConfig
from model.model import DecoderLM
from training.optimizer import adamw
from training.scheduler import build_scheduler_cosine

def test_scheduler_warmup_and_decay():
    config = ModelConfig(vocab_size=100,
                         max_seq_len=32,
                         d_model=128,
                         d_ff=384,
                         n_heads=4,
                         n_layers=2,
                         n_kv_heads=2)
    model = DecoderLM(config)

    optimizer = adamw(model,
                      learning_rate=3e-4,)

    scheduler = build_scheduler_cosine(optimizer = optimizer,
                                       warmup_steps=10,
                                       total_steps=100,
                                       min_lr_ratio=0.1,)

    learning_rates = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

    assert learning_rates[0] > 0.0

    assert learning_rates[-1] < learning_rates[10]

    assert learning_rates[-1] > 0.0

