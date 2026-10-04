import torch

from architectures.qwen3 import DecoderLM, Qwen3Config
from architectures.registry import build_model
from training.optimizer import adamw
from training.scheduler import build_scheduler_cosine
from training.trainer import Trainer

EXPECTED_PARAMETERS = 197_034_240


def test_default_count_and_qk_norm():
    config = Qwen3Config()
    assert config.head_dim == 128
    assert config.n_heads * config.head_dim != config.d_model
    model = build_model("qwen3")
    assert isinstance(model, DecoderLM)
    assert model.parameter_count == EXPECTED_PARAMETERS
    assert model.embedding.weight is model.lm_head.embedding.weight
    layer = model.layers[0]
    assert layer.attention.q_norm.weight.shape == (config.head_dim,)
    assert layer.attention.k_norm.weight.shape == (config.head_dim,)
    assert layer.attention.q_proj.bias is None


def test_forward_and_trainer_step():
    config = Qwen3Config()
    model = DecoderLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(logits).all()

    optimizer = adamw(model, learning_rate=3e-4)
    scheduler = build_scheduler_cosine(optimizer, total_steps=2, warmup_steps=0)
    trainer = Trainer(model, optimizer, scheduler, torch.device("cpu"))
    result = trainer.train_step(
        {
            "input_ids": tokens[:1, :4],
            "labels": torch.randint(0, config.vocab_size, (1, 4)),
        }
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["grad_norm"])
