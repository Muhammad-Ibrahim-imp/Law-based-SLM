import torch

from architectures.lfm25 import (
    OFFICIAL_LAYER_TYPES,
    DecoderLM,
    LFM25Config,
    adjusted_ffn_size,
)
from architectures.registry import build_model
from training.optimizer import adamw
from training.scheduler import build_scheduler_cosine
from training.trainer import Trainer

EXPECTED_PARAMETERS = 194_691_072


def test_official_ffn_adjustment():
    # LiquidAI/LFM2.5-350M intermediate_size 6656 becomes 4608 after the
    # Hugging Face Lfm2MLP adjustment.
    assert adjusted_ffn_size(6656, True, 1.0, 256) == 4608


def test_default_shape_and_count():
    config = LFM25Config()
    assert config.layer_types == OFFICIAL_LAYER_TYPES
    assert config.layer_types.count("conv") == 10
    assert config.layer_types.count("full_attention") == 6
    assert config.d_ff == 3328

    model = DecoderLM(config)
    assert model.parameter_count == EXPECTED_PARAMETERS
    assert model.embedding_parameter_count == config.vocab_size * config.d_model
    assert model.embedding.weight is model.lm_head.embedding.weight


def test_forward_shape():
    config = LFM25Config()
    model = DecoderLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(tokens, position_offset=0)
    assert logits.shape == (2, 8, config.vocab_size)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_registry_and_trainer_step():
    model = build_model("lfm25")
    optimizer = adamw(model, learning_rate=3e-4)
    scheduler = build_scheduler_cosine(optimizer, total_steps=2, warmup_steps=0)
    trainer = Trainer(model, optimizer, scheduler, torch.device("cpu"))
    batch = {
        "input_ids": torch.randint(0, model.config.vocab_size, (1, 4)),
        "labels": torch.randint(0, model.config.vocab_size, (1, 4)),
    }
    result = trainer.train_step(batch)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["grad_norm"])
