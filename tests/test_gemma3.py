import torch

from architectures.gemma3 import DecoderLM, Gemma3Config
from architectures.registry import build_model
from training.optimizer import adamw
from training.scheduler import build_scheduler_cosine
from training.trainer import Trainer

EXPECTED_PARAMETERS = 200_069_440


def test_default_pattern_and_count():
    config = Gemma3Config()
    kinds = config.layer_types
    assert len(kinds) == 18
    assert kinds.count("sliding_attention") == 15
    assert kinds.count("full_attention") == 3
    assert kinds[:6] == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )
    model = build_model("gemma3")
    assert isinstance(model, DecoderLM)
    assert model.parameter_count == EXPECTED_PARAMETERS
    assert model.embedding.weight is model.lm_head.embedding.weight
    assert model.layers[0].attention.sliding_window == 512
    assert model.layers[5].attention.sliding_window is None
    assert model.layers[0].attention.q_norm.weight.shape == (256,)
    assert torch.count_nonzero(model.final_norm.weight) == 0


def test_forward_and_trainer_step():
    config = Gemma3Config()
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
