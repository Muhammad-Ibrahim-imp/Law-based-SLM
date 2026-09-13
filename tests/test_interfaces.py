import torch

from common.interfaces import (
    Batch,
    EvaluationResult,
    GenerationConfig,
    ModelOutput,
    TrainerState,
)


def test_model_output():
    logits = torch.randn(2, 8, 100)

    output = ModelOutput(logits=logits)

    assert output.logits.shape == (2, 8, 100)
    assert output.loss is None


def test_batch_contract():
    batch: Batch = {
        "input_ids": torch.zeros(
            2, 8, dtype=torch.long
        ),
        "labels": torch.zeros(
            2, 8, dtype=torch.long
        ),
    }

    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_generation_config():
    config = GenerationConfig()

    assert config.max_new_tokens == 128
    assert config.temperature == 1.0


def test_trainer_state():
    state = TrainerState(
        step=100,
        tokens_seen=1_000_000,
    )

    assert state.step == 100
    assert state.tokens_seen == 1_000_000
    assert state.checkpoint_version == 1


def test_evaluation_result():
    result = EvaluationResult(
        metrics={
            "loss": 2.5,
            "perplexity": 12.18,
        },
        num_tokens=100_000,
    )

    assert result.metrics["loss"] == 2.5
    assert result.num_tokens == 100_000