import gc
from pathlib import Path

from experiments.config import load_experiment_config
from experiments.data import StatefulBatchSampler
from experiments.models import build_experiment_model


ROOT = Path(__file__).resolve().parents[1]


def test_all_twelve_configs_build_with_exact_parameter_counts():
    paths = sorted((ROOT / "configs" / "training").glob("*/*.json"))
    assert len(paths) == 12
    for path in paths:
        config = load_experiment_config(path)
        model = build_experiment_model(config)
        assert model.parameter_count == config.expected_parameters
        assert model.config.vocab_size == 49_152
        assert model.config.max_seq_len == 8192
        assert config.sequence_length == 2048
        del model
        gc.collect()


def test_size_budgets_and_shared_training_policy():
    expected = {
        "25m": (100_000_000, 6e-4),
        "50m": (500_000_000, 4.5e-4),
        "100m": (1_000_000_000, 3e-4),
    }
    grouped = {size: [] for size in expected}
    for path in (ROOT / "configs" / "training").glob("*/*.json"):
        config = load_experiment_config(path)
        grouped[config.size].append(config)
    for size, configs in grouped.items():
        assert len(configs) == 4
        target_tokens, learning_rate = expected[size]
        for config in configs:
            assert config.target_train_tokens == target_tokens
            assert config.learning_rate == learning_rate
            assert config.tokens_per_optimizer_step == 524_288
            assert config.betas == (0.9, 0.95)
            assert config.warmup_ratio == 0.02


def test_stateful_sampler_resume_reproduces_remaining_batches():
    uninterrupted = StatefulBatchSampler(20, batch_size=4, seed=42)
    uninterrupted_batches = list(uninterrupted)

    partial = StatefulBatchSampler(20, batch_size=4, seed=42)
    iterator = iter(partial)
    first_two = [next(iterator), next(iterator)]
    state = partial.state_dict()

    resumed = StatefulBatchSampler(20, batch_size=4, seed=42)
    resumed.load_state_dict(state)
    assert first_two + list(resumed) == uninterrupted_batches
