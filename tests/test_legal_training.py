import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evaluation.validation import evaluate_validation
from experiments.config import load_experiment_config
from experiments import runner
from training.trainer import Trainer


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)
        self.config = SimpleNamespace(vocab_size=8)
        self.parameter_count = sum(p.numel() for p in self.parameters())

    def forward(self, ids):
        return self.head(self.embedding(ids))


def test_masked_accumulation_matches_combined_batch_and_partial_step():
    torch.manual_seed(42)
    model = TinyLM()
    combined = copy.deepcopy(model)
    batch = {"input_ids": torch.tensor([[1, 2, 3], [3, 2, 1]]),
             "labels": torch.tensor([[2, -100, -100], [2, 1, 4]])}
    def trainer(m, accumulation):
        optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        return Trainer(m, optimizer, scheduler, torch.device("cpu"),
                       max_grad_norm=100, gradient_accumulation_steps=accumulation,
                       normalize_by_tokens=True)
    accumulated = trainer(model, 8)
    single = trainer(combined, 1)
    for i in range(2):
        accumulated.train_step({k: v[i:i+1] for k, v in batch.items()},
                               finish_accumulation=i == 1)
    single.train_step(batch)
    assert accumulated.global_step == single.global_step == 1
    for actual, expected in zip(model.parameters(), combined.parameters()):
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-4)


def test_validation_excludes_masked_tokens():
    model = TinyLM()
    batch = {"input_ids": torch.tensor([[1, 2, 3]]),
             "labels": torch.tensor([[2, -100, -100]])}
    actual = evaluate_validation(model, [batch], torch.device("cpu"))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = nn.functional.cross_entropy(model(batch["input_ids"])[:, 0], torch.tensor([2]))
    assert actual["validation_tokens"] == 1
    assert actual["validation_loss"] == pytest.approx(float(expected.detach()), abs=1e-5)


def test_sft_one_pass_initialization_and_resume(tmp_path, monkeypatch):
    torch.manual_seed(42)
    model = TinyLM()
    original = tmp_path / "general.pt"
    torch.save({"model_state_dict": {"_orig_mod." + k: v for k, v in model.state_dict().items()}}, original)
    original_bytes = original.read_bytes()
    config = load_experiment_config(Path(__file__).parents[1] / "configs/sft/gemma3_200m_family_sft.json")
    config = replace(config, sequence_length=3, micro_batch_size=2,
                     gradient_accumulation_steps=2, target_train_tokens=15,
                     initialize_from=str(original), expected_parameters=model.parameter_count,
                     log_every_steps=1, eval_every_steps=1, checkpoint_every_steps=1)
    seen = []
    def dataset(root, mix, sequence_length):
        seen.append(mix)
        return [{"input_ids": torch.tensor([1, 2, 3]),
                 "labels": torch.tensor([2, -100, 4])} for _ in range(5)]
    monkeypatch.setattr(runner, "SFTDataset", dataset)
    monkeypatch.setattr(runner, "build_experiment_model", lambda _: TinyLM())
    result = runner.run_training(config, tmp_path, tmp_path / "runs", torch.device("cpu"))
    assert result["optimizer_steps"] == 2
    assert result["actual_train_tokens"] == 15
    assert seen[0].count(("casehold", "train")) == 2
    checkpoint_path = tmp_path / "runs" / config.name / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["micro_step"] == 3
    assert checkpoint["runtime_state"]["sampler"]["batch_index"] == 3
    resumed = runner.run_training(config, tmp_path, tmp_path / "runs", torch.device("cpu"))
    assert resumed["optimizer_steps"] == 2
    assert original.read_bytes() == original_bytes
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()
    events = [json.loads(line) for line in checkpoint_path.with_name("metrics.jsonl").read_text().splitlines()]
    assert events[-1]["tokens_seen"] == 15


def test_legal_configs_use_separate_domains_and_outputs():
    root = Path(__file__).parents[1]
    for domain in ("family", "criminal"):
        config = load_experiment_config(root / f"configs/legal/gemma3_200m_{domain}_legal.json")
        assert config.train_category == f"{domain}_law_train"
        assert config.validation_category == f"{domain}_law_validation"
        assert config.target_train_tokens == 4_700_000_000
        assert config.tokens_per_optimizer_step == 524_288
        assert config.total_optimizer_steps == 8965
        assert "preserved" in config.initialize_from
