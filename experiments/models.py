"""Build any sweep model without changing the architecture implementations."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from experiments.config import ExperimentConfig


def build_experiment_model(config: ExperimentConfig) -> nn.Module:
    model_config = _build_model_config(config.architecture, config.model)
    if config.architecture == "gqa":
        from model.model import DecoderLM

        model = DecoderLM(model_config)
    else:
        from architectures.registry import build_model

        model = build_model(config.architecture, model_config)

    actual = model.parameter_count
    if actual != config.expected_parameters:
        raise RuntimeError(
            f"{config.name} has {actual:,} parameters; "
            f"config expects {config.expected_parameters:,}"
        )
    return model


def _build_model_config(architecture: str, values: dict[str, Any]):
    if architecture == "gqa":
        from model.config import ModelConfig

        config_type = ModelConfig
    elif architecture == "lfm25":
        from architectures.lfm25 import LFM25Config

        config_type = LFM25Config
        values = dict(values)
        values["layer_types"] = tuple(values["layer_types"])
    elif architecture == "qwen3":
        from architectures.qwen3 import Qwen3Config

        config_type = Qwen3Config
    elif architecture == "gemma3":
        from architectures.gemma3 import Gemma3Config

        config_type = Gemma3Config
    else:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    return config_type(**values)
