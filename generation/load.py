"""Load a sweep checkpoint into a freshly built model."""

from __future__ import annotations

from pathlib import Path

import torch

from experiments.config import load_experiment_config
from experiments.models import build_experiment_model


def load_trained_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Build the config's model and load its weights.

    A training checkpoint includes the optimizer, so the first load writes a
    weights-only file beside it. The next launch reads that smaller file.
    ``torch.compile`` stores keys under ``_orig_mod.``; that prefix is removed
    when every key has it. On CUDA the model is stored in BF16.
    """
    config = load_experiment_config(config_path)
    model = build_experiment_model(config)
    state = _load_weights(Path(checkpoint_path))
    model.load_state_dict(state)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    return model.eval()


def _load_weights(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_name(path.stem + ".weights.pt")
    if path.suffix == ".pt" and sidecar.is_file() and sidecar != path:
        loaded = torch.load(sidecar, map_location="cpu", weights_only=True)
        print(f"loaded weights from {sidecar.name}", flush=True)
        return _as_state_dict(loaded)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    state = _as_state_dict(loaded)
    if "model_state_dict" in loaded and not sidecar.exists():
        torch.save(state, sidecar)
        print(f"wrote {sidecar.name} for the next launch", flush=True)
    return state


def _as_state_dict(loaded) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state = loaded["model_state_dict"]
    elif isinstance(loaded, dict) and loaded and all(
        torch.is_tensor(value) for value in loaded.values()
    ):
        state = loaded
    else:
        raise ValueError("checkpoint has no model weights")
    return _strip_compile_prefix(state)


def _strip_compile_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if state and all(key.startswith(prefix) for key in state):
        return {key.removeprefix(prefix): value for key, value in state.items()}
    return state
