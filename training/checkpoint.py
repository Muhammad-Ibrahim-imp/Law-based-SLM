from pathlib import Path
from dataclasses import asdict, is_dataclass

import torch

def _to_dict_(config):
    """Converts dicts, dataclasses or standard class instances into a dictionary"""
    if isinstance(config, dict):
        return config
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return vars(config)
    return dict(config)

def save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        global_step,
        micro_step,
        model_config,
        training_config,
        metadata = None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "global_step": global_step,
        "micro_step": micro_step,
        "model_config": _to_dict_(model_config),
        "training_config": _to_dict_(training_config),
        "metadata": metadata or {}
    }

    torch.save(checkpoint, path)