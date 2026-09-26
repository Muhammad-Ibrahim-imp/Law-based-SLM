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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "micro_step": micro_step,
        "model_config": _to_dict_(model_config),
        "training_config": _to_dict_(training_config),
        "metadata": metadata or {}
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist")
    checkpoint = torch.load(path, map_location=device)

    required_keys = {"model_state_dict",
                     "optimizer_state_dict",
                     "scheduler_state_dict",
                     "global_step",
                     "micro_step",
                     "model_config",
                     "training_config",
                     "metadata"}

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise ValueError(f"Invalid checkpoint. Missing keys in checkpoint: {missing_keys}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint