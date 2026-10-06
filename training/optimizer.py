import torch
import torch.nn as nn

def adamw(model: nn.Module,
          learning_rate: float = 3e-4,
          betas: tuple[float,float] = (0.9, 0.95),
          eps: float = 1e-8,
          weight_decay: float = 0.1,
          fused: bool | None = None):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if param.ndim > 1:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    kwargs = {}
    if fused is not None:
        kwargs["fused"] = fused
    optimzer = torch.optim.AdamW([
            {
            "params": decay_params,
            "weight_decay": weight_decay,
            },
            {
            "params": no_decay_params,
            "weight_decay": 0.0,
            }
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        **kwargs)

    return optimzer