import torch
import torch.nn as nn

from training.gradient import clip_gradients
from training.loss import causal_language_model_loss

def mixed_precision_training(model: nn.Module,
                             input_ids: torch.Tensor,
                             label: torch.Tensor,
                             optimizer: torch.optim.Optimizer,
                             max_grad_norm: float = 1.0,):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):
        logits = model(input_ids)
        loss = causal_language_model_loss(logits, label)
    loss.backward()
    grad_norm = clip_gradients(model, max_grad_norm)

    optimizer.step()

    return loss.detach(), grad_norm