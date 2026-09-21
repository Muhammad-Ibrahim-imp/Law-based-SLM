import torch
import torch.nn as nn

from training.loss import causal_language_model_loss

def training_step(input_ids: torch.Tensor,
                  labels: torch.Tensor,
                  model: nn.Module) -> torch.Tensor:

    logits = model(input_ids)

    loss = causal_language_model_loss(
        logits,
        labels
    )
    
    return loss
