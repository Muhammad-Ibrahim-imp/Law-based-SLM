import torch
import torch.nn.functional as F

def causal_language_model_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100 ) -> torch.Tensor:
    B, T, V = logits.shape
    logits = logits.reshape(B*T, V)
    labels = labels.reshape(B*T)

    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index = ignore_index)

    return loss