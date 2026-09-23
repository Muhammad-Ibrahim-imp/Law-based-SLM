import torch
import torch.nn as nn

def clip_gradients(model: nn.Module,
                   max_norm:float = 1.0,
                   ) -> torch.Tensor:
    if max_norm <= 0:
        raise ValueError('max_norm must be > 0')
    # Modifies the value of all gradients so that the total norm remains
    # inside max norm. It returns the value of total norm before clipping
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                max_norm)
    return total_norm