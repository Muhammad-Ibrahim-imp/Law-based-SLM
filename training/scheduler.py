import math

import torch

def build_scheduler_cosine(optimizer: torch.optim.Optimizer,
                    total_steps: int,
                    warmup_steps: int,
                    min_lr_ratio: float = 0.1):
    if total_steps <= 0:
        raise ValueError('total_steps must be positive')
    if warmup_steps < 0:
        raise ValueError('warm_up_steps cant be negative')
    if total_steps < warmup_steps:
        raise ValueError('total_steps cant be smaller than warmup_steps')
    if not 0<=min_lr_ratio<=1:
        raise ValueError('min_lr_ration must be between 0 and 1')

    def lr_lambda(current_step: int):

        if current_step < warmup_steps:
            return current_step / max(1,warmup_steps)

        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Here we have taken 1 as max_lr_ratio
        return min_lr_ratio + (1-min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler