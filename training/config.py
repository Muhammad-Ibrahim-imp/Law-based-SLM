from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingConfig:

    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1

    max_grad_norm: float = 1.0

    warmup_steps: int = 1000
    total_steps: int = 10000
    min_lr_ratio: float = 0.1

    gradient_accumulation_steps: int = 1

    log_every: int = 10
    checkpoint_every: int = 100

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")

        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")

        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")

        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1.")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive."
            )

        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")

        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive.")