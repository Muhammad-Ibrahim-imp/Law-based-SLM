import torch
import torch.nn as nn

from training.loss import causal_language_model_loss
from training.gradient import clip_gradients

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 device: torch.device,
                 max_grad_norm: float = 1.0,
                 gradient_accumulation_steps: int = 1,
                 ):
        if gradient_accumulation_steps <= 0:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be > 0".format(gradient_accumulation_steps))
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_step = 0
        self.global_step = 0
    def train_step(self,
                   inputs: torch.Tensor,
                   labels: torch.Tensor,
                   ):

        self.model.train()

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        with torch.autocast(device_type=self.device.type,
                        dtype=torch.bfloat16):
            logits = self.model(inputs)
            loss = causal_language_model_loss(logits,labels)

        logging_loss = loss.detach()
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        self.micro_step += 1

        grad_norm = None

        optimizer_step = False

        if self.micro_step % self.gradient_accumulation_steps == 0:
            grad_norm = clip_gradients(
                self.model,
                self.max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1

            optimizer_step = True

        return {
            "loss": logging_loss,
            "grad_norm": grad_norm,
            "optimizer_step": optimizer_step,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
        }