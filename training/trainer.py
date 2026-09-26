import torch
import torch.nn as nn

from training.loss import causal_language_model_loss
from training.gradient import clip_gradients
from training.checkpoint import (save_checkpoint as save_training_checkpoint,
                                 load_checkpoint as load_training_checkpoint)

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
        self.optimizer.zero_grad(set_to_none=True)
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

    def save_checkpoint(self, path, model_config, training_config, metadata = None):
        save_training_checkpoint(path, self.model, self.optimizer,
                                 self.scheduler, self.global_step,
                                 self.micro_step, model_config,
                                 training_config, metadata)
    def load_checkpoint(self, path):
        checkpoint = load_training_checkpoint(path, self.model, self.optimizer,self.scheduler,self.device)
        self.global_step = checkpoint["global_step"]
        self.micro_step = checkpoint["micro_step"]

        self.optimizer.zero_grad(set_to_none=True)

        return checkpoint