from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

import torch


# ============================================================
# TOKENIZER CONTRACT
# Owner: Member A
# Used by: A, B, C
# ============================================================

class Tokenizer(Protocol):
    vocab_size: int
    bos_id: int | None
    eos_id: int
    pad_id: int | None

    def encode(self, text: str) -> list[int]:
        """
        Convert text into token IDs.
        """
        ...

    def decode(self, ids: list[int]) -> str:
        """
        Convert token IDs back into text.
        """
        ...


# ============================================================
# TRAINING BATCH CONTRACT
# Owner: Member A produces it
# Used by: Member B
#
# Shapes:
# input_ids: [B, T]
# labels:    [B, T]
#
# dtype:
# torch.long
# ============================================================

class Batch(TypedDict):
    input_ids: torch.Tensor
    labels: torch.Tensor


# ============================================================
# MODEL OUTPUT CONTRACT
# Owner: Member B
# Used by: B, C
#
# logits shape:
# [B, T, V]
# ============================================================

@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


# ============================================================
# GENERATION CONFIG
# Owner: Member B
# Used by: B, C
# ============================================================

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128

    temperature: float = 1.0

    top_k: int | None = None
    top_p: float | None = None


# ============================================================
# MODEL CONTRACT
# Owner: Member B
# Used mainly by: Member C
#
# Member B's actual model should support these operations.
# ============================================================

class TrainableLM(Protocol):

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        """
        input_ids:
            [B, T]

        labels:
            [B, T] or None

        returns:
            ModelOutput
        """
        ...

    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        """
        Generate new token IDs.

        Returns:
            Tensor of shape [B, T + generated_tokens]
        """
        ...


# ============================================================
# EVALUATION RESULT CONTRACT
# Owner: Member C
# Used by: B, C
# ============================================================

@dataclass
class EvaluationResult:
    metrics: dict[str, float]

    num_tokens: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# EVALUATOR CONTRACT
# Owner: Member C
# ============================================================

class Evaluator(Protocol):

    def evaluate(
        self,
        model: TrainableLM,
        tokenizer: Tokenizer,
    ) -> EvaluationResult:
        """
        Evaluate a model and return standardized metrics.
        """
        ...


# ============================================================
# CHECKPOINT CONTRACT
# Owner: Member B
# Used by: B, C
# ============================================================

CHECKPOINT_VERSION = 1


@dataclass
class TrainerState:
    step: int
    tokens_seen: int
    checkpoint_version: int = CHECKPOINT_VERSION