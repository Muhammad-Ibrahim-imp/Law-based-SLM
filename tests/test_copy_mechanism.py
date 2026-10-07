import torch
from torch import nn

from model.config import ModelConfig
from model.model import DecoderLM
from training.copy_mechanism import CopyLM, mixture_nll


def test_mixture_matches_vocabulary_loss_when_gate_always_generates():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 11)
    hidden = torch.randn(2, 4, 5)
    context = torch.randint(0, 11, (2, 4))
    labels = torch.randint(0, 11, (2, 4))
    gate = nn.Linear(5, 1)
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, 20.0)

    mixed = mixture_nll(logits, hidden, context, labels, gate, reduction="none")
    vocab = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 11), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    assert torch.allclose(mixed, vocab, atol=1e-4)


def test_copy_path_assigns_the_repeated_context_token():
    logits = torch.full((1, 1, 6), -20.0)
    hidden = torch.ones(1, 1, 4)
    context = torch.tensor([[3]])
    labels = torch.tensor([[3]])
    gate = nn.Linear(4, 1)
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, -20.0)

    loss = mixture_nll(logits, hidden, context, labels, gate)
    assert float(loss) < 0.2


def test_copy_wrapper_loss_is_finite_and_trains():
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        d_ff=64,
    )
    model = CopyLM(DecoderLM(config))
    tokens = torch.randint(0, 32, (2, 9))
    loss = model(tokens[:, :-1], labels=tokens[:, 1:])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.generate_gate.weight.grad is not None
