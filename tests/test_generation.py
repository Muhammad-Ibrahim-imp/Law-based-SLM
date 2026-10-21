import torch
from torch import nn

from common.interfaces import GenerationConfig
from generation.generate import generate
from generation.load import _strip_compile_prefix
from generation.sampling import sample_token


class _Prefer(nn.Module):
    def __init__(self, preferred: int, max_seq_len: int = 8, vocab_size: int = 5):
        super().__init__()
        self.config = type("Config", (), {"max_seq_len": max_seq_len})()
        self.preferred = preferred
        self.vocab_size = vocab_size

    def forward(self, input_ids, position_offset: int = 0):
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], self.vocab_size), -100.0
        )
        logits[..., self.preferred] = 10.0
        return logits


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 3.0, 2.0], [5.0, 0.0, 1.0]])
    chosen = sample_token(logits, GenerationConfig(temperature=0.0, top_k=1))
    assert chosen.tolist() == [[1], [0]]


def test_top_k_keeps_only_the_largest():
    logits = torch.tensor([[1.0, 4.0, 3.0]])
    chosen = sample_token(
        logits,
        GenerationConfig(temperature=1.0, top_k=1),
        generator=torch.Generator().manual_seed(0),
    )
    assert chosen.tolist() == [[1]]


def test_top_p_keeps_the_leading_token():
    logits = torch.tensor([[0.0, 0.0, 20.0]])
    chosen = sample_token(
        logits,
        GenerationConfig(temperature=1.0, top_p=0.1),
        generator=torch.Generator().manual_seed(0),
    )
    assert chosen.tolist() == [[2]]


def test_generate_appends_the_preferred_token():
    model = _Prefer(preferred=3, max_seq_len=6)
    prompt = torch.tensor([[1, 1]])
    output = generate(model, prompt, GenerationConfig(max_new_tokens=2, temperature=0.0))
    assert output.tolist() == [[1, 1, 3, 3]]


def test_generate_stops_at_eos_and_pads_the_batch():
    model = _Prefer(preferred=4, max_seq_len=8)
    prompt = torch.tensor([[1], [1]])
    output = generate(
        model,
        prompt,
        GenerationConfig(max_new_tokens=4, temperature=0.0),
        eos_id=4,
    )
    assert output.tolist() == [[1, 4], [1, 4]]


def test_generate_stops_at_the_context_limit():
    model = _Prefer(preferred=2, max_seq_len=4)
    prompt = torch.tensor([[1, 1, 1]])
    output = generate(model, prompt, GenerationConfig(max_new_tokens=10, temperature=0.0))
    assert output.shape == (1, 4)
    assert output[0, -1].item() == 2


def test_generate_rejects_a_full_prompt():
    model = _Prefer(preferred=1, max_seq_len=2)
    try:
        generate(
            model,
            torch.tensor([[1, 1]]),
            GenerationConfig(max_new_tokens=1, temperature=0.0),
        )
    except ValueError as error:
        assert "max_seq_len" in str(error)
    else:
        raise AssertionError("expected a full prompt to be rejected")


def test_compile_prefix_is_removed_only_when_every_key_has_it():
    stripped = _strip_compile_prefix({"_orig_mod.weight": torch.zeros(1)})
    assert set(stripped) == {"weight"}
    kept = _strip_compile_prefix({"weight": torch.zeros(1)})
    assert set(kept) == {"weight"}


def _tiny_gemma():
    from architectures.gemma3 import DecoderLM, Gemma3Config

    config = Gemma3Config(
        vocab_size=128,
        max_seq_len=32,
        d_model=32,
        n_layers=6,
        n_heads=4,
        n_kv_heads=1,
        head_dim=8,
        d_ff=64,
        sliding_window=4,
        query_pre_attn_scalar=8,
    )
    return DecoderLM(config).eval()


def _tiny_lfm():
    from architectures.lfm25 import DecoderLM, LFM25Config

    config = LFM25Config(
        vocab_size=128,
        max_seq_len=32,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        block_auto_adjust_ff_dim=False,
        conv_kernel=3,
        layer_types=("conv", "full_attention", "conv"),
    )
    return DecoderLM(config).eval()


def _cached_matches_full(model, steps: int):
    torch.manual_seed(0)
    tokens = torch.randint(0, 128, (2, steps))
    with torch.no_grad():
        full = model(tokens)
        cache = model.build_cache(tokens.shape[0])
        pieces = [
            model(tokens[:, index : index + 1], cache=cache)
            for index in range(steps)
        ]
    cached = torch.cat(pieces, dim=1)
    assert torch.allclose(cached, full, atol=1e-4, rtol=1e-4)
    attention = next(slot for slot in cache.slots if getattr(slot, "k", None) is not None)
    assert attention.k.shape[2] == model.config.max_seq_len


def test_gemma_cache_matches_full_forward_past_the_sliding_window():
    _cached_matches_full(_tiny_gemma(), steps=10)


def test_lfm_cache_matches_full_forward_including_convolution():
    _cached_matches_full(_tiny_lfm(), steps=7)


def test_cached_generate_matches_recomputing_the_prefix():
    model = _tiny_gemma()
    prompt = torch.randint(0, 128, (1, 4))
    generated = generate(
        model, prompt, GenerationConfig(max_new_tokens=3, temperature=0.0)
    )
    manual = prompt
    with torch.no_grad():
        for _ in range(3):
            nxt = model(manual)[:, -1, :].argmax(dim=-1, keepdim=True)
            manual = torch.cat([manual, nxt], dim=1)
    assert generated.tolist() == manual.tolist()


def test_weights_file_drops_the_optimizer(tmp_path):
    from generation.load import _load_weights

    source = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(2)},
            "optimizer_state_dict": {"moment": torch.ones(1000)},
        },
        source,
    )
    state = _load_weights(source)
    assert torch.equal(state["weight"], torch.ones(2))
    sidecar = tmp_path / "model.weights.pt"
    assert sidecar.is_file()
    assert sidecar.stat().st_size < source.stat().st_size
    assert torch.equal(_load_weights(source)["weight"], torch.ones(2))
