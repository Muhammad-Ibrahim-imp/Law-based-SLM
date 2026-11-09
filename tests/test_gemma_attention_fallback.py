from types import SimpleNamespace

from architectures import gemma3


def test_cuda_build_without_flash_uses_standard_attention(monkeypatch):
    monkeypatch.setattr(gemma3.torch.backends.cuda, "is_flash_attention_available", lambda: False)

    def forbidden_flash_context(**kwargs):
        raise AssertionError("must not force an unavailable Flash Attention kernel")

    monkeypatch.setattr(gemma3, "sdpa_kernel", forbidden_flash_context)
    expected = object()

    def attention(q, k, v, **kwargs):
        assert kwargs["is_causal"] and kwargs["enable_gqa"]
        assert kwargs["scale"] == 0.5
        return expected

    monkeypatch.setattr(gemma3.F, "scaled_dot_product_attention", attention)
    assert gemma3._attend(SimpleNamespace(is_cuda=True), None, None, None, 0.5) is expected
