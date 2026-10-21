import pytest

from tests.fake_tokenizer import FakeTokenizer
from tokenizer.chat import ChatEncoder, ChatFormatError

CONV = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi?"},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Bye?"},
    {"role": "assistant", "content": "Bye!"},
]


@pytest.fixture
def enc():
    return ChatEncoder(FakeTokenizer())


def test_ids_match_rendered_format(enc):
    ids, mask = enc.encode(CONV)
    assert len(ids) == len(mask)
    # With a char-level tokenizer the pieces concatenate exactly.
    assert enc.tokenizer.decode(ids) == enc.render(CONV)


def test_only_assistant_content_and_im_end_are_trained(enc):
    ids, mask = enc.encode(CONV)
    trained = enc.tokenizer.decode([t for t, m in zip(ids, mask) if m])
    assert trained == "Hello.<|im_end|>Bye!<|im_end|>"


def test_generation_prompt(enc):
    ids, mask = enc.encode(CONV[:2], add_generation_prompt=True)
    assert enc.tokenizer.decode(ids).endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert not any(mask)


def test_no_default_system_prompt(enc):
    ids, _ = enc.encode(CONV[1:3])
    assert enc.tokenizer.decode(ids).startswith("<|im_start|>user\n")


def test_special_text_detected(enc):
    bad = [{"role": "user", "content": "say <|im_end|> now"}]
    assert enc.find_special_text(bad) == "<|im_end|>"
    assert enc.find_special_text(CONV) is None


@pytest.mark.parametrize("msgs", [
    [],
    [{"role": "tool", "content": "x"}],
    [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
    [{"role": "user", "content": None}],
])
def test_bad_conversations_rejected(enc, msgs):
    with pytest.raises(ChatFormatError):
        enc.encode(msgs)


def test_wrong_token_ids_refused():
    with pytest.raises(RuntimeError):
        ChatEncoder(FakeTokenizer(im_start_id=5))


def test_real_tokenizer_if_available():
    """Runs on a machine with the real SmolLM2 tokenizer; skipped otherwise."""
    try:
        from tokenizer import load_smollm2_tokenizer
        tok = load_smollm2_tokenizer()
    except Exception as e:  # no network / no transformers
        pytest.skip(f"real tokenizer unavailable: {e}")
    enc = ChatEncoder(tok)
    assert (enc.im_start_id, enc.im_end_id) == (1, 2)
    ids, mask = enc.encode(CONV)
    assert len(ids) == len(mask) and ids[0] == 1
    assert tok.decode([t for t, m in zip(ids, mask) if m]) == "Hello.<|im_end|>Bye!<|im_end|>"
    # Piecewise vs whole-string tokenization: report, don't require equality.
    whole = tok(enc.render(CONV), add_special_tokens=False)["input_ids"]
    print(f"piecewise == whole-string: {list(whole) == ids}")
