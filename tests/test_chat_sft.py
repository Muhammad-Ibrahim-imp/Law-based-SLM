from types import SimpleNamespace

import pytest
import torch

from scripts import chat_sft
from tokenizer.chat import ChatEncoder
from tests.fake_tokenizer import FakeTokenizer


def test_sft_reply_uses_conversation_and_assistant_stop(monkeypatch):
    tokenizer = FakeTokenizer()
    decode = tokenizer.decode
    tokenizer.decode = lambda ids, **kwargs: decode(ids)
    encoder = ChatEncoder(tokenizer)
    messages = [{"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"}]

    def generate(model, prompt, config, eos_id):
        assert tokenizer.decode(prompt[0].tolist()).endswith("Again<|im_end|>\n<|im_start|>assistant\n")
        assert "Hello" in tokenizer.decode(prompt[0].tolist())
        assert eos_id == 2 and config.temperature == 0.0
        return torch.cat([prompt, torch.tensor([[10 + ord("O"), 10 + ord("K"), 2]])], dim=1)

    monkeypatch.setattr(chat_sft, "generate", generate)
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=8192))
    assert chat_sft.reply(model, encoder, messages, torch.device("cpu"), 16) == "OK"


def test_sft_reply_refuses_overlong_history():
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=8))
    with pytest.raises(ValueError, match="/reset"):
        chat_sft.reply(model, ChatEncoder(FakeTokenizer()),
                       [{"role": "user", "content": "long question"}], torch.device("cpu"), 16)
