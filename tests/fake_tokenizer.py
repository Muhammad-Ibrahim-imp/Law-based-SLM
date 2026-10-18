"""Character-level stand-in for the SmolLM2 tokenizer, for unit tests that
must run without downloading it. Mirrors the parts of the HF interface the
chat code uses. IDs: 0 <|endoftext|>, 1 <|im_start|>, 2 <|im_end|>, 3 <x>,
then 10 + ord(c) for ASCII characters."""

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<x>"]


class FakeTokenizer:
    eos_token_id = 0
    all_special_tokens = SPECIALS

    def __init__(self, im_start_id: int = 1):
        self._special = {s: i for i, s in enumerate(SPECIALS)}
        self._special["<|im_start|>"] = im_start_id

    def convert_tokens_to_ids(self, tok):
        return self._special[tok]

    def _enc(self, text):
        return [10 + ord(c) for c in text]

    def __call__(self, texts, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("chat code must not add special tokens")
        return {"input_ids": [self._enc(t) for t in texts]}

    def decode(self, ids):
        inv = {i: s for s, i in self._special.items()}
        return "".join(inv[i] if i in inv else chr(i - 10) for i in ids)
