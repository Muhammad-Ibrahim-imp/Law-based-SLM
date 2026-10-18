"""Chat format shared by SFT data prep, training and inference.

The SmolLM2 base tokenizer ships no ``chat_template``, so the project defines
its own ChatML-style format here. Conversations are assembled as token IDs turn
by turn (never by templating a string and tokenizing it), so the loss mask is
exact.

Format of one turn::

    <|im_start|>{role}\\n{content}<|im_end|>\\n

* Trained tokens (mask = 1): an assistant turn's content and its closing
  ``<|im_end|>``, so the model learns to stop.
* Not trained (mask = 0): ``<|im_start|>``, the role header, the newline after
  ``<|im_end|>``, and every token of system/user turns.
* No default system prompt is injected. A system turn appears only if the
  data has one.
* Generation: build the prompt with ``add_generation_prompt=True`` (it ends in
  ``<|im_start|>assistant\\n``) and stop on ``<|im_end|>``.

Training and inference MUST both use :func:`encode_chat`: encoding the pieces
separately can differ slightly from encoding the rendered string as a whole.

This is a shared contract (README "shared contracts" rule). Bump
``CHAT_FORMAT_VERSION`` on any change and tell Members B and C.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

CHAT_FORMAT_VERSION = "chatml-v1"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ROLES = ("system", "user", "assistant")

# Expected IDs in the SmolLM2 vocab (verified on 14 Oct 2026). Checked against
# the tokenizer at runtime by get_chat_token_ids(); never used blindly.
EXPECTED_IM_START_ID = 1
EXPECTED_IM_END_ID = 2


class ChatFormatError(ValueError):
    """A conversation that cannot be encoded in the project chat format."""


def get_chat_token_ids(tokenizer) -> dict[str, int]:
    """Look up and verify the special-token IDs the chat format relies on."""
    im_start = tokenizer.convert_tokens_to_ids(IM_START)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    pad = tokenizer.eos_token_id
    if im_start != EXPECTED_IM_START_ID or im_end != EXPECTED_IM_END_ID:
        raise RuntimeError(
            f"Unexpected chat token IDs: {IM_START}={im_start}, {IM_END}={im_end} "
            f"(expected {EXPECTED_IM_START_ID}, {EXPECTED_IM_END_ID}). "
            "Wrong tokenizer checkpoint?"
        )
    if pad is None:
        raise RuntimeError("Tokenizer has no eos_token_id to use as the pad ID.")
    return {"im_start": im_start, "im_end": im_end, "pad": pad}


def _encode_many(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Batch-encode texts with no special tokens added."""
    if not texts:
        return []
    out = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
    return [list(x) for x in out]


class ChatEncoder:
    """Encodes conversations into (token IDs, loss mask).

    Role headers are encoded once and cached; message contents are
    batch-encoded per conversation, which keeps prep fast on ~0.5M examples.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        ids = get_chat_token_ids(tokenizer)
        self.im_start_id = ids["im_start"]
        self.im_end_id = ids["im_end"]
        self.pad_id = ids["pad"]
        headers = _encode_many(tokenizer, [f"{r}\n" for r in ROLES])
        self._header = {r: [self.im_start_id] + h for r, h in zip(ROLES, headers)}
        self._newline = _encode_many(tokenizer, ["\n"])[0]
        self._special_strings = tuple(
            s for s in getattr(tokenizer, "all_special_tokens", ()) if s
        )

    # ------------------------------------------------------------------ checks
    def find_special_text(self, messages: Iterable[Mapping[str, str]]) -> str | None:
        """Return the first special-token string appearing literally in any
        message content, or None. Such text would be tokenized into a real
        special token and corrupt the turn structure, so callers drop it."""
        for m in messages:
            content = m.get("content") or ""
            for s in self._special_strings:
                if s in content:
                    return s
        return None

    @staticmethod
    def check_messages(messages: Sequence[Mapping[str, str]]) -> None:
        if not messages:
            raise ChatFormatError("empty conversation")
        for i, m in enumerate(messages):
            role = m.get("role")
            if role not in ROLES:
                raise ChatFormatError(f"unknown role {role!r} at turn {i}")
            if role == "system" and i != 0:
                raise ChatFormatError(f"system turn at position {i} (only allowed first)")
            if not isinstance(m.get("content"), str):
                raise ChatFormatError(f"non-string content at turn {i}")

    # ---------------------------------------------------------------- encoding
    def encode(
        self,
        messages: Sequence[Mapping[str, str]],
        add_generation_prompt: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Encode a conversation.

        Returns ``(ids, mask)`` of equal length; ``mask[i] == 1`` means token
        ``i`` is a training target.
        """
        self.check_messages(messages)
        bodies = _encode_many(self.tokenizer, [m["content"] for m in messages])
        ids: list[int] = []
        mask: list[int] = []
        for m, body in zip(messages, bodies):
            head = self._header[m["role"]]
            body = body + [self.im_end_id]
            train = 1 if m["role"] == "assistant" else 0
            ids += head + body + self._newline
            mask += [0] * len(head) + [train] * len(body) + [0] * len(self._newline)
        if add_generation_prompt:
            head = self._header["assistant"]
            ids += head
            mask += [0] * len(head)
        return ids, mask

    def render(self, messages: Sequence[Mapping[str, str]],
               add_generation_prompt: bool = False) -> str:
        """Human-readable string of the format (debugging/docs only; do not
        tokenize this for training or inference)."""
        parts = [f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages]
        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")
        return "".join(parts)


def encode_chat(tokenizer, messages, add_generation_prompt: bool = False):
    """Convenience wrapper; prefer a reused :class:`ChatEncoder` in loops."""
    return ChatEncoder(tokenizer).encode(messages, add_generation_prompt)
