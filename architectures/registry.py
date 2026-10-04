"""Select a decoder by name without importing architecture modules up front."""

from __future__ import annotations


def build_model(name: str, config=None):
    """Build one decoder.

    `name` selects the architecture. `config` defaults to its ~200M settings.
    """
    if not isinstance(name, str):
        raise TypeError("architecture name must be a string")
    key = name.strip().lower()
    if key == "lfm25":
        from architectures.lfm25 import DecoderLM, LFM25Config

        cfg = LFM25Config() if config is None else config
        if not isinstance(cfg, LFM25Config):
            raise TypeError("lfm25 requires an LFM25Config")
        return DecoderLM(cfg)
    raise ValueError(
        f"Unknown architecture {name!r}. Choose one of: {', '.join(implemented_names())}."
    )


def implemented_names() -> tuple[str, ...]:
    """Names that build_model can construct. `custom` is excluded."""
    return ("lfm25",)
