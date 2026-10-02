"""
tokenizer/

SmolLM2 tokenizer integration, per project README section 17 (`tokenizer/`
folder responsibilities): loading, encoding/decoding, special-token handling,
vocabulary analysis.

Vocabulary size: 49,152 (shared across the whole SmolLM2 family — 135M,
360M, 1.7B all use the identical tokenizer).

NOTE: per README section 20 (Collaboration Rules), the tokenizer interface
and configuration are a shared contract. Changes here (e.g. adding special
tokens) should be coordinated with Member B / Member C before merging.
"""

import hashlib
import json

from transformers import AutoTokenizer

TOKENIZER_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
VOCAB_SIZE = 49_152

_tokenizer = None
_fingerprint = None


def load_smollm2_tokenizer():
    """Load (and cache) the SmolLM2 tokenizer. CPU-only, no GPU required."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)
    return _tokenizer


def get_tokenizer_fingerprint() -> str:
    """
    A short, stable hash of the tokenizer's actual vocabulary, so every
    manifest can record *which exact tokenizer* produced its shards --
    not just a checkpoint name, which could in principle point to a
    checkpoint that's since changed upstream. Used for the manifest's
    required "tokenizer name/hash" field.
    """
    global _fingerprint
    if _fingerprint is None:
        tok = load_smollm2_tokenizer()
        vocab_items = sorted(tok.get_vocab().items())
        payload = json.dumps(vocab_items, sort_keys=True).encode("utf-8")
        _fingerprint = hashlib.sha256(payload).hexdigest()[:16]
    return _fingerprint