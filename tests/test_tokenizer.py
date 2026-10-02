from tokenizer import load_smollm2_tokenizer, get_tokenizer_fingerprint

def test_encode_decode_roundtrip():
    tok = load_smollm2_tokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert decoded.strip() == text.strip()

def test_special_token_ids_exist():
    tok = load_smollm2_tokenizer()
    assert tok.eos_token_id is not None
    assert tok.vocab_size == 49152  # SmolLM2 tokenizer, fixed vocab

def test_fingerprint_is_deterministic():
    assert get_tokenizer_fingerprint() == get_tokenizer_fingerprint()
    assert len(get_tokenizer_fingerprint()) == 16

def test_tokenizer_is_cached_singleton():
    assert load_smollm2_tokenizer() is load_smollm2_tokenizer()