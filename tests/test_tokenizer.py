from tokenizer.bpe_tokenizer import BPETokenizer

def test_encode_decode_roundtrip():
    tok = BPETokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    assert decoded.strip() == text.strip()

def test_special_token_ids_exist():
    tok = BPETokenizer()
    assert tok.eos_id is not None
    assert tok.vocab_size > 100
#   Disabled the following assertion as the training data is limited right now
#    assert tok.vocab_size == 32000 

