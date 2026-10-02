from transformers import AutoTokenizer

_tokenizer = None

def load_smollm2_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    return _tokenizer