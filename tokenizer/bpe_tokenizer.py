from tokenizers import Tokenizer as HFTokenizer

class BPETokenizer:
    def __init__(self, path: str = "tokenizer/bpe_tokenizer.json"):
        self._tok = HFTokenizer.from_file(path)

        self.vocab_size = self._tok.get_vocab_size()
        self.pad_id = self._tok.token_to_id("<pad>")
        self.bos_id = self._tok.token_to_id("<bos>")
        self.eos_id = self._tok.token_to_id("<eos>")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)