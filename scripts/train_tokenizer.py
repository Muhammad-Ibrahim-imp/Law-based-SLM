import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe_trainer import train_bpe_tokenizer

if __name__ == "__main__":
    train_bpe_tokenizer(
        files=["data/raw/sample.txt"],
        vocab_size=32000,
        save_path="tokenizer/bpe_tokenizer.json",
    )