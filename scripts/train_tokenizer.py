import sys
import os
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.bpe_trainer import train_bpe_tokenizer

if __name__ == "__main__":
    shard_files = sorted(glob.glob("data/processed/shard_*.txt"))
    print(f"Training on {len(shard_files)} shard files: {shard_files}")
    train_bpe_tokenizer(
        files=shard_files,
        vocab_size=32000,
        save_path="tokenizer/bpe_tokenizer.json",
    )