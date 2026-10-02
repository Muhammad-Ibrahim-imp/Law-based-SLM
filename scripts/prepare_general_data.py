"""
scripts/prepare_general_data.py

Thin entry point for building the general pretraining data slice.
All reusable logic lives in data/general_prep.py, per README section 17.

README target for this slot: 5.0B tokens (see section 3). At the finalized
shard size this yields 19 shards (5.0B / 268,435,456 ≈ 18.6 -> 19, with the
last shard partially filled).

Shard/training spec this is pretokenized for:
    Shard size:     268,435,456 tokens (512 MiB/shard, uint16)
    Sequence length: 2,048
    GPU: H200 | Micro-batch: 32 | Accumulation: 8 | Global batch: 524,288 tokens

Usage:

# 10M-token test
python -m scripts.prepare_general_data --target-tokens 10_000_000

# Full 5B-token general corpus
python -m scripts.prepare_general_data --target-tokens 5_000_000_000

# Full corpus with custom dataset location (USE THIS FOR ACTUAL COLLECTION)
python -m scripts.prepare_general_data --target-tokens 5_000_000_000 --dataset-root ./dataset
"""

import argparse

from data.general_prep import build_shards, push_to_hub, validate_shards

TRAINING_CONFIG = {
    "gpu": "H200",
    "sequence_length": 2048,
    "micro_batch": 32,
    "grad_accumulation": 8,
    "global_batch_tokens": 524_288,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-tokens", type=int, default=5_000_000_000,
                    help="Total tokens to collect for the general slot "
                         "(README target: 5.0B).")
    p.add_argument("--fineweb-ratio", type=float, default=0.8,
                    help="Fraction of tokens drawn from fineweb-edu-dedup "
                         "(remainder from cosmopedia-v2). Confirm with team.")
    p.add_argument("--shard-tokens", type=int, default=268_435_456,
                    help="Tokens per shard file. Default is the project's "
                         "finalized shard size (2^28 tokens, 512 MiB/shard).")
    p.add_argument("--sequence-length", type=int, default=2048,
                    help="Training-side sequence length, recorded in the "
                         "manifest for the downstream dataloader.")
    p.add_argument("--dataset-root", type=str, default="./dataset",
                    help="Shared dataset root. Shards go to "
                         "<dataset-root>/general/, manifest.json lives at "
                         "<dataset-root>/manifest.json alongside the other "
                         "categories (family_law, criminal_law, sft).")
    p.add_argument("--push-repo", type=str, default=None,
                    help="If set, pushes the whole dataset root to this HF "
                         "Hub dataset repo id, e.g. your-org/law-slm-dataset")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--skip-validate", action="store_true",
                    help="Skip the post-build integrity check (not recommended).")
    return p.parse_args()


def main():
    args = parse_args()

    build_shards(
        target_tokens=args.target_tokens,
        dataset_root=args.dataset_root,
        category="general",
        shard_tokens=args.shard_tokens,
        fineweb_ratio=args.fineweb_ratio,
        sequence_length=args.sequence_length,
        training_config=TRAINING_CONFIG,
    )

    if not args.skip_validate:
        validate_shards(args.dataset_root, category="general")

    if args.push_repo:
        push_to_hub(args.dataset_root, args.push_repo, private=args.private)


if __name__ == "__main__":
    main()