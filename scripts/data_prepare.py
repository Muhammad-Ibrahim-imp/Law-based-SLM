import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from data.cleaner import clean_legal_text
from data.dedup import hash_text


def prepare_data(
    subset: str = "atticus_contracts",
    max_documents: int = 200,
    # documents will be divided into chunks of text (shards) for easier processing
    docs_per_shard: int = 50,
    output_dir: str = "data/processed",
    min_chars: int = 200,
):
    """
    Stream documents from Pile of Law, clean them, remove exact duplicates,
    and save the result as sharded .txt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset(
        "pile-of-law/pile-of-law",
        subset,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    seen_hashes = set()
    current_shard = []
    shard_index = 0
    total_kept = 0
    total_seen = 0
    total_duplicates = 0
    total_too_short = 0

    def write_shard(shard_docs, index):
        shard_path = os.path.join(output_dir, f"shard_{index:04d}.txt")
        with open(shard_path, "w", encoding="utf-8") as f:
            for doc in shard_docs:
                f.write(doc)
                f.write("\n\n")
        return shard_path

    shard_paths = []

    for example in ds:
        if total_seen >= max_documents:
            break
        total_seen += 1

        raw_text = example.get("text", "")
        cleaned = clean_legal_text(raw_text)    # calling the clean_legal_text function from data/cleaner.py

        if len(cleaned) < min_chars:
            total_too_short += 1
            continue

        fingerprint = hash_text(cleaned)
        if fingerprint in seen_hashes:
            total_duplicates += 1
            continue
        seen_hashes.add(fingerprint)    # adding the fingerprint to the seen_hashes set to keep track of unique documents

        current_shard.append(cleaned)
        total_kept += 1

        if len(current_shard) >= docs_per_shard:    # if the current shard has reached the specified number of documents, write it to a file and reset for the next shard
            path = write_shard(current_shard, shard_index)
            shard_paths.append(path)
            print(f"Wrote {path} ({len(current_shard)} documents)")
            shard_index += 1
            current_shard = []

    # Write any remaining documents as a final, smaller shard
    if current_shard:   # if there are any documents left in the current shard after processing all examples, write them to a final shard file
        path = write_shard(current_shard, shard_index)
        shard_paths.append(path)
        print(f"Wrote {path} ({len(current_shard)} documents)")

    manifest = {
        "subset": subset,
        "total_documents_seen": total_seen,
        "total_documents_kept": total_kept,
        "total_exact_duplicates_removed": total_duplicates,
        "total_too_short_removed": total_too_short,
        "num_shards": len(shard_paths),
        "shard_files": shard_paths,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n--- Summary ---")
    for key, value in manifest.items():
        if key != "shard_files":
            print(f"{key}: {value}")
    print(f"Manifest saved to {manifest_path}")

    return manifest


if __name__ == "__main__":
    prepare_data(
        subset="atticus_contracts",
        max_documents=200,
        docs_per_shard=50,
    )