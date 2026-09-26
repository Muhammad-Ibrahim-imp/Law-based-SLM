# scripts/export_training_samples.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_prepare import prepare_data


if __name__ == "__main__":
    sizes = {
        "sample_050": 50,
        "sample_1000": 1000,
        "sample_4000": 4000,
        "sample_12000": 12000,
    }

    for folder_name, size in sizes.items():
        print(f"\n=== Building {folder_name} ({size} documents) ===")
        prepare_data(
            subset="atticus_contracts",
            max_documents=size,
            docs_per_shard=500,
            output_dir=f"data/processed/{folder_name}",
        )