"""Merge per-GPU benchmark CSVs and build cross-GPU rankings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmarks.reporting import FIELDS, write_rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/all_gpus.csv"),
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_rankings(args.output.with_suffix(".md"), rows)

    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("no successful configurations to rank")
    fastest = max(successful, key=lambda row: float(row["tokens_per_second"]))
    cheapest = min(
        successful, key=lambda row: float(row["cost_per_1b_tokens_usd"])
    )
    print(
        "Fastest: "
        f"{fastest['gpu_name']}, seq={fastest['sequence_length']}, "
        f"micro_batch={fastest['micro_batch_size']}, "
        f"{float(fastest['tokens_per_second']):,.1f} tokens/s"
    )
    print(
        "Cheapest: "
        f"{cheapest['gpu_name']}, seq={cheapest['sequence_length']}, "
        f"micro_batch={cheapest['micro_batch_size']}, "
        f"${float(cheapest['cost_per_1b_tokens_usd']):,.4f}/1B tokens"
    )


if __name__ == "__main__":
    main()
