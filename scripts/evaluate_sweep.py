"""Evaluate trained sweep checkpoints with one shared protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evaluation.multiple_choice import (
    evaluate_multiple_choice,
    load_benchmark,
)
from experiments.config import load_experiment_config
from experiments.models import build_experiment_model
from scripts.train_sweep import discover_configs
from tokenizer import load_smollm2_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("25m", "50m", "100m", "all"), default="all")
    parser.add_argument("--name")
    parser.add_argument("--output-root", default="./results/training_sweep")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--benchmarks",
        default="hellaswag,arc_easy,piqa",
        help="Comma-separated benchmark names",
    )
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = load_smollm2_tokenizer()
    benchmark_names = [name.strip() for name in args.benchmarks.split(",")]
    datasets = {name: load_benchmark(name) for name in benchmark_names}

    for config_path in discover_configs(args.size, args.name):
        config = load_experiment_config(config_path)
        run_dir = Path(args.output_root) / config.name
        checkpoint_path = run_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = build_experiment_model(config).to(device)
        state = checkpoint["model_state_dict"]
        if state and all(key.startswith("_orig_mod.") for key in state):
            state = {
                key.removeprefix("_orig_mod."): value
                for key, value in state.items()
            }
        model.load_state_dict(state)

        results = []
        for benchmark_name, rows in datasets.items():
            started = time.perf_counter()
            result = evaluate_multiple_choice(
                model,
                tokenizer,
                rows,
                benchmark_name,
                device,
                max_examples=args.max_examples,
            )
            result["elapsed_seconds"] = time.perf_counter() - started
            results.append(result)
            print(json.dumps({"experiment": config.name, **result}), flush=True)
        (run_dir / "benchmarks.json").write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
