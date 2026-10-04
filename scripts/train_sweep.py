"""Run one experiment or a sequential architecture/size sweep locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.config import load_experiment_config
from experiments.runner import run_smoke_test, run_training

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "training"
ARCHITECTURE_ORDER = {"gqa": 0, "qwen3": 1, "gemma3": 2, "lfm25": 3}


def discover_configs(size: str, name: str | None) -> list[Path]:
    paths = list(CONFIG_ROOT.glob("*/*.json"))
    if size != "all":
        paths = [path for path in paths if path.parent.name == size]
    if name is not None:
        paths = [path for path in paths if path.stem == name]
    configs = [(path, load_experiment_config(path)) for path in paths]
    configs.sort(
        key=lambda item: (
            int(item[1].size.removesuffix("m")),
            ARCHITECTURE_ORDER[item[1].architecture],
        )
    )
    if not configs:
        raise ValueError("no experiment configs matched the selection")
    return [path for path, _ in configs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--size", choices=("25m", "50m", "100m", "all"), default="all")
    parser.add_argument("--name", help="Run one config by filename stem")
    parser.add_argument("--dataset-root", default="./dataset")
    parser.add_argument("--output-root", default="./results/training_sweep")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    paths = discover_configs(args.size, args.name)
    if args.mode == "smoke":
        results = [
            run_smoke_test(load_experiment_config(path), device)
            for path in paths
        ]
        output = Path(args.output_root) / "smoke_results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    for path in paths:
        config = load_experiment_config(path)
        summary = run_training(
            config,
            args.dataset_root,
            args.output_root,
            device,
            resume=not args.no_resume,
            compile_model=args.compile,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
