"""Prepare data and run the sweep sequentially on one Modal H200."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"
DATA_ROOT = "/vol/data/dataset"
OUTPUT_ROOT = "/vol/runs/training_sweep"
FUNCTION_TIMEOUT_SECONDS = 86_400

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "datasets==5.0.1",
        "numpy",
        "huggingface_hub",
        "tqdm",
    )
    .add_local_dir(ROOT, remote_path=REMOTE_ROOT, copy=True)
    .env({"PYTHONPATH": REMOTE_ROOT})
)
app = modal.App("law-slm-architecture-sweep", image=image)
data_volume = modal.Volume.from_name("law-slm-data", create_if_missing=True)
run_volume = modal.Volume.from_name("law-slm-training", create_if_missing=True)


@app.function(
    timeout=FUNCTION_TIMEOUT_SECONDS,
    cpu=8,
    memory=32_768,
    volumes={"/vol/data": data_volume},
)
def prepare_data(target_tokens: int = 1_000_000_000) -> None:
    subprocess.run(
        [
            "python",
            "-m",
            "scripts.prepare_general_data",
            "--target-tokens",
            str(target_tokens),
            "--dataset-root",
            DATA_ROOT,
            "--validation-ratio",
            "0.005",
        ],
        cwd=REMOTE_ROOT,
        check=True,
    )
    data_volume.commit()


@app.function(
    gpu="H200",
    timeout=FUNCTION_TIMEOUT_SECONDS,
    cpu=8,
    memory=32_768,
    volumes={"/vol/data": data_volume, "/vol/runs": run_volume},
)
def run_config(
    config_name: str,
    mode: str,
    compile_model: bool = False,
) -> dict[str, object] | list[dict[str, object]]:
    from experiments.config import load_experiment_config
    from experiments.runner import run_smoke_test, run_training
    import torch

    matches = list(
        Path(REMOTE_ROOT, "configs", "training").glob(f"*/{config_name}.json")
    )
    if len(matches) != 1:
        raise ValueError(f"expected one config named {config_name!r}, found {matches}")
    config = load_experiment_config(matches[0])
    device = torch.device("cuda")
    if mode == "smoke":
        result = run_smoke_test(config, device)
    elif mode == "profile":
        result = run_smoke_test(
            config,
            device,
            steps=3,
            sequence_length=config.sequence_length,
            batch_size=config.micro_batch_size,
            compile_model=compile_model,
            warmup_steps=1 if compile_model else 0,
        )
    elif mode == "train":
        data_volume.reload()
        result = run_training(
            config,
            DATA_ROOT,
            OUTPUT_ROOT,
            device,
            resume=True,
            compile_model=compile_model,
            checkpoint_callback=run_volume.commit,
        )
        run_volume.commit()
    elif mode == "evaluate":
        from evaluation.multiple_choice import (
            evaluate_multiple_choice,
            load_benchmark,
        )
        from experiments.models import build_experiment_model
        from tokenizer import load_smollm2_tokenizer
        import time

        run_volume.reload()
        checkpoint_path = Path(OUTPUT_ROOT, config.name, "checkpoint.pt")
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
        tokenizer = load_smollm2_tokenizer()
        result = []
        for benchmark_name in ("hellaswag", "arc_easy", "piqa"):
            started = time.perf_counter()
            metrics = evaluate_multiple_choice(
                model,
                tokenizer,
                load_benchmark(benchmark_name),
                benchmark_name,
                device,
            )
            metrics["elapsed_seconds"] = time.perf_counter() - started
            result.append(metrics)
        Path(OUTPUT_ROOT, config.name, "benchmarks.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        run_volume.commit()
    else:
        raise ValueError("mode must be 'smoke', 'profile', 'train', or 'evaluate'")
    return result


def config_names(size: str) -> list[str]:
    from scripts.train_sweep import discover_configs

    return [path.stem for path in discover_configs(size, None)]


@app.local_entrypoint()
def main(
    mode: str,
    size: str = "all",
    name: str = "",
    compile_model: bool = False,
    target_tokens: int = 1_000_000_000,
) -> None:
    if mode == "prepare-data":
        prepare_data.remote(target_tokens)
        return
    if mode not in {"smoke", "profile", "train", "evaluate"}:
        raise ValueError(
            "--mode must be prepare-data, smoke, profile, train, or evaluate"
        )
    names = [name] if name else config_names(size)
    results = []
    for config_name in names:
        result = run_config.remote(config_name, mode, compile_model)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    if mode in {"smoke", "profile"}:
        filename = (
            "modal_smoke_results.json"
            if mode == "smoke"
            else (
                f"modal_profile_compiled_{size}_results.json"
                if compile_model
                else "modal_profile_results.json"
            )
        )
        output = ROOT / "results" / "training_sweep" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
