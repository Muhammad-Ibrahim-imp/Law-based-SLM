"""Launch benchmark jobs on the four requested Modal GPU types."""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"
GPU_CONFIGS = {
    "b300": ("B300", "Nvidia B300"),
    "b200": ("B200", "Nvidia B200"),
    "h100": ("H100!", "Nvidia H100 SXM5"),
    "h200": ("H200", "Nvidia H200 SXM"),
    "rtx-pro-6000": ("RTX-PRO-6000", "Nvidia RTX PRO 6000"),
    "a100-80gb": ("A100-80GB", "Nvidia A100 80GB"),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "datasets==5.0.1",
        "pytest==9.1.1",
    )
    .add_local_dir(ROOT, remote_path=REMOTE_ROOT, copy=True)
)
app = modal.App("law-slm-200m-gpu-benchmarks", image=image)


@app.function(timeout=3_600, cpu=8, memory=16_384)
def tests() -> None:
    subprocess.run(
        ["python", "-m", "pytest", "-q"],
        cwd=REMOTE_ROOT,
        check=True,
    )


@app.function(timeout=21_600, cpu=4, memory=16_384)
def benchmark(
    key: str,
    warmup_steps: int,
    timed_steps: int,
    sequence_lengths: str,
    micro_batch_sizes: str,
) -> dict[str, str]:
    _, price_name = GPU_CONFIGS[key]
    output = Path(f"/tmp/{key}.csv")
    command = [
        "python",
        "-m",
        "benchmarks.run_benchmark",
        "--gpu-name",
        price_name,
        "--output",
        str(output),
        "--warmup-steps",
        str(warmup_steps),
        "--timed-steps",
        str(timed_steps),
        "--sequence-lengths",
        sequence_lengths,
        "--micro-batch-sizes",
        micro_batch_sizes,
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    return {
        "csv": output.read_text(encoding="utf-8"),
        "markdown": output.with_suffix(".md").read_text(encoding="utf-8"),
    }


@app.function(timeout=21_600, cpu=4, memory=16_384)
def real_data(
    key: str,
    sequence_length: int,
    micro_batch_size: int,
    global_batch_tokens: str,
    steps: int,
) -> dict[str, str]:
    _, price_name = GPU_CONFIGS[key]
    output = Path(f"/tmp/{key}-tinystories.csv")
    command = [
        "python",
        "-m",
        "benchmarks.run_real_data",
        "--gpu-name",
        price_name,
        "--sequence-length",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-tokens",
        global_batch_tokens,
        "--steps",
        str(steps),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)
    return {
        "csv": output.read_text(encoding="utf-8"),
        "markdown": output.with_suffix(".md").read_text(encoding="utf-8"),
    }


def save_result(key: str, result: dict[str, str], suffix: str = "") -> None:
    results_dir = ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{key}{suffix}"
    (results_dir / f"{stem}.csv").write_text(
        result["csv"], encoding="utf-8"
    )
    (results_dir / f"{stem}.md").write_text(
        result["markdown"], encoding="utf-8"
    )


@app.local_entrypoint()
def main(
    mode: str = "benchmark",
    gpu: str = "all",
    warmup_steps: int = 2,
    timed_steps: int = 5,
    sequence_lengths: str = "2048,4096,8192",
    micro_batch_sizes: str = "2,4,8,16,32,64,128",
    sequence_length: int = 2048,
    micro_batch_size: int = 2,
    global_batch_tokens: str = "262144,524288,1048576",
    steps: int = 200,
    result_suffix: str = "",
) -> None:
    keys = list(GPU_CONFIGS) if gpu == "all" else [gpu]
    unknown = set(keys) - set(GPU_CONFIGS)
    if unknown:
        raise ValueError(
            f"unknown GPU key(s): {sorted(unknown)}; "
            f"choose from {sorted(GPU_CONFIGS)} or all"
        )

    if mode == "real-data":
        if len(keys) != 1:
            raise ValueError("real-data mode requires one --gpu")
        key = keys[0]
        selector, _ = GPU_CONFIGS[key]
        result = real_data.with_options(gpu=selector).remote(
            key,
            sequence_length,
            micro_batch_size,
            global_batch_tokens,
            steps,
        )
        save_result(key, result, suffix="-tinystories")
        print(f"Saved TinyStories results for {key}")
        return
    if mode == "test":
        tests.remote()
        return
    if mode != "benchmark":
        raise ValueError("--mode must be benchmark, real-data, or test")

    calls: dict[str, modal.functions.FunctionCall] = {}
    for key in keys:
        selector, _ = GPU_CONFIGS[key]
        calls[key] = benchmark.with_options(gpu=selector).spawn(
            key,
            warmup_steps,
            timed_steps,
            sequence_lengths,
            micro_batch_sizes,
        )
    for key, call in calls.items():
        save_result(key, call.get(), suffix=result_suffix)
        print(f"Saved benchmark results for {key}")
