"""Run the CUDA training-throughput grid and save crash-safe results."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.model_200m import EXPECTED_PARAMETER_COUNT, build_model
from benchmarks.reporting import FIELDS, write_rankings

SEQUENCE_LENGTHS = (2048, 4096, 8192)
MICRO_BATCH_SIZES = (2, 4, 8, 16, 32, 64, 128)
GPU_PRICES = {
    "Nvidia B300": 0.001972,
    "Nvidia B200": 0.001736,
    "Nvidia H200 SXM": 0.001261,
    "Nvidia H100 SXM5": 0.001097,
    "Nvidia RTX PRO 6000": 0.000842,
    "Nvidia A100 80GB": 0.000694,
}
def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def cleanup_after_attempt(optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def train_micro_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def base_row(
    gpu_name: str,
    actual_gpu_name: str,
    seq_len: int,
    micro_batch: int,
    warmup_steps: int,
    timed_steps: int,
    largest_working: int | None,
) -> dict[str, object]:
    return {
        "gpu_name": gpu_name,
        "actual_gpu_name": actual_gpu_name,
        "attention_backend": "PyTorch SDPA FlashAttention",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "sequence_length": seq_len,
        "micro_batch_size": micro_batch,
        "tokens_per_micro_step": seq_len * micro_batch,
        "status": "",
        "largest_working_micro_batch": largest_working or "",
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "elapsed_seconds": "",
        "steps_per_second": "",
        "tokens_per_second": "",
        "peak_memory_gib": "",
        "price_per_second_usd": GPU_PRICES[gpu_name],
        "micro_steps_per_dollar": "",
        "tokens_per_dollar": "",
        "cost_per_1b_tokens_usd": "",
        "error": "",
    }


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA GPU does not support BF16")
    if args.gpu_name not in GPU_PRICES:
        raise ValueError(f"unknown GPU price label: {args.gpu_name}")

    output = Path(args.output)
    rows: list[dict[str, object]] = []
    if args.resume and output.exists():
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    completed = {
        (int(row["sequence_length"]), int(row["micro_batch_size"]))
        for row in rows
    }

    device = torch.device("cuda")
    actual_gpu_name = torch.cuda.get_device_name(device)
    print(
        f"PyTorch {torch.__version__}, CUDA runtime {torch.version.cuda}, "
        f"GPU {actual_gpu_name}",
        flush=True,
    )
    model = build_model().to(device=device, dtype=torch.bfloat16)
    print(f"Exact parameter count: {model.parameter_count:,}", flush=True)
    if model.parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("parameter count does not match architecture")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, fused=True
    )
    model.train()

    for seq_len in args.sequence_lengths:
        largest_working: int | None = None
        sequence_oom = False
        for micro_batch in args.micro_batch_sizes:
            key = (seq_len, micro_batch)
            if key in completed:
                continue
            row = base_row(
                args.gpu_name,
                actual_gpu_name,
                seq_len,
                micro_batch,
                args.warmup_steps,
                args.timed_steps,
                largest_working,
            )
            if sequence_oom:
                row["status"] = "oom_skipped"
                row["largest_working_micro_batch"] = largest_working or ""
                row["error"] = "skipped after first OOM at this sequence length"
                rows.append(row)
                write_rows(output, rows)
                write_rankings(output.with_suffix(".md"), rows)
                continue

            input_ids = labels = None
            try:
                tokens = torch.randint(
                    0,
                    model.config.vocab_size,
                    (micro_batch, seq_len + 1),
                    device=device,
                    dtype=torch.long,
                )
                input_ids = tokens[:, :-1].contiguous()
                labels = tokens[:, 1:].contiguous()
                del tokens

                for _ in range(args.warmup_steps):
                    train_micro_step(model, optimizer, input_ids, labels)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                started = time.perf_counter()
                for _ in range(args.timed_steps):
                    train_micro_step(model, optimizer, input_ids, labels)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started

                steps_per_second = args.timed_steps / elapsed
                tokens_per_step = seq_len * micro_batch
                tokens_per_second = tokens_per_step * steps_per_second
                price = GPU_PRICES[args.gpu_name]
                tokens_per_dollar = tokens_per_second / price

                largest_working = micro_batch
                row.update(
                    {
                        "status": "ok",
                        "largest_working_micro_batch": largest_working,
                        "elapsed_seconds": elapsed,
                        "steps_per_second": steps_per_second,
                        "tokens_per_second": tokens_per_second,
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated() / 1024**3
                        ),
                        "micro_steps_per_dollar": steps_per_second / price,
                        "tokens_per_dollar": tokens_per_dollar,
                        "cost_per_1b_tokens_usd": 1e9 / tokens_per_dollar,
                    }
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(error).lower()
                )
                if not is_oom:
                    raise
                sequence_oom = True
                row["status"] = "oom"
                row["largest_working_micro_batch"] = largest_working or ""
                row["error"] = "CUDA out of memory"
                print(
                    f"OOM: sequence={seq_len}, micro_batch={micro_batch}; "
                    f"largest successful={largest_working}",
                    flush=True,
                )
            finally:
                del input_ids, labels
                cleanup_after_attempt(optimizer)

            rows.append(row)
            write_rows(output, rows)
            write_rankings(output.with_suffix(".md"), rows)
            completed.add(key)

    successful = sum(row["status"] == "ok" for row in rows)
    if not successful:
        raise RuntimeError("all benchmark configurations failed")
    print(f"Wrote {len(rows)} rows to {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-name", required=True, choices=GPU_PRICES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--sequence-lengths",
        type=parse_int_list,
        default=SEQUENCE_LENGTHS,
    )
    parser.add_argument(
        "--micro-batch-sizes",
        type=parse_int_list,
        default=MICRO_BATCH_SIZES,
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.timed_steps < 1:
        parser.error("warmup and timed steps must be positive")
    if any(length > 8192 for length in args.sequence_lengths):
        parser.error("sequence lengths cannot exceed 8192")
    return args


if __name__ == "__main__":
    run(parse_args())
