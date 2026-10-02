"""Short TinyStories runs for global-token-batch stability comparison."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks.model_200m import build_model
from benchmarks.run_benchmark import GPU_PRICES, parse_int_list

TOKENIZER_ID = "HuggingFaceTB/SmolLM2-135M"
DATASET_ID = "roneneldan/TinyStories"
FIELDS = (
    "gpu_name",
    "actual_gpu_name",
    "sequence_length",
    "micro_batch_size",
    "global_batch_tokens",
    "gradient_accumulation_steps",
    "optimizer_step",
    "loss",
    "elapsed_seconds",
    "tokens_seen",
    "tokens_per_second",
    "cost_usd",
)


def token_blocks(tokenizer: object, sequence_length: int) -> Iterator[torch.Tensor]:
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("SmolLM2 tokenizer has no EOS token")
    buffer: list[int] = []
    for example in dataset:
        buffer.extend(
            tokenizer.encode(example["text"], add_special_tokens=False)
        )
        buffer.append(eos_id)
        while len(buffer) >= sequence_length + 1:
            yield torch.tensor(
                buffer[: sequence_length + 1], dtype=torch.long
            )
            del buffer[: sequence_length + 1]


def next_batch(
    blocks: Iterator[torch.Tensor], micro_batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.stack([next(blocks) for _ in range(micro_batch_size)])
    return (
        packed[:, :-1].contiguous().to(device, non_blocking=True),
        packed[:, 1:].contiguous().to(device, non_blocking=True),
    )


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def summarize(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# TinyStories global-batch comparison",
        "",
        f"Tokenizer: `{TOKENIZER_ID}` (49,152 tokens). Dataset: `{DATASET_ID}`.",
        "",
        "| Global tokens | Accumulation | Initial loss | Final loss | "
        "Final-20 mean | Final-20 std | Tokens/s | Stable |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    candidates = sorted({int(row["global_batch_tokens"]) for row in rows})
    for candidate in candidates:
        selected = [
            row for row in rows if int(row["global_batch_tokens"]) == candidate
        ]
        losses = [float(row["loss"]) for row in selected]
        if not losses:
            continue
        tail = losses[-min(20, len(losses)) :]
        stable = all(math.isfinite(loss) for loss in losses)
        lines.append(
            f"| {candidate:,} | "
            f"{selected[0]['gradient_accumulation_steps']} | "
            f"{losses[0]:.4f} | {losses[-1]:.4f} | "
            f"{statistics.mean(tail):.4f} | "
            f"{statistics.pstdev(tail):.4f} | "
            f"{float(selected[-1]['tokens_per_second']):,.1f} | "
            f"{'yes' if stable else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_candidate(
    args: argparse.Namespace,
    global_batch_tokens: int,
    tokenizer: object,
    all_rows: list[dict[str, object]],
) -> None:
    device = torch.device("cuda")
    micro_tokens = args.sequence_length * args.micro_batch_size
    if global_batch_tokens % micro_tokens:
        raise ValueError(
            f"global batch {global_batch_tokens} is not divisible by "
            f"micro batch tokens {micro_tokens}"
        )
    accumulation_steps = global_batch_tokens // micro_tokens
    blocks = token_blocks(tokenizer, args.sequence_length)
    model = build_model().to(device=device, dtype=torch.bfloat16).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, fused=True
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    tokens_seen = 0

    for optimizer_step in range(1, args.steps + 1):
        loss_total = 0.0
        for _ in range(accumulation_steps):
            input_ids, labels = next_batch(
                blocks, args.micro_batch_size, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                )
                scaled_loss = loss / accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at optimizer step {optimizer_step}"
                )
            scaled_loss.backward()
            loss_total += float(loss.detach())
            tokens_seen += micro_tokens
            del input_ids, labels, logits, loss, scaled_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        average_loss = loss_total / accumulation_steps
        row = {
            "gpu_name": args.gpu_name,
            "actual_gpu_name": torch.cuda.get_device_name(device),
            "sequence_length": args.sequence_length,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_tokens": global_batch_tokens,
            "gradient_accumulation_steps": accumulation_steps,
            "optimizer_step": optimizer_step,
            "loss": average_loss,
            "elapsed_seconds": elapsed,
            "tokens_seen": tokens_seen,
            "tokens_per_second": tokens_seen / elapsed,
            "cost_usd": elapsed * GPU_PRICES[args.gpu_name],
        }
        all_rows.append(row)
        save_rows(Path(args.output), all_rows)
        summarize(Path(args.output).with_suffix(".md"), all_rows)
        print(
            f"global_tokens={global_batch_tokens} "
            f"step={optimizer_step}/{args.steps} loss={average_loss:.4f}",
            flush=True,
        )

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-name", required=True, choices=GPU_PRICES)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--global-batch-tokens",
        type=parse_int_list,
        default=(262_144, 524_288, 1_048_576),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 200 <= args.steps <= 500:
        parser.error("--steps must be between 200 and 500")
    if args.sequence_length > 8192:
        parser.error("--sequence-length cannot exceed 8192")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if len(tokenizer) != 49_152:
        raise RuntimeError(
            f"SmolLM2 tokenizer size is {len(tokenizer):,}, expected 49,152"
        )
    print(f"Using tokenizer {TOKENIZER_ID} with {len(tokenizer):,} tokens")
    rows: list[dict[str, object]] = []
    for candidate in args.global_batch_tokens:
        train_candidate(args, candidate, tokenizer, rows)


if __name__ == "__main__":
    main()
