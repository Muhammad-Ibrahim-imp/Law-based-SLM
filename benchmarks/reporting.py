"""Dependency-free CSV schema and Markdown ranking output."""

from __future__ import annotations

from pathlib import Path

FIELDS = (
    "gpu_name",
    "actual_gpu_name",
    "attention_backend",
    "parameter_count",
    "sequence_length",
    "micro_batch_size",
    "tokens_per_micro_step",
    "status",
    "largest_working_micro_batch",
    "warmup_steps",
    "timed_steps",
    "elapsed_seconds",
    "steps_per_second",
    "tokens_per_second",
    "peak_memory_gib",
    "price_per_second_usd",
    "micro_steps_per_dollar",
    "tokens_per_dollar",
    "cost_per_1b_tokens_usd",
    "error",
)


def write_rankings(path: Path, rows: list[dict[str, object]]) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    successful.sort(
        key=lambda row: float(row["micro_steps_per_dollar"]), reverse=True
    )
    lines = [
        "# Benchmark rankings",
        "",
        "“Steps per dollar” means optimizer **micro-steps** per dollar. "
        "Each row states its tokens per micro-step.",
        "",
        "## By micro-steps per dollar",
        "",
        "| Rank | GPU | Seq | Micro-batch | Tokens/micro-step | "
        "Micro-steps/$ | Tokens/$ | $/1B tokens | Peak GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(successful, 1):
        lines.append(
            f"| {rank} | {row['gpu_name']} | {row['sequence_length']} | "
            f"{row['micro_batch_size']} | {row['tokens_per_micro_step']} | "
            f"{float(row['micro_steps_per_dollar']):.3f} | "
            f"{float(row['tokens_per_dollar']):.0f} | "
            f"{float(row['cost_per_1b_tokens_usd']):.4f} | "
            f"{float(row['peak_memory_gib']):.2f} |"
        )

    cheapest = sorted(
        successful, key=lambda row: float(row["cost_per_1b_tokens_usd"])
    )
    lines.extend(
        [
            "",
            "## By cost per 1B tokens",
            "",
            "| Rank | GPU | Seq | Micro-batch | Tokens/s | $/1B tokens |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(cheapest, 1):
        lines.append(
            f"| {rank} | {row['gpu_name']} | {row['sequence_length']} | "
            f"{row['micro_batch_size']} | "
            f"{float(row['tokens_per_second']):.1f} | "
            f"{float(row['cost_per_1b_tokens_usd']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
