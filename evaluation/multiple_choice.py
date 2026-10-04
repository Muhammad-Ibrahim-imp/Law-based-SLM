"""Zero-shot multiple-choice evaluation using conditional log likelihood."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset


BENCHMARKS = {
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "piqa": ("ybisk/piqa", None, "validation"),
}


def load_benchmark(name: str):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}")
    dataset_name, subset, split = BENCHMARKS[name]
    return load_dataset(dataset_name, subset, split=split)


def benchmark_example(name: str, row: dict) -> tuple[str, list[str], int]:
    if name == "hellaswag":
        prompt = f"{row['ctx']}\nWhat happens next?"
        return prompt, list(row["endings"]), int(row["label"])
    if name == "piqa":
        prompt = f"Goal: {row['goal']}\nBest solution:"
        return prompt, [row["sol1"], row["sol2"]], int(row["label"])
    if name == "arc_easy":
        labels = list(row["choices"]["label"])
        answer = labels.index(row["answerKey"])
        prompt = f"Question: {row['question']}\nAnswer:"
        return prompt, list(row["choices"]["text"]), answer
    raise ValueError(f"unknown benchmark {name!r}")


@torch.no_grad()
def evaluate_multiple_choice(
    model: torch.nn.Module,
    tokenizer,
    rows: Iterable[dict],
    benchmark_name: str,
    device: torch.device,
    max_examples: int | None = None,
) -> dict[str, float | int | str]:
    """Report raw and length-normalized zero-shot accuracy."""
    model.eval()
    correct_sum = 0
    correct_normalized = 0
    total = 0
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    for row in rows:
        if max_examples is not None and total >= max_examples:
            break
        prompt, choices, answer = benchmark_example(benchmark_name, row)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            prompt_ids = [tokenizer.bos_token_id] + prompt_ids
        encoded_choices = [
            tokenizer.encode(f" {choice}", add_special_tokens=False)
            for choice in choices
        ]
        full_sequences = [prompt_ids + choice for choice in encoded_choices]
        max_length = max(len(sequence) for sequence in full_sequences)
        if max_length > model.config.max_seq_len:
            raise ValueError(
                f"{benchmark_name} example length {max_length} exceeds "
                f"context {model.config.max_seq_len}"
            )
        tokens = torch.full(
            (len(choices), max_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for index, sequence in enumerate(full_sequences):
            tokens[index, : len(sequence)] = torch.tensor(sequence, device=device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type in {"cuda", "cpu"},
        ):
            logits = model(tokens[:, :-1])
        summed_scores = []
        normalized_scores = []
        start = len(prompt_ids) - 1
        for index, choice_ids in enumerate(encoded_choices):
            count = len(choice_ids)
            selected = logits[index, start : start + count].float()
            labels = tokens[index, start + 1 : start + count + 1]
            score = -F.cross_entropy(selected, labels, reduction="sum")
            summed_scores.append(float(score))
            normalized_scores.append(float(score) / count)

        correct_sum += int(max(range(len(choices)), key=summed_scores.__getitem__) == answer)
        correct_normalized += int(
            max(range(len(choices)), key=normalized_scores.__getitem__) == answer
        )
        total += 1

    if total == 0:
        raise ValueError("benchmark contained no examples")
    return {
        "benchmark": benchmark_name,
        "examples": total,
        "accuracy": correct_sum / total,
        "normalized_accuracy": correct_normalized / total,
    }
