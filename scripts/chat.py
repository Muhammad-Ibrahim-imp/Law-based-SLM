"""Continue text from a trained checkpoint.

The model predicts the next token. It was not trained as a chat assistant,
so a prompt is completed as more text. Run from the repository root:

    python scripts/chat.py --config configs/training/200m/gemma3_200m.json --checkpoint gemma3_200m_checkpoint.pt
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.interfaces import GenerationConfig
from generation.generate import generate
from generation.load import load_trained_model
from tokenizer import load_smollm2_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment JSON used to train the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pt from a finished run")
    parser.add_argument("--prompt", help="Continue this text once and exit")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"loading checkpoint on {device.type}", flush=True)
    model = load_trained_model(args.config, args.checkpoint, device)
    if hasattr(model, "build_cache"):
        print("each new token uses the cache instead of the whole conversation", flush=True)
    tokenizer = load_smollm2_tokenizer()
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    print(
        "This model continues text. It was not trained to answer questions.",
        flush=True,
    )

    if args.prompt is not None:
        context = _encode(tokenizer, args.prompt, add_special_tokens=True)
        context, _new_text = _continue(model, tokenizer, context, config, generator, device)
        print(tokenizer.decode(context, skip_special_tokens=True), flush=True)
        return

    print("Type a prompt and press Enter. Commands: /reset, /quit.", flush=True)
    context: list[int] = []
    first_turn = True
    while True:
        try:
            line = input("you> ")
        except EOFError:
            print(flush=True)
            return
        command = line.strip().lower()
        if command in {"/quit", "/exit", "/q"}:
            return
        if command == "/reset":
            context = []
            first_turn = True
            print("context cleared", flush=True)
            continue
        if not line.strip():
            continue
        if context:
            line = "\n" + line
        context.extend(_encode(tokenizer, line, add_special_tokens=first_turn))
        first_turn = False
        context, new_text = _continue(model, tokenizer, context, config, generator, device)
        if not new_text:
            new_text = "(no further text)"
        print("model> " + new_text, flush=True)


def _encode(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def _continue(model, tokenizer, context, config, generator, device) -> tuple[list[int], str]:
    room = model.config.max_seq_len - config.max_new_tokens
    if room < 1:
        raise ValueError("max_new_tokens leaves no room for a prompt")
    if len(context) > room:
        context = context[-room:]
        print("earlier context was dropped to stay inside the context limit", flush=True)
    if not context:
        raise ValueError("prompt is empty")
    prompt = torch.tensor([context], dtype=torch.long, device=device)
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with autocast:
        output = generate(
            model,
            prompt,
            config,
            eos_id=tokenizer.eos_token_id,
            generator=generator,
        )
    new_tokens = output[0, prompt.shape[1] :].tolist()
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in new_tokens:
        new_tokens = new_tokens[: new_tokens.index(eos_id)]
    return context + new_tokens, tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    main()
