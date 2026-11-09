"""Chat locally with the downloaded Family or Criminal SFT model."""
from __future__ import annotations

import argparse
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
from tokenizer.chat import ChatEncoder


def reply(model, encoder, messages, device, max_new_tokens):
    ids, _ = encoder.encode(messages, add_generation_prompt=True)
    if len(ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("Conversation is too long. Type /reset or use a shorter question.")
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    output = generate(model, prompt,
                      GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0),
                      eos_id=encoder.im_end_id)
    tokens = output[0, len(ids):].tolist()
    if encoder.im_end_id in tokens:
        tokens = tokens[:tokens.index(encoder.im_end_id)]
    return encoder.tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("family", "criminal"))
    parser.add_argument("--prompt", help="Answer one question and exit")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 < args.max_new_tokens < 8192:
        parser.error("--max-new-tokens must be between 1 and 8191")
    device = torch.device(args.device)
    print(f"Loading {args.domain} model on {device}...", flush=True)
    model = load_trained_model(ROOT / "configs/sft" / f"gemma3_200m_{args.domain}_sft.json",
                               ROOT / "checkpoints" / f"{args.domain}_sft.pt", device)
    encoder = ChatEncoder(load_smollm2_tokenizer())
    messages = []
    print("Ready. Type a question. /reset clears history; /quit exits.", flush=True)
    while True:
        try:
            question = args.prompt if args.prompt is not None else input("You> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.strip().lower() in ("/quit", "/exit"):
            return
        if question.strip().lower() == "/reset":
            messages.clear()
            print("Conversation cleared.", flush=True)
        elif question.strip():
            messages.append({"role": "user", "content": question})
            try:
                answer = reply(model, encoder, messages, device, args.max_new_tokens)
            except ValueError as error:
                messages.pop()
                print(str(error), flush=True)
            else:
                messages.append({"role": "assistant", "content": answer})
                print("Model> " + (answer or "(empty response)"), flush=True)
        if args.prompt is not None:
            return


if __name__ == "__main__":
    main()
