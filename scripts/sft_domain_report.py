"""Domain shares of the legal SFT train data (family / criminal / both / other).

Tags each example's user turn with ``data.domains`` (short profile up to
4,000 chars, loose above) and counts examples, kept tokens and trained
(assistant) tokens exactly as ``data/sft_prep.py`` would encode them at
8,192 tokens (examples prep would drop are left out). Writes
results/sft_domains/domains.json and precision_samples.md (git-ignored).

    python -m scripts.sft_domain_report --stage samples   # precision samples only
    python -m scripts.sft_domain_report                   # full counts
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from data.domains import classify, profile_for, short_matches
from data.sft_prep import _prepare_example
from data.sft_sources import load_source
from tokenizer import load_smollm2_tokenizer
from tokenizer.chat import ChatEncoder

OUT = Path("results/sft_domains")
SOURCES = ("casehold", "lawinstruct", "reddit_domain")
SAMPLE_SUBSETS = ("RedditLegalQA-reddit_legal_qa", "StackExchangeQuestionsLegal-stack_exchange_questions_legal")


def user_text(ex):
    return "\n\n".join(m["content"] for m in ex["messages"] if m["role"] == "user")


def label_of(tags):
    fam, crim = "family" in tags, "criminal" in tags
    return "both" if fam and crim else "family" if fam else "criminal" if crim else "other"


def write_samples(examples, k=30, seed=7):
    pools = defaultdict(list)
    for ex in examples:
        if ex["source"] in SAMPLE_SUBSETS:
            text = user_text(ex)
            tags = classify(text, profile_for(text))
            for d in ("family", "criminal"):
                if d in tags:
                    pools[d].append((ex, text))
            if not tags & {"family", "criminal"}:
                pools["other"].append((ex, text))
    rng = random.Random(seed)
    lines = [f"# Short-profile precision samples ({k} per label, seed {seed})\n"]
    for label in ("family", "criminal", "other"):
        lines.append(f"\n## {label.upper()} ({len(pools[label])} in pool)\n")
        for i, (ex, text) in enumerate(rng.sample(pools[label], min(k, len(pools[label]))), 1):
            m = short_matches(text) if profile_for(text) == "short" else "loose profile"
            lines.append(f"### {label}-{i} [{ex['source'].split('-')[0]}] {m if label == 'other' else m.get(label) if isinstance(m, dict) else m}\n"
                         f"> {' '.join(text.split())[:600]}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "precision_samples.md").write_text("\n".join(lines), encoding="utf-8")
    return {label: len(v) for label, v in pools.items()}


def count(examples, encoder, seq_len=8192):
    out = defaultdict(lambda: {"examples": 0, "tokens": 0, "trained": 0})
    tax = {"examples": 0, "tokens": 0, "trained": 0}
    for ex in examples:
        ids, mask, why = _prepare_example(encoder, ex["messages"], seq_len)
        if why:
            continue
        text = user_text(ex)
        tags = classify(text, profile_for(text))
        n, t = len(ids), int(sum(mask))
        for key in (label_of(tags), f"subset:{ex['source']}:{label_of(tags)}"):
            out[key]["examples"] += 1
            out[key]["tokens"] += n
            out[key]["trained"] += t
        if "tax" in tags:
            tax["examples"] += 1
            tax["tokens"] += n
            tax["trained"] += t
    return {**dict(out), "tax_any": tax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("samples", "report"), default="report")
    ap.add_argument("--seed", type=int, default=8, help="sample seed (round 1 of the check used 7)")
    a = ap.parse_args()
    li, _ = load_source("lawinstruct", "train")
    if a.stage == "samples":
        print("sample pools:", write_samples(li, seed=a.seed))
        return
    encoder = ChatEncoder(load_smollm2_tokenizer())
    result = {"tagger": "data.domains: short profile <= 4,000 chars of user text, loose above",
              "sources": {}}
    for name in SOURCES:
        examples = li if name == "lawinstruct" else load_source(name, "train")[0]
        result["sources"][name] = count(examples, encoder)
        print(name, {k: v for k, v in result["sources"][name].items() if not k.startswith("subset:")})
    (OUT / "domains.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
