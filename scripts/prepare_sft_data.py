"""Thin CLI for SFT data prep. Logic lives in data/sft_prep.py and
data/sft_sources.py.

Sources: smol_smoltalk (conversational), casehold and lawinstruct (legal).
None of them needs datasets<4.

Recommended order (measure first, then build with caps):

    # 1. Measure each source (writes nothing): kept/trained tokens, drops, padding
    python -m scripts.prepare_sft_data --root dataset --source smol_smoltalk --measure
    python -m scripts.prepare_sft_data --root dataset --source casehold --measure
    python -m scripts.prepare_sft_data --root dataset --source lawinstruct --measure

    # 2. Build train + test for each (rerun the same command to resume after a crash)
    python -m scripts.prepare_sft_data --root dataset --source casehold --split train
    python -m scripts.prepare_sft_data --root dataset --source casehold --split test
    python -m scripts.prepare_sft_data --root dataset --source lawinstruct --split train \\
        --max-tokens <cap from step 1>
    ...

    # 3. Validate and eyeball one row
    python -m scripts.prepare_sft_data --root dataset --source casehold --split train --validate
"""

import argparse
import json

from data.sft_prep import SFTPrepConfig, build_sft_split, decode_row, validate_sft_split
from data.sft_sources import (
    LAWINSTRUCT_MAX_PER_SUBSET, LAWINSTRUCT_SUBSETS, SOURCES, load_source,
)
from tokenizer import get_tokenizer_fingerprint, load_smollm2_tokenizer

HF_REPOS = {
    "smol_smoltalk": "HuggingFaceTB/smol-smoltalk",
    "casehold": "casehold/casehold",
    "lawinstruct": "lawinstruct/lawinstruct",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root (contains manifest.json)")
    p.add_argument("--source", choices=sorted(SOURCES), default="smol_smoltalk")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sequence-length", type=int, default=8192,
                   help="packed row length; 8192 matches the 200M pre-training context")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="cap on kept (non-pad) tokens; default: whole split")
    p.add_argument("--lawinstruct-subsets", default=",".join(LAWINSTRUCT_SUBSETS),
                   help="comma-separated LawInstruct dataset-subset names")
    p.add_argument("--max-per-subset", type=int, default=LAWINSTRUCT_MAX_PER_SUBSET,
                   help="LawInstruct: max examples per subset (0 = no cap)")
    p.add_argument("--measure", action="store_true", help="count only, write nothing")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate", action="store_true", help="validate an existing build")
    a = p.parse_args()

    tok = load_smollm2_tokenizer()
    if a.validate:
        print(json.dumps(validate_sft_split(a.root, a.source, a.split), indent=2))
        print(decode_row(a.root, a.source, a.split, tok)[:3000])
        return

    options = {}
    if a.source == "lawinstruct":
        options = {"subsets": [s.strip() for s in a.lawinstruct_subsets.split(",") if s.strip()],
                   "max_per_subset": a.max_per_subset or None}

    cfg = SFTPrepConfig(dataset_root=a.root, source_name=a.source,
                        hf_dataset=HF_REPOS[a.source], split=a.split, seed=a.seed,
                        sequence_length=a.sequence_length, max_tokens=a.max_tokens,
                        loader_options=options or None)
    examples, info = load_source(a.source, a.split, seed=a.seed, **options)
    result = build_sft_split(cfg, examples, tok, get_tokenizer_fingerprint(),
                             overwrite=a.overwrite, measure_only=a.measure, source_info=info)
    if not a.measure:
        result = {k: v for k, v in result.items() if k != "shards"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
