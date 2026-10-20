"""Build the FAMILY / CRIMINAL law pretraining corpora. Logic: data/legal_prep.py.

    # pilot: 1 CourtListener + 1 HFforLegal file + the statute sources, into its own root
    python -m scripts.prepare_legal_data --pilot
    # full build into dataset/ (rerun to resume; finished files/stages are skipped)
    python -m scripts.prepare_legal_data --root dataset --work dataset/_work/legal
    # one stage
    python -m scripts.prepare_legal_data --root dataset --work dataset/_work/legal --stage validate

Stages: extract, evalsets, features, dedup, tokenize, write, validate (default: all, in order).
"""

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

from data import legal_prep as lp

STAGES = ("extract", "evalsets", "features", "dedup", "tokenize", "write", "validate")
CATEGORIES = [(d, s) for d in lp.DOMAINS for s in ("train", "validation")]


def run_pool(fn, args_list, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        results = [pool.apply_async(fn, a) for a in args_list]
        out = []
        for a, r in zip(args_list, results):
            out.append(r.get())
            lp.log(f"{fn.__name__} {a[0] if isinstance(a[0], str) else a[0]['job']}: done ({out[-1]})")
        return out


def stage_extract(jobs, paths, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        pending = []
        for job in jobs:
            if (paths.cand / f"{job['job']}.stats.json").exists():
                continue
            try:
                path = lp.ensure_downloaded(job)   # sequential downloads while workers extract
            except Exception as e:
                lp.log(f"extract {job['job']}: SKIPPED ({e})")
                continue
            pending.append((job, pool.apply_async(lp.extract_job, (job, path, str(paths.cand)))))
        for job, r in pending:
            st = r.get()
            lp.log(f"extract {job['job']}: {st['docs_read']} docs read, family kept {st['family']['kept']}, "
                   f"criminal kept {st['criminal']['kept']} ({st['seconds']}s; downloaded {lp.downloaded_gb():.1f} GB)")


def done_jobs(jobs, paths):
    return [j["job"] for j in jobs if (paths.cand / f"{j['job']}.stats.json").exists()]


def stage_validate(root, work):
    from data.general_prep import validate_shards
    from data.shard_dataset import ShardedTokenDataset
    from tokenizer import load_smollm2_tokenizer
    import torch
    tok = load_smollm2_tokenizer()
    out = {}
    for d, s in CATEGORIES:
        cat = f"{d}_law_{s}"
        validate_shards(root, cat)
        ds = ShardedTokenDataset(root, cat, sequence_length=lp.SEQUENCE_LENGTH)
        rng = random.Random(lp.SEED)
        windows = []
        for i in rng.sample(range(len(ds)), min(3, len(ds))):
            ex = ds[i]
            assert ex["input_ids"].shape == (lp.SEQUENCE_LENGTH,) and ex["labels"].shape == (lp.SEQUENCE_LENGTH,)
            assert ex["input_ids"].dtype == torch.long and ex["labels"].dtype == torch.long
            assert torch.equal(ex["labels"][:-1], ex["input_ids"][1:])
            windows.append(tok.decode(ex["input_ids"][:400].tolist()))
        out[cat] = {"windows_of_8192": len(ds), "decoded_samples_first_400_tokens": windows}
        lp.log(f"validate {cat}: OK, {len(ds)} windows")
    Path("results/legal_prep").mkdir(parents=True, exist_ok=True)
    (Path(work) / "validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="dataset root (default: dataset, or <work>/out for --pilot)")
    ap.add_argument("--work", default=None, help="intermediates (default dataset/_work/legal[_pilot])")
    ap.add_argument("--stage", choices=STAGES, action="append")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--decontam-min-run", type=int, default=lp.DECONTAM_MIN_RUN,
                    help="consecutive rare evaluation 13-grams that mark a document as contaminated")
    ap.add_argument("--decontam-max-df", type=int, default=lp.DECONTAM_MAX_DF,
                    help="evaluation 13-grams found in more candidate documents than this are ignored")
    a = ap.parse_args()
    work = a.work or ("dataset/_work/legal_pilot" if a.pilot else "dataset/_work/legal")
    root = a.root or (f"{work}/out" if a.pilot else "dataset")
    paths = lp.Paths(work)
    jobs = lp.all_jobs(a.pilot)
    for stage in a.stage or STAGES:
        lp.log(f"== {stage} (work={work}, root={root})")
        if stage == "extract":
            stage_extract(jobs, paths, a.workers)
        elif stage == "evalsets":
            lp.log(f"evalsets: {lp.build_evalsets(paths)}")
        elif stage == "features":
            todo = [(j, work) for j in done_jobs(jobs, paths) if not (paths.feat / f"{j}.npz").exists()]
            run_pool(lp.features_job, todo, a.workers)
        elif stage == "dedup":
            st = lp.dedup_and_split(paths, done_jobs(jobs, paths), a.decontam_min_run, a.decontam_max_df)
            lp.log(f"dedup: " + json.dumps({d: {k: st[d][k] for k in ("candidates", "exact_duplicates_removed",
                   "near_duplicates_removed", "decontaminated", "final")} for d in lp.DOMAINS}))
        elif stage == "tokenize":
            todo = [(j, work) for j in done_jobs(jobs, paths) if not (paths.tok / f"{j}.idx.npz").exists()]
            run_pool(lp.tokenize_job, todo, a.workers)
        elif stage == "write":
            names = done_jobs(jobs, paths)
            dedup = json.loads((paths.work / "dedup.json").read_text())
            extract = {j: json.loads((paths.cand / f"{j}.stats.json").read_text()) for j in names}
            for d, s in CATEGORIES:
                cap = lp.CRIMINAL_TRAIN_CAP if (d, s) == ("criminal", "train") else None
                marker = paths.work / f"write.{d}_law_{s}.json"
                if marker.exists():
                    continue
                w = lp.write_category(paths, names, root, d, s, cap)
                lp.register_category(root, w["category"], lp.manifest_entry(w, d, s, dedup, extract, cap))
                lp._write_json(marker, w)
                lp.log(f"write {w['category']}: {w['total_tokens']:,} tokens, {w['docs']:,} docs, "
                       f"left out by cap {w['left_out_by_cap']}")
        elif stage == "validate":
            stage_validate(root, work)


if __name__ == "__main__":
    main()
