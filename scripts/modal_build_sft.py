"""Build the SFT data on Modal, exactly as the reference build (reports/sft_expected.json).

    modal run --detach scripts/modal_build_sft.py

Run it with the Modal profile of the account that will run SFT. It writes
/vol/data/dataset/sft/ (volume ``law-slm-data``) and adds categories.sft to the
existing /vol/data/dataset/manifest.json (atomic, locked; refuses another tokenizer).

* Every split is rebuilt from the config recorded in reports/sft_expected.json
  (same seed, sequence length, max_tokens, loader options), reading the upstream
  datasets at the revisions pinned in data/sft_sources.REVISIONS.
* Resumable: completed splits are skipped, an interrupted split resumes from its
  last shard (data/sft_prep.py); the volume is committed after each split.
* At the end every split is validated (validate_sft_split) and compared with the
  reference: examples kept, kept / trained tokens, rows and every shard file's
  SHA-256. All equal -> "SFT build matches the reference build"; otherwise the
  differences are printed and nothing is deleted.
* The HF token is optional (all sources are public); the ``huggingface`` Modal
  Secret is used if it exists, to avoid rate limits.

Locally:
    python -m scripts.modal_build_sft --write-expected     # refresh reports/sft_expected.json
    python -m scripts.modal_build_sft --local-check ROOT casehold:test lawinstruct:test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"
DATA_ROOT = "/vol/data/dataset"
EXPECTED = ROOT / "reports" / "sft_expected.json"
REMOTE_EXPECTED = f"{REMOTE_ROOT}/reports/sft_expected.json"
# Same order as reports/sft_data.md; smol_smoltalk train's max_tokens is recorded in the expected file.
BUILD_ORDER = [("casehold", "train"), ("casehold", "test"), ("lawinstruct", "train"), ("lawinstruct", "test"),
               ("reddit_domain", "train"), ("reddit_domain", "test"), ("smol_smoltalk", "train"),
               ("smol_smoltalk", "test")]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("transformers==5.17.0", "datasets==5.0.1", "huggingface_hub", "numpy", "pyarrow")
    .add_local_dir(ROOT, remote_path=REMOTE_ROOT, copy=True,
                   ignore=["dataset/**", "results/**", ".git/**", "**/*.pt", "**/*.bin", "**/*.pyc",
                           "**/__pycache__/**", ".pytest_cache/**", "frontend/**"])
    .env({"PYTHONPATH": REMOTE_ROOT})
)
app = modal.App("law-slm-build-sft", image=image)
data_volume = modal.Volume.from_name("law-slm-data", create_if_missing=True)


# ---------------------------------------------------------------- reference file + comparison
def expected_from_manifest(manifest: dict) -> dict:
    """The reference record (configs, counts, shard hashes) from a manifest's categories.sft."""
    sources = manifest["categories"]["sft"]["sources"]
    out = {"tokenizer_fingerprint": manifest.get("tokenizer_fingerprint"), "splits": {}}
    for source, split in BUILD_ORDER:
        e = sources[source][split]
        st = e["stats"]
        out["splits"][f"{source}/{split}"] = {
            "config": e["config"],
            "examples_kept": st["examples_kept"], "kept_tokens": st["kept_tokens"],
            "trained_tokens": st["trained_tokens"], "rows": st["rows"], "total_tokens": e["total_tokens"],
            "shards": [{"files": s["files"], "num_tokens": s["num_tokens"], "sha256": s["sha256"]}
                       for s in e["shards"]],
        }
    return out


def compare_with_expected(manifest: dict, expected: dict, splits=None) -> list[str]:
    """Differences between a built manifest and the reference (empty list = match)."""
    diffs = []
    if manifest.get("tokenizer_fingerprint") != expected.get("tokenizer_fingerprint"):
        diffs.append(f"tokenizer_fingerprint: built {manifest.get('tokenizer_fingerprint')}, "
                     f"expected {expected.get('tokenizer_fingerprint')}")
    sources = manifest.get("categories", {}).get("sft", {}).get("sources", {})
    for key in splits or expected["splits"]:
        exp = expected["splits"][key]
        source, split = key.split("/")
        e = sources.get(source, {}).get(split)
        if e is None or e.get("status") != "complete":
            diffs.append(f"{key}: not built (status {None if e is None else e.get('status')})")
            continue
        st = e["stats"]
        got = {"config": e["config"], "examples_kept": st["examples_kept"], "kept_tokens": st["kept_tokens"],
               "trained_tokens": st["trained_tokens"], "rows": st["rows"], "total_tokens": e["total_tokens"]}
        for field, value in got.items():
            if value != exp[field]:
                diffs.append(f"{key}: {field} built {value!r}, expected {exp[field]!r}")
        if len(e["shards"]) != len(exp["shards"]):
            diffs.append(f"{key}: {len(e['shards'])} shards built, expected {len(exp['shards'])}")
        for i, (s, x) in enumerate(zip(e["shards"], exp["shards"])):
            if s["num_tokens"] != x["num_tokens"]:
                diffs.append(f"{key}: shard {i} num_tokens {s['num_tokens']}, expected {x['num_tokens']}")
            for stream, h in x["sha256"].items():
                if s["sha256"].get(stream) != h:
                    diffs.append(f"{key}: shard {i} {x['files'][stream]} sha256 {s['sha256'].get(stream)}, "
                                 f"expected {h}")
    return diffs


# ---------------------------------------------------------------- build core (Modal and local)
def build_split(root: str, source: str, split: str, expected: dict, tok, fingerprint: str) -> str:
    """Build one split from its recorded config; returns 'skipped' or 'built'."""
    from data.sft_prep import SFTPrepConfig, build_sft_split
    from data.shard_writer import read_manifest
    from data.sft_sources import load_source

    entry = (read_manifest(Path(root)).get("categories", {}).get("sft", {}).get("sources", {})
             .get(source, {}).get(split))
    if entry and entry.get("status") == "complete":
        return "skipped"
    cfg_dict = dict(expected["splits"][f"{source}/{split}"]["config"])
    cfg = SFTPrepConfig(dataset_root=root, **cfg_dict)
    options = cfg.loader_options or {}
    examples, info = load_source(source, split, seed=cfg.seed, **options)
    build_sft_split(cfg, examples, tok, fingerprint, source_info=info)
    return "built"


def run_builds(root: str, splits, expected: dict, after_split=None) -> list[str]:
    """Build the given splits, validate them and compare with the reference; returns the differences."""
    from data.shard_writer import read_manifest
    from data.sft_prep import validate_sft_split
    from tokenizer import get_tokenizer_fingerprint, load_smollm2_tokenizer

    tok = load_smollm2_tokenizer()
    fingerprint = get_tokenizer_fingerprint()
    for source, split in splits:
        t0 = time.time()
        status = build_split(root, source, split, expected, tok, fingerprint)
        print(f"{source}/{split}: {status} ({time.time() - t0:.0f}s)", flush=True)
        if after_split and status == "built":
            after_split()
    diffs = []
    for source, split in splits:
        try:
            validate_sft_split(root, source, split)
        except ValueError as e:
            diffs.append(f"{source}/{split}: validation failed: {e}")
    diffs += compare_with_expected(read_manifest(Path(root)), expected, [f"{s}/{p}" for s, p in splits])
    if diffs:
        print("SFT build DIFFERS from the reference build (nothing was deleted):", flush=True)
        for d in diffs:
            print("  " + d, flush=True)
    else:
        print("SFT build matches the reference build", flush=True)
    return diffs


def commit_volume() -> None:
    for attempt in range(6):   # same retry as modal_training.py
        try:
            data_volume.commit()
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))


# ---------------------------------------------------------------- Modal
# CPU only; 16 GB is ample (the reference build ran on a 16 GB PC with ~3-5 GB free).
@app.function(timeout=86_400, cpu=4, memory=16_384, nonpreemptible=True, volumes={"/vol/data": data_volume})
def build() -> list[str]:
    data_volume.reload()
    expected = json.loads(Path(REMOTE_EXPECTED).read_text())
    diffs = run_builds(DATA_ROOT, BUILD_ORDER, expected, after_split=commit_volume)
    commit_volume()
    return diffs


def _hf_secret():
    """The ``huggingface`` Modal Secret if this account has one, else None."""
    try:
        secret = modal.Secret.from_name("huggingface")
        secret.hydrate()
        return secret
    except Exception:
        return None


@app.local_entrypoint()
def main() -> None:
    # The token only avoids Hub rate limits (all sources are public): attach the secret if it exists.
    secret = _hf_secret()
    fn = build.with_options(secrets=[secret]) if secret else build
    print("using the 'huggingface' secret" if secret else "no 'huggingface' secret: anonymous Hub access", flush=True)
    # Spawn and hard-exit, as in modal_training.py, so a dropped connection doesn't cancel the build.
    call = fn.spawn()
    print(call.object_id, flush=True)
    time.sleep(3)
    os._exit(0)


# ---------------------------------------------------------------- local CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-expected", action="store_true", help="write reports/sft_expected.json from dataset/")
    ap.add_argument("--local-check", nargs="+", metavar=("ROOT", "SOURCE:SPLIT"),
                    help="build the given splits into ROOT (outside dataset/) and compare")
    a = ap.parse_args()
    if a.write_expected:
        from data.shard_writer import read_manifest
        EXPECTED.write_text(json.dumps(expected_from_manifest(read_manifest(ROOT / "dataset")), indent=1) + "\n")
        print(f"wrote {EXPECTED}")
    elif a.local_check:
        root, *pairs = a.local_check
        exp = json.loads(EXPECTED.read_text())
        raise SystemExit(1 if run_builds(root, [tuple(p.split(":")) for p in pairs], exp) else 0)
