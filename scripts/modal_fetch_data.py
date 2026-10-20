"""Fetch the legal pretraining data from our private Hugging Face repo onto a Modal volume.

    modal run --detach scripts/modal_fetch_data.py
    modal run --detach scripts/modal_fetch_data.py --categories family_law_train,family_law_validation
    modal run --detach scripts/modal_fetch_data.py --overwrite

Run it with the Modal profile of the account that trains (`modal profile activate <name>`).
It needs a Modal Secret named ``huggingface`` holding a token that can read
Law-based-slms/legal-pre-training-data (a read token is enough):

    modal secret create huggingface HF_TOKEN=hf_...

Steps (function ``fetch``, on the volume ``law-slm-data`` mounted at /vol/data):

1. snapshot_download the chosen categories + manifest.json into /vol/data/_incoming/
2. check every shard's size (tokens x 2 bytes) and SHA-256 against the downloaded manifest
3. refuse if a category already exists in /vol/data/dataset/manifest.json (unless --overwrite)
4. move the category folders into /vol/data/dataset/
5. merge the categories into the existing manifest (scripts/merge_manifest_section.py;
   refuses on a tokenizer_fingerprint mismatch)
6. general_prep.validate_shards on each category
7. commit the volume (with retries), delete _incoming, commit again

Its own app and a slim image; it does not import scripts/modal_training.py. The local
entrypoint spawns the call and exits, so a dropped connection does not cancel it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/project"
DATA_ROOT = "/vol/data/dataset"            # matches modal_training.py
INCOMING = "/vol/data/_incoming"
HF_REPO = "Law-based-slms/legal-pre-training-data"
LEGAL_CATEGORIES = ("family_law_train", "family_law_validation", "criminal_law_train", "criminal_law_validation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "numpy")
    .add_local_dir(ROOT, remote_path=REMOTE_ROOT, copy=True,
                   ignore=["dataset/**", "results/**", ".git/**", "**/*.pt", "**/*.bin", "**/*.pyc",
                           "**/__pycache__/**", ".pytest_cache/**", "frontend/**"])
    .env({"PYTHONPATH": REMOTE_ROOT})
)
app = modal.App("law-slm-fetch-data", image=image)
data_volume = modal.Volume.from_name("law-slm-data", create_if_missing=True)


# ---------------------------------------------------------------- local logic (tested without Modal)
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_downloaded(incoming: Path, manifest: dict, categories) -> list[str]:
    """Problems with the downloaded shards (empty list = all present, right size and SHA-256)."""
    problems = []
    for cat in categories:
        entry = manifest.get("categories", {}).get(cat)
        if entry is None:
            problems.append(f"{cat}: not in the downloaded manifest")
            continue
        for s in entry["shards"]:
            p = incoming / cat / s["file"]
            if not p.exists():
                problems.append(f"{cat}/{s['file']}: missing")
            elif p.stat().st_size != 2 * s["tokens"]:
                problems.append(f"{cat}/{s['file']}: size {p.stat().st_size} != {2 * s['tokens']}")
            elif sha256_file(p) != s["checksum_sha256"]:
                problems.append(f"{cat}/{s['file']}: SHA-256 mismatch")
    return problems


def existing_categories(dataset_root: Path, categories) -> list[str]:
    path = dataset_root / "manifest.json"
    have = json.loads(path.read_text()).get("categories", {}) if path.exists() else {}
    return [c for c in categories if c in have or (dataset_root / c).exists()]


def install(incoming: Path, dataset_root: Path, categories, overwrite: bool = False) -> None:
    """Steps 2-6: verify, refuse overwrite, move folders, merge manifest, validate shards."""
    from data.general_prep import validate_shards
    from scripts.merge_manifest_section import merge_sections

    src_manifest = json.loads((incoming / "manifest.json").read_text())
    problems = verify_downloaded(incoming, src_manifest, categories)
    if problems:
        raise RuntimeError("downloaded files do not match the manifest:\n  " + "\n  ".join(problems))
    clash = existing_categories(dataset_root, categories)
    if clash and not overwrite:
        raise RuntimeError(f"{clash} already exist in {dataset_root}; rerun with --overwrite to replace them")
    dst_fp = (json.loads((dataset_root / "manifest.json").read_text()).get("tokenizer_fingerprint")
              if (dataset_root / "manifest.json").exists() else None)
    if dst_fp is not None and dst_fp != src_manifest.get("tokenizer_fingerprint"):   # check before moving files
        raise ValueError(f"tokenizer_fingerprint differs: target {dst_fp}, "
                         f"source {src_manifest.get('tokenizer_fingerprint')}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        if (dataset_root / cat).exists():
            shutil.rmtree(dataset_root / cat)
        shutil.move(str(incoming / cat), str(dataset_root / cat))
    merge_sections(src_manifest, dataset_root, list(categories))
    for cat in categories:
        validate_shards(str(dataset_root), cat)


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
@app.function(timeout=86_400, cpu=4, memory=8_192, volumes={"/vol/data": data_volume},
              secrets=[modal.Secret.from_name("huggingface")])
def fetch(categories: list[str], overwrite: bool = False) -> None:
    from huggingface_hub import snapshot_download

    data_volume.reload()
    incoming = Path(INCOMING)
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=incoming, token=os.environ.get("HF_TOKEN"),
                      allow_patterns=["manifest.json"] + [f"{c}/*" for c in categories])
    install(incoming, Path(DATA_ROOT), categories, overwrite)
    commit_volume()
    shutil.rmtree(incoming, ignore_errors=True)
    commit_volume()
    print(f"fetched and validated {categories} into {DATA_ROOT}", flush=True)


@app.local_entrypoint()
def main(categories: str = ",".join(LEGAL_CATEGORIES), overwrite: bool = False) -> None:
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    # Spawn and hard-exit, as in modal_training.py: a waiting call is cancelled if this machine
    # loses its connection, and a normal return stops the app.
    call = fetch.spawn(cats, overwrite)
    print(call.object_id, flush=True)
    time.sleep(3)
    os._exit(0)
