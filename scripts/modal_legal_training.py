"""Two independent H200 jobs: domain pretraining, then the shared SFT mix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import modal

from scripts.modal_training import DATA_ROOT, REMOTE_ROOT, image, data_volume, run_volume, commit_run_volume

OUTPUT_ROOT = "/vol/runs/legal_finetuning"
PRESERVED_ROOT = "/vol/runs/preserved/200m_general"
DOMAINS = ("family", "criminal")
app = modal.App("law-slm-legal-finetuning", image=image)
sft_volume = modal.Volume.from_name("law-slm-sft-data", create_if_missing=True)


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@app.function(cpu=4, memory=16384, timeout=86400,
              volumes={"/vol/data": data_volume, "/vol/sft-data": sft_volume, "/vol/runs": run_volume})
def preflight(sft_only: bool = False) -> None:
    """Verify inputs and preserve both finished checkpoints before allocating GPUs."""
    import torch
    from data.general_prep import validate_shards
    from scripts.modal_build_sft import BUILD_ORDER, run_builds

    (sft_volume if sft_only else data_volume).reload()
    data_root = "/vol/sft-data/dataset" if sft_only else DATA_ROOT
    run_volume.reload()
    for name in (() if sft_only else ("gemma3_200m", "lfm25_200m")):
        source = Path("/vol/runs/training_sweep", name)
        summary = json.loads((source / "summary.json").read_text())
        if summary["optimizer_steps"] != 9490:
            raise ValueError(f"{name}: expected completed general step 9490; got {summary['optimizer_steps']}")
        checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["global_step"] != 9490:
            raise ValueError(f"{name}: checkpoint does not match final summary")
        del checkpoint
        destination = Path(PRESERVED_ROOT, name)
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for filename in ("checkpoint.pt", "summary.json", "metrics.jsonl"):
            original, saved = source / filename, destination / filename
            digest = file_hash(original)
            if saved.exists():
                if file_hash(saved) != digest:
                    raise ValueError(f"preserved file differs: {saved}")
            else:
                temporary = saved.with_suffix(saved.suffix + ".tmp")
                shutil.copyfile(original, temporary)
                if file_hash(temporary) != digest:
                    raise ValueError(f"copy checksum failed: {temporary}")
                temporary.replace(saved)
            hashes[filename] = digest
        (destination / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    commit_run_volume()
    for domain in (() if sft_only else DOMAINS):
        for split in ("train", "validation"):
            validate_shards(DATA_ROOT, f"{domain}_law_{split}")
    expected = json.loads(Path(REMOTE_ROOT, "reports/sft_expected.json").read_text())
    if run_builds(data_root, BUILD_ORDER, expected):
        raise ValueError("SFT data differs from the reference; training refused")
    if sft_only:
        for domain in DOMAINS:
            checkpoint = torch.load(Path(OUTPUT_ROOT, f"gemma3_200m_{domain}_sft", "checkpoint.pt"), map_location="cpu", weights_only=False)
            if not 0 < checkpoint["global_step"] <= 727:
                raise ValueError(f"{domain}: invalid saved SFT step")
            print(f"Verified {domain} SFT checkpoint step {checkpoint['global_step']}", flush=True)
            del checkpoint
    print("Preflight passed; SFT checkpoints and data verified" if sft_only else "Preflight passed; both general checkpoints preserved and data verified", flush=True)


@app.function(gpu="H200", cpu=8, memory=32768, timeout=86400,
              max_containers=2,
              volumes={"/vol/data": data_volume, "/vol/sft-data": sft_volume, "/vol/runs": run_volume})
def train_domain(domain: str, sft_only: bool = False) -> list[dict]:
    import torch
    from experiments.config import load_experiment_config
    from experiments.runner import run_training

    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain}")
    (sft_volume if sft_only else data_volume).reload()
    data_root = "/vol/sft-data/dataset" if sft_only else DATA_ROOT
    run_volume.reload()
    results = []
    print(f"{domain}: {torch.cuda.get_device_name(0)}", flush=True)
    for stage in (("sft",) if sft_only else ("legal", "sft")):
        if sft_only and not Path(OUTPUT_ROOT, f"gemma3_200m_{domain}_sft", "checkpoint.pt").is_file():
            raise FileNotFoundError(f"missing SFT resume checkpoint: {domain}")
        config = load_experiment_config(
            Path(REMOTE_ROOT, "configs", stage, f"gemma3_200m_{domain}_{stage}.json")
        )
        result = run_training(config, data_root, OUTPUT_ROOT, torch.device("cuda"),
                              resume=True, compile_model=True,
                              checkpoint_callback=commit_run_volume)
        commit_run_volume()
        results.append(result)
    return results


@app.local_entrypoint()
def main() -> None:
    preflight.remote()
    calls = [train_domain.spawn(domain) for domain in DOMAINS]
    print("Training calls: " + " ".join(call.object_id for call in calls), flush=True)
    time.sleep(3)
    os._exit(0)
