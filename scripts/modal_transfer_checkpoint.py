"""Copy a finished general checkpoint between authorized Modal workspaces on CPU."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import modal

app = modal.App("law-slm-transfer-checkpoint")


@app.function(cpu=2, memory=4096, timeout=3600)
def transfer(name: str) -> dict:
    if name not in {"gemma3_200m", "lfm25_200m"}:
        raise ValueError("unsupported checkpoint")
    source_client = modal.Client.from_credentials(os.environ["TRANSFER_TOKEN_ID"],
                                                 os.environ["TRANSFER_TOKEN_SECRET"])
    source_volume = modal.Volume.from_name("law-slm-training", client=source_client,
                                           environment_name="main")
    staging = tempfile.TemporaryDirectory()
    source = Path(staging.name)
    files = ("summary.json", "metrics.jsonl", "checkpoint.pt")
    hashes = {}
    for filename in files:
        digest = hashlib.sha256()
        with (source / filename).open("wb") as handle:
            for chunk in source_volume.read_file(f"/training_sweep/{name}/{filename}"):
                handle.write(chunk)
                digest.update(chunk)
        hashes[filename] = digest.hexdigest()
        print(f"Read source {filename}", flush=True)
    summary = json.loads((source / "summary.json").read_text())
    if summary["optimizer_steps"] != 9490:
        raise ValueError("source run is not the completed step-9490 run")
    target = modal.Volume.from_name("law-slm-training", create_if_missing=True)
    remote_root = f"/training_sweep/{name}"
    with target.batch_upload() as batch:
        for filename in files:
            batch.put_file(source / filename, f"{remote_root}/{filename}")
    print(f"Uploaded {name}; verifying destination SHA-256", flush=True)
    for filename in files:
        digest = hashlib.sha256()
        for chunk in target.read_file(f"{remote_root}/{filename}"):
            digest.update(chunk)
        if digest.hexdigest() != hashes[filename]:
            raise ValueError(f"destination checksum mismatch: {filename}")
    result = {"name": name, "optimizer_steps": 9490, "sha256": hashes}
    staging.cleanup()
    print(json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(name: str, source_profile: str) -> None:
    from modal.config import config

    # Run on the destination so an exhausted source workspace needs no compute.
    secret = modal.Secret.from_dict({
        "TRANSFER_TOKEN_ID": config.get("token_id", profile=source_profile, use_env=False),
        "TRANSFER_TOKEN_SECRET": config.get("token_secret", profile=source_profile, use_env=False),
    })
    result = transfer.with_options(secrets=[secret]).remote(name)
    print(json.dumps(result, indent=2))
