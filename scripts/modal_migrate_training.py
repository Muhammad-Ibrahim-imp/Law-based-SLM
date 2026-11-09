"""Copy prepared legal/SFT data and resumable runs to a funded Modal workspace."""

import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

app = modal.App("law-slm-migrate-training")
RUN_ROOTS = (
    "/training_sweep/gemma3_200m", "/training_sweep/lfm25_200m",
    "/legal_finetuning/gemma3_200m_family_legal",
    "/legal_finetuning/gemma3_200m_criminal_legal",
    "/legal_finetuning/gemma3_200m_family_sft",
    "/legal_finetuning/gemma3_200m_criminal_sft",
)
DATA_ROOTS = (
    "/dataset/family_law_train", "/dataset/family_law_validation",
    "/dataset/criminal_law_train", "/dataset/criminal_law_validation", "/dataset/sft",
)


@app.function(cpu=4, memory=4096, timeout=7200)
def migrate(checkpoints_only: bool = False, sft_only: bool = False) -> dict:
    client = modal.Client.from_credentials(os.environ["SOURCE_TOKEN_ID"],
                                          os.environ["SOURCE_TOKEN_SECRET"])
    records = {}
    run_roots = tuple(root for root in RUN_ROOTS if root.endswith("_sft")) if sft_only else RUN_ROOTS
    data_roots = ("/dataset/sft",) if sft_only else DATA_ROOTS
    for volume_name, roots in (("law-slm-data", data_roots), ("law-slm-training", run_roots)):
        if checkpoints_only and volume_name == "law-slm-data":
            continue
        source = modal.Volume.from_name(volume_name, client=client, environment_name="main")
        target_name = "law-slm-sft-data" if sft_only and volume_name == "law-slm-data" else volume_name
        target = modal.Volume.from_name(target_name, create_if_missing=True)
        existing = {"/" + e.path.lstrip("/") for e in target.iterdir("/", recursive=True)
                    if e.type == modal.volume.FileEntryType.FILE}
        entries = []
        for root in roots:
            entries.extend(source.iterdir(root, recursive=True))
        if volume_name == "law-slm-data":
            entries.extend(e for e in source.iterdir("/dataset") if e.path.endswith("/manifest.json"))
        copied = {}
        def copy_file(entry):
            if entry.type != modal.volume.FileEntryType.FILE or entry.path.endswith((".tmp", ".lock")):
                return None
            remote = "/" + entry.path.lstrip("/")
            with tempfile.TemporaryDirectory() as directory:
                local = Path(directory, "payload")
                for attempt in range(3):
                    digest = hashlib.sha256()
                    size = 0
                    with local.open("wb") as handle:
                        for chunk in source.read_file(remote):
                            handle.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if size == entry.size:
                        break
                    print(f"Retry truncated download: {remote} ({size}/{entry.size})", flush=True)
                if size != entry.size:
                    raise ValueError(f"source size changed: {remote}")
                merge_manifest = volume_name == "law-slm-data" and remote == "/dataset/manifest.json" and remote in existing
                if merge_manifest:
                    incoming = json.loads(local.read_text())
                    if sft_only:
                        incoming["categories"] = {"sft": incoming["categories"]["sft"]}
                    previous = json.loads(b"".join(target.read_file(remote)))
                    if {k: v for k, v in incoming.items() if k != "categories"} != {k: v for k, v in previous.items() if k != "categories"}:
                        raise ValueError("dataset manifest metadata differs")
                    for category, value in incoming["categories"].items():
                        if category in previous["categories"] and previous["categories"][category] != value:
                            raise ValueError(f"existing dataset category differs: {category}")
                    incoming["categories"] = previous["categories"] | incoming["categories"]
                    local.write_text(json.dumps(incoming, indent=2) + "\n")
                    size = local.stat().st_size
                    digest = hashlib.sha256(local.read_bytes())
                    with target.batch_upload(force=True) as batch:
                        batch.put_file(local, remote)
                if remote not in existing:
                    with target.batch_upload() as batch:
                        batch.put_file(local, remote)
                destination_hash = hashlib.sha256()
                for chunk in target.read_file(remote):
                    destination_hash.update(chunk)
                if destination_hash.hexdigest() != digest.hexdigest():
                    if remote in existing:
                        raise ValueError(f"existing destination differs; refusing overwrite: {volume_name}:{remote}")
                    # Repair only a file created by this migration.
                    with target.batch_upload(force=True) as batch:
                        batch.put_file(local, remote)
                    destination_hash = hashlib.sha256()
                    for chunk in target.read_file(remote):
                        destination_hash.update(chunk)
                    if destination_hash.hexdigest() != digest.hexdigest():
                        raise ValueError(f"destination checksum mismatch: {volume_name}:{remote}")
                print(f"Verified {volume_name}:{remote} ({size:,} bytes)", flush=True)
                return remote, {"bytes": size, "sha256": digest.hexdigest()}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(copy_file, entries):
                if result is not None:
                    copied[result[0]] = result[1]
        required = ["/dataset/manifest.json"] if volume_name == "law-slm-data" else [
            root + "/checkpoint.pt" for root in run_roots
        ]
        if not all(path in copied for path in required):
            raise ValueError(f"migration missed required files in {volume_name}")
        records[target_name] = copied
    result = {"files": sum(len(v) for v in records.values()),
              "bytes": sum(f["bytes"] for v in records.values() for f in v.values())}
    with tempfile.TemporaryDirectory() as directory:
        local = Path(directory, "migration_sha256.json")
        local.write_text(json.dumps(records, indent=2) + "\n")
        with modal.Volume.from_name("law-slm-training").batch_upload() as batch:
            batch.put_file(local, "/migration_sha256.json")
    print("Migration verified: " + json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main(source_profile: str = "abdulrehmanriaz299", checkpoints_only: bool = False, sft_only: bool = False) -> None:
    from modal.config import config

    secret = modal.Secret.from_dict({
        "SOURCE_TOKEN_ID": config.get("token_id", profile=source_profile, use_env=False),
        "SOURCE_TOKEN_SECRET": config.get("token_secret", profile=source_profile, use_env=False),
    })
    call = migrate.with_options(secrets=[secret]).spawn(checkpoints_only, sft_only)
    print(f"Migration call: {call.object_id}", flush=True)
    time.sleep(3)
    os._exit(0)
