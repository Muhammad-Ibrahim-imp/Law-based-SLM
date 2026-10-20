"""Merge ``categories`` sections from one manifest into another.

Use it before uploading shards to a Modal volume whose manifest already holds
other categories (e.g. ``general_train``): uploading the local manifest.json
as-is would overwrite those. Steps: reports/sft_data.md (SFT) and
reports/legal_pretraining_data.md (legal pretraining).

    python -m scripts.merge_manifest_section --from dataset/manifest.json --into-root remote_manifest
    python -m scripts.merge_manifest_section --from dataset/manifest.json --into-root remote_manifest \\
        --section family_law_train --section family_law_validation \\
        --section criminal_law_train --section criminal_law_validation

``--into-root`` is the directory holding the target ``manifest.json``; the
write goes through ``data.shard_writer.update_manifest`` (locked, atomic) and
happens only if every section can be merged. Each section in the target is
replaced as a whole; others are untouched. Refuses when the two manifests were
built with different tokenizers.
"""

import argparse
import json
from pathlib import Path

from data.shard_writer import MANIFEST_NAME, read_manifest, update_manifest

HEADER_KEYS = ("manifest_version", "dtype", "tokenizer_checkpoint", "tokenizer_fingerprint")


def merge_sections(src_manifest: dict, dst_root: Path, sections: list[str]) -> dict:
    missing = [s for s in sections if s not in src_manifest.get("categories", {})]
    if missing:
        raise ValueError(f"source manifest has no categories.{', categories.'.join(missing)}")
    src_fp = src_manifest.get("tokenizer_fingerprint")
    if not src_fp:
        raise ValueError("source manifest has no tokenizer_fingerprint")

    def fn(dst: dict) -> None:
        dst_fp = dst.get("tokenizer_fingerprint")
        if dst_fp is not None and dst_fp != src_fp:
            raise ValueError(f"tokenizer_fingerprint differs: target {dst_fp}, source {src_fp}; "
                             "the shards are not compatible")
        for k in HEADER_KEYS:  # an empty target gets the source header
            dst.setdefault(k, src_manifest.get(k))
        for s in sections:
            dst.setdefault("categories", {})[s] = src_manifest["categories"][s]

    return update_manifest(Path(dst_root), fn)


def merge_section(src_manifest: dict, dst_root: Path, section: str = "sft") -> dict:
    return merge_sections(src_manifest, dst_root, [section])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="source manifest.json")
    ap.add_argument("--into-root", required=True, help=f"directory holding the target {MANIFEST_NAME}")
    ap.add_argument("--section", action="append", help="category to merge (repeatable; default: sft)")
    a = ap.parse_args()
    src = json.loads(Path(a.src).read_text())
    if not (Path(a.into_root) / MANIFEST_NAME).exists():
        raise SystemExit(f"{Path(a.into_root) / MANIFEST_NAME} not found (download the remote manifest first)")
    before = sorted(read_manifest(a.into_root).get("categories", {}))
    after = sorted(merge_sections(src, Path(a.into_root), a.section or ["sft"])["categories"])
    print(f"categories before: {before}\ncategories after:  {after}")


if __name__ == "__main__":
    main()
