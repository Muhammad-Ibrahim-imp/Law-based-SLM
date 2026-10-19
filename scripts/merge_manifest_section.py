"""Merge one ``categories`` section (default ``sft``) from one manifest into another.

Use it before uploading SFT shards to a Modal volume whose manifest already
holds other categories (e.g. ``general``): uploading the local manifest.json
as-is would overwrite those. Steps are in reports/sft_data.md.

    python -m scripts.merge_manifest_section --from dataset/manifest.json --into-root remote_manifest

``--into-root`` is the directory holding the target ``manifest.json``; the
write goes through ``data.shard_writer.update_manifest`` (locked, atomic).
The section in the target is replaced as a whole. Refuses when the two
manifests were built with different tokenizers.
"""

import argparse
import json
from pathlib import Path

from data.shard_writer import MANIFEST_NAME, read_manifest, update_manifest

HEADER_KEYS = ("manifest_version", "dtype", "tokenizer_checkpoint", "tokenizer_fingerprint")


def merge_section(src_manifest: dict, dst_root: Path, section: str = "sft") -> dict:
    if section not in src_manifest.get("categories", {}):
        raise ValueError(f"source manifest has no categories.{section}")
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
        dst.setdefault("categories", {})[section] = src_manifest["categories"][section]

    return update_manifest(Path(dst_root), fn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, help="source manifest.json")
    ap.add_argument("--into-root", required=True, help=f"directory holding the target {MANIFEST_NAME}")
    ap.add_argument("--section", default="sft")
    a = ap.parse_args()
    src = json.loads(Path(a.src).read_text())
    if not (Path(a.into_root) / MANIFEST_NAME).exists():
        raise SystemExit(f"{Path(a.into_root) / MANIFEST_NAME} not found (download the remote manifest first)")
    before = sorted(read_manifest(a.into_root).get("categories", {}))
    after = sorted(merge_section(src, Path(a.into_root), a.section)["categories"])
    print(f"categories before: {before}\ncategories after:  {after}")


if __name__ == "__main__":
    main()
