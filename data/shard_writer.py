"""Shared shard-writing helpers for the legal and SFT prep scripts.

These carry the fixes for the run-behaviour issues recorded in
``data/general_prep.py`` (KNOWN QUIRKS); that file stays frozen.

* Preallocated NumPy buffers instead of a Python list of tokens.
* Several aligned streams per shard (e.g. uint16 tokens + uint8 loss mask).
* Each shard is written to a temp file and renamed into place, so a crash never
  leaves a half-written ``shard_XXXXX.bin`` behind.
* Manifest updates take a file lock and are written atomically, so two prep
  scripts finishing at the same time cannot overwrite each other's sections.
* Timestamps are real UTC.
* Errors are raised explicitly (no ``assert``), so ``python -O`` keeps them.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

if os.name == "nt":
    import msvcrt
else:
    import fcntl

MANIFEST_NAME = "manifest.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- manifest IO
@contextmanager
def _manifest_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / (MANIFEST_NAME + ".lock"), "a+") as lock:
        if os.name == "nt":
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write("\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def read_manifest(root: Path) -> dict:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def update_manifest(root: Path, fn: Callable[[dict], None]) -> dict:
    """Locked read-modify-write of the shared manifest. ``fn`` mutates the
    dict in place; only the sections it touches change."""
    root = Path(root)
    with _manifest_lock(root):
        manifest = read_manifest(root)
        fn(manifest)
        tmp = root / (MANIFEST_NAME + ".tmp")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, root / MANIFEST_NAME)
    return manifest


# ---------------------------------------------------------------- shard writer
@dataclass
class StreamSpec:
    dtype: str      # numpy dtype name, e.g. "uint16"
    suffix: str     # file suffix, e.g. ".bin" or ".mask.bin"


@dataclass
class ShardInfo:
    index: int
    num_tokens: int
    files: dict[str, str] = field(default_factory=dict)    # stream -> file name
    sha256: dict[str, str] = field(default_factory=dict)   # stream -> digest

    def to_json(self) -> dict:
        return {"index": self.index, "num_tokens": self.num_tokens,
                "files": self.files, "sha256": self.sha256}


class ShardWriter:
    """Writes aligned streams into fixed-size shards.

    ``write(tokens=..., mask=...)`` takes arrays of equal length; data that
    crosses a shard boundary is split. Returns the shards completed by that
    call so the caller can checkpoint after each one.
    """

    def __init__(self, out_dir: Path, shard_tokens: int,
                 streams: dict[str, StreamSpec], start_index: int = 0):
        if shard_tokens <= 0:
            raise ValueError("shard_tokens must be positive")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.streams = streams
        self.index = start_index
        self._fill = 0
        self._buf = {n: np.empty(shard_tokens, dtype=s.dtype) for n, s in streams.items()}

    def file_name(self, stream: str, index: int) -> str:
        return f"shard_{index:05d}{self.streams[stream].suffix}"

    def write(self, **arrays: np.ndarray) -> list[ShardInfo]:
        if set(arrays) != set(self.streams):
            raise ValueError(f"expected streams {sorted(self.streams)}, got {sorted(arrays)}")
        lengths = {len(a) for a in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"streams have different lengths: {lengths}")
        n = lengths.pop()
        done: list[ShardInfo] = []
        pos = 0
        while pos < n:
            take = min(n - pos, self.shard_tokens - self._fill)
            for name, a in arrays.items():
                self._buf[name][self._fill:self._fill + take] = a[pos:pos + take]
            self._fill += take
            pos += take
            if self._fill == self.shard_tokens:
                done.append(self._flush())
        return done

    def close(self) -> ShardInfo | None:
        """Write the final partial shard, if any."""
        return self._flush() if self._fill else None

    def _flush(self) -> ShardInfo:
        info = ShardInfo(index=self.index, num_tokens=self._fill)
        for name in self.streams:
            fname = self.file_name(name, self.index)
            final = self.out_dir / fname
            tmp = self.out_dir / (fname + ".tmp")
            with open(tmp, "wb") as f:
                f.write(self._buf[name][:self._fill].tobytes())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, final)
            info.files[name] = fname
            info.sha256[name] = sha256_file(final)
        self.index += 1
        self._fill = 0
        return info
