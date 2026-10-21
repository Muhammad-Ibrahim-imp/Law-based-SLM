"""SFT data prep: chat datasets -> packed, loss-masked uint16 shards.

First source: ``HuggingFaceTB/smol-smoltalk`` (columns ``messages``, ``source``;
train ~460k / test ~24k rows; Apache 2.0). The law-specific SFT data, once
chosen, goes through the same code under another ``source_name``.

Pipeline
--------
1. Load the split and shuffle it with a fixed seed (exact permutation, so a
   ``max_tokens`` cap gives an unbiased subset and resume is deterministic).
2. Encode each conversation with ``tokenizer.chat.ChatEncoder`` -> (ids, mask).
3. Drop examples that cannot be used, counting each reason:
   ``bad_format``, ``special_text`` (literal special-token strings in content),
   ``empty_assistant``, ``no_assistant`` (nothing to train on), ``too_long``
   (> sequence_length tokens; dropped, never truncated).
4. Pack whole conversations into rows of exactly ``sequence_length`` tokens
   with a best-fit packer over a small set of open rows. A conversation never
   crosses a row (and therefore never crosses a shard). Leftover space is
   filled with the pad ID (``<|endoftext|>``) and mask 0.
5. Write shards via ``data/shard_writer.py``. Each shard has three files:

   * ``shard_XXXXX.bin``        uint16 token IDs, ``num_rows * sequence_length``
   * ``shard_XXXXX.mask.bin``   uint8, 1 = token is a training target
   * ``shard_XXXXX.spans.npy``  int32 ``(n, 2)``: (start, length) of every
                                conversation within the shard, for validation
                                and for optional per-document attention masking

Layout: ``dataset/sft/<source_name>/<split>/`` with the manifest entry at
``manifest["categories"]["sft"]["sources"][<source_name>][<split>]``, i.e. in
the same ``categories`` map that ``data/general_prep.py`` and
``ShardedTokenDataset`` use. If SFT prep creates the manifest, it writes the
same top-level header (``manifest_version``, ``dtype``, ``tokenizer_checkpoint``,
``tokenizer_fingerprint``) so general/legal prep can add to it later.

Sequence length defaults to 8,192, the context the 200M models were
pre-trained at (``configs/training/200m``); conversations longer than that are
dropped. Pass ``--sequence-length 2048`` to pack shorter rows instead.

Labels are NOT stored: ``data/sft_dataset.py`` builds them as
``labels[t] = ids[t+1] if mask[t+1] else -100``.

Packing note: packed conversations share a row with no attention reset, so a
later conversation can attend to an earlier one. This is common practice and
the loss mask already prevents cross-conversation targets; the spans file makes
document-masked attention possible later if Member B wants it.

Resume: after each completed shard the manifest entry is checkpointed (status
``in_progress``) and the packer state is saved to ``_checkpoint.npz``. Rerunning
with the same config continues from there and produces byte-identical shards.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from data.shard_writer import (
    ShardWriter, StreamSpec, read_manifest, sha256_file, update_manifest, utc_now_iso,
)
from tokenizer.chat import CHAT_FORMAT_VERSION, ChatEncoder, ChatFormatError

CATEGORY = "sft"
MANIFEST_VERSION = "1.0"   # matches data/general_prep.py
CHECKPOINT_NAME = "_checkpoint.npz"
STREAMS = {"tokens": StreamSpec("uint16", ".bin"), "mask": StreamSpec("uint8", ".mask.bin")}
SPANS_SUFFIX = ".spans.npy"


@dataclass
class SFTPrepConfig:
    dataset_root: str
    source_name: str = "smol_smoltalk"
    hf_dataset: str = "HuggingFaceTB/smol-smoltalk"
    split: str = "train"
    seed: int = 1234
    sequence_length: int = 8192
    shard_tokens: int = 2 ** 28
    max_tokens: int | None = None      # cap on kept (non-pad) tokens; None = all
    max_open_rows: int = 64
    vocab_size: int = 49_152
    # Options passed to the source loader (data/sft_sources.py), e.g. LawInstruct
    # subsets / per-subset cap. Part of the resume key. Use lists, not tuples.
    loader_options: dict | None = None

    def check(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.shard_tokens < self.sequence_length:
            raise ValueError(
                f"shard_tokens ({self.shard_tokens}) < sequence_length ({self.sequence_length})")
        if self.shard_tokens % self.sequence_length:
            raise ValueError("shard_tokens must be a multiple of sequence_length")
        if self.vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError("vocab does not fit in uint16")
        if self.max_open_rows < 1:
            raise ValueError("max_open_rows must be >= 1")

    # Fields that must match for a resume to be valid.
    def resume_key(self) -> dict:
        d = asdict(self)
        d.pop("dataset_root")
        return d


# ===================================================================== packer
class _Row:
    __slots__ = ("ids", "mask", "spans")

    def __init__(self):
        self.ids: list[int] = []
        self.mask: list[int] = []
        self.spans: list[tuple[int, int]] = []


class BestFitPacker:
    """Packs variable-length examples into fixed-length rows.

    Each example goes into the open row with the least free space that still
    fits it; if none fits, a new row opens. When more than ``max_open`` rows
    are open, the fullest one is emitted. Deterministic for a given input
    order. Emitted rows wait in ``ready`` until the caller takes them.
    """

    def __init__(self, seq_len: int, pad_id: int, max_open: int):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.max_open = max_open
        self.open: list[_Row] = []
        self.ready: list[_Row] = []

    def add(self, ids: Sequence[int], mask: Sequence[int]) -> None:
        n = len(ids)
        if n > self.seq_len:
            raise ValueError("example longer than a row")
        best, best_rem = None, None
        for r in self.open:
            rem = self.seq_len - len(r.ids)
            if rem >= n and (best_rem is None or rem < best_rem):
                best, best_rem = r, rem
        if best is None:
            best = _Row()
            self.open.append(best)
        best.spans.append((len(best.ids), n))
        best.ids.extend(ids)
        best.mask.extend(mask)
        if len(best.ids) == self.seq_len:
            self.open.remove(best)
            self.ready.append(best)
        while len(self.open) > self.max_open:
            fullest = min(self.open, key=lambda r: self.seq_len - len(r.ids))
            self.open.remove(fullest)
            self.ready.append(fullest)

    def flush(self) -> None:
        self.ready.extend(self.open)
        self.open = []

    def pop_ready(self) -> _Row | None:
        return self.ready.pop(0) if self.ready else None

    def padded(self, row: _Row) -> tuple[np.ndarray, np.ndarray]:
        ids = np.full(self.seq_len, self.pad_id, dtype=np.uint16)
        mask = np.zeros(self.seq_len, dtype=np.uint8)
        ids[:len(row.ids)] = row.ids
        mask[:len(row.mask)] = row.mask
        return ids, mask

    # -- checkpoint state (open rows + ready rows, in order)
    def state_arrays(self) -> dict[str, np.ndarray]:
        rows = self.ready + self.open
        lens, ids, mask, spans, span_rows = [], [], [], [], []
        for k, r in enumerate(rows):
            lens.append(len(r.ids))
            ids.extend(r.ids)
            mask.extend(r.mask)
            spans.extend(r.spans)
            span_rows.extend([k] * len(r.spans))
        return {
            "n_ready": np.array([len(self.ready)], dtype=np.int64),
            "row_lens": np.array(lens, dtype=np.int64),
            "ids": np.array(ids, dtype=np.uint16),
            "mask": np.array(mask, dtype=np.uint8),
            "spans": np.array(spans, dtype=np.int64).reshape(-1, 2),
            "span_rows": np.array(span_rows, dtype=np.int64),
        }

    def load_state(self, st: Mapping[str, np.ndarray]) -> None:
        rows, pos = [], 0
        for k, ln in enumerate(st["row_lens"].tolist()):
            r = _Row()
            r.ids = st["ids"][pos:pos + ln].tolist()
            r.mask = st["mask"][pos:pos + ln].tolist()
            r.spans = [tuple(s) for s in st["spans"][st["span_rows"] == k].tolist()]
            rows.append(r)
            pos += ln
        n_ready = int(st["n_ready"][0])
        self.ready, self.open = rows[:n_ready], rows[n_ready:]


# ===================================================================== stats
def _new_stats() -> dict:
    return {
        "examples_seen": 0, "examples_kept": 0,
        "dropped": {k: 0 for k in
                    ("bad_format", "special_text", "empty_assistant", "no_assistant", "too_long")},
        "kept_tokens": 0, "trained_tokens": 0, "padding_tokens": 0, "rows": 0,
        "per_source": {},
    }


def _bump_source(stats: dict, src: str, n_tok: int, n_train: int) -> None:
    s = stats["per_source"].setdefault(src, {"examples": 0, "tokens": 0, "trained_tokens": 0})
    s["examples"] += 1
    s["tokens"] += n_tok
    s["trained_tokens"] += n_train


def _prepare_example(encoder: ChatEncoder, messages, seq_len: int):
    """Returns (ids, mask, None) or (None, None, drop_reason)."""
    try:
        encoder.check_messages(messages)
    except ChatFormatError:
        return None, None, "bad_format"
    if encoder.find_special_text(messages) is not None:
        return None, None, "special_text"
    if any(m["role"] == "assistant" and not m["content"].strip() for m in messages):
        return None, None, "empty_assistant"
    ids, mask = encoder.encode(messages)
    if not any(mask):
        return None, None, "no_assistant"
    if len(ids) > seq_len:
        return None, None, "too_long"
    return ids, mask, None


# ===================================================================== build
class _StopForTest(RuntimeError):
    """Raised by the test hook to simulate a crash after N shards."""


def _entry_path(manifest: dict, source: str, split: str):
    """The manifest entry for one SFT source/split, or None."""
    return (manifest.get("categories", {}).get(CATEGORY, {})
            .get("sources", {}).get(source, {}).get(split))


def _check_manifest_tokenizer(manifest: dict, fingerprint: str) -> None:
    """Refuse to mix shards from different tokenizers in one dataset root."""
    if CATEGORY in manifest:
        raise RuntimeError("manifest has a top-level 'sft' key from an older version of "
                           "this script; delete it (SFT now lives under 'categories').")
    existing = manifest.get("tokenizer_fingerprint")
    if existing is not None and existing != fingerprint:
        raise RuntimeError(f"manifest tokenizer_fingerprint {existing!r} != this tokenizer's "
                           f"{fingerprint!r}; refusing to write SFT shards into this root")


def _ensure_header(m: dict, fingerprint: str) -> None:
    """Same top-level keys as data/general_prep.py::_load_or_init_manifest."""
    try:
        from tokenizer import TOKENIZER_CHECKPOINT
    except Exception:  # pragma: no cover - only if tokenizer/ is unavailable
        TOKENIZER_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"
    m.setdefault("manifest_version", MANIFEST_VERSION)
    m.setdefault("dtype", "uint16")
    m.setdefault("tokenizer_checkpoint", TOKENIZER_CHECKPOINT)
    m.setdefault("tokenizer_fingerprint", fingerprint)
    m.setdefault("categories", {})


def _set_entry(root: Path, source: str, split: str, entry: dict | None,
               fingerprint: str) -> None:
    def fn(m):
        _check_manifest_tokenizer(m, fingerprint)
        _ensure_header(m, fingerprint)
        sft = m["categories"].setdefault(CATEGORY, {})
        sft["layout"] = (
            "rows of sequence_length tokens; per shard: .bin uint16 tokens, .mask.bin uint8 "
            "loss mask (1 = target), .spans.npy int32 (start, length) per conversation; "
            "labels[t] = ids[t+1] if mask[t+1] else -100"
        )
        src = sft.setdefault("sources", {}).setdefault(source, {})
        if entry is None:
            src.pop(split, None)
        else:
            src[split] = entry
    update_manifest(root, fn)


def _save_checkpoint(out_dir: Path, packer: BestFitPacker) -> None:
    tmp = out_dir / (CHECKPOINT_NAME + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **packer.state_arrays())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_dir / CHECKPOINT_NAME)


def build_sft_split(cfg: SFTPrepConfig, examples, tokenizer, tokenizer_fingerprint: str,
                    overwrite: bool = False, measure_only: bool = False,
                    source_info: dict | None = None,
                    _stop_after_shards: int | None = None) -> dict:
    """Build (or resume) one source/split. ``examples`` must be an indexable
    sequence of dicts with ``messages`` (and optionally ``source``), already in
    final (shuffled) order. Returns the manifest entry, or a report when
    ``measure_only``."""
    cfg.check()
    root = Path(cfg.dataset_root)
    out_dir = root / CATEGORY / cfg.source_name / cfg.split
    encoder = ChatEncoder(tokenizer)
    L = cfg.sequence_length
    rows_per_shard = cfg.shard_tokens // L
    n_total = len(examples)

    base_entry = {
        "config": cfg.resume_key(),
        "num_examples_in_source": n_total,
        "chat_format_version": CHAT_FORMAT_VERSION,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "special_ids": {"im_start": encoder.im_start_id, "im_end": encoder.im_end_id,
                        "pad": encoder.pad_id},
        "dtype": {"tokens": "uint16", "mask": "uint8", "spans": "int32"},
        "sequence_length": L,
        "shard_tokens": cfg.shard_tokens,
        "source_info": source_info or {},
    }

    packer = BestFitPacker(L, encoder.pad_id, cfg.max_open_rows)
    stats = _new_stats()
    shards: list[dict] = []
    start = 0

    # ---------------------------------------------------- existing state
    if not measure_only:
        manifest = read_manifest(root)
        _check_manifest_tokenizer(manifest, tokenizer_fingerprint)
        existing = _entry_path(manifest, cfg.source_name, cfg.split)
        if existing and overwrite:
            existing = None
            _set_entry(root, cfg.source_name, cfg.split, None, tokenizer_fingerprint)
            shutil.rmtree(out_dir, ignore_errors=True)
        elif existing is None and out_dir.exists() and any(out_dir.iterdir()):
            if not overwrite:
                raise RuntimeError(f"{out_dir} has files but no manifest entry; "
                                   "pass overwrite=True to rebuild")
            shutil.rmtree(out_dir)
        if existing:
            if existing.get("status") == "complete":
                raise RuntimeError(f"{cfg.source_name}/{cfg.split} already built; "
                                   "pass overwrite=True to rebuild")
            for k in ("config", "num_examples_in_source", "chat_format_version",
                      "tokenizer_fingerprint"):
                if existing.get(k) != base_entry[k]:
                    raise RuntimeError(f"cannot resume: {k} changed "
                                       f"({existing.get(k)!r} -> {base_entry[k]!r}); "
                                       "pass overwrite=True to rebuild")
            shards = existing["shards"]
            for s in shards:
                if not (out_dir / s["files"]["spans"]).exists():
                    raise RuntimeError(f"cannot resume: {s['files']['spans']} missing")
                for stream in STREAMS:
                    p = out_dir / s["files"][stream]
                    want = s["num_tokens"] * np.dtype(STREAMS[stream].dtype).itemsize
                    if not p.exists() or p.stat().st_size != want:
                        raise RuntimeError(f"cannot resume: {p} missing or wrong size")
            stats = existing["stats"]
            start = existing["examples_consumed"]
            with np.load(out_dir / CHECKPOINT_NAME) as st:
                packer.load_state(dict(st))
        out_dir.mkdir(parents=True, exist_ok=True)
        for stray in out_dir.glob("*.tmp"):
            stray.unlink()

    writer = None if measure_only else ShardWriter(
        out_dir, cfg.shard_tokens, STREAMS, start_index=len(shards))
    shard_spans: list[tuple[int, int]] = []
    lengths: list[int] = []
    consumed = start
    shards_this_run = 0

    def record_shard(info) -> None:
        spans_name = f"shard_{info.index:05d}{SPANS_SUFFIX}"
        spans_arr = np.array(shard_spans, dtype=np.int32).reshape(-1, 2)
        with open(out_dir / (spans_name + ".tmp"), "wb") as f:
            np.save(f, spans_arr)
        os.replace(out_dir / (spans_name + ".tmp"), out_dir / spans_name)
        d = info.to_json()
        d["files"]["spans"] = spans_name
        d["sha256"]["spans"] = sha256_file(out_dir / spans_name)
        d["num_rows"] = info.num_tokens // L
        d["num_examples"] = len(shard_spans)
        shards.append(d)
        shard_spans.clear()

    def emit_ready(final: bool = False) -> None:
        nonlocal shards_this_run
        while (row := packer.pop_ready()) is not None:
            stats["rows"] += 1
            stats["padding_tokens"] += L - len(row.ids)
            if measure_only:
                continue
            row_in_shard = (stats["rows"] - 1) % rows_per_shard
            shard_spans.extend((row_in_shard * L + s, n) for s, n in row.spans)
            ids, mask = packer.padded(row)
            for info in writer.write(tokens=ids, mask=mask):
                record_shard(info)
                _save_checkpoint(out_dir, packer)
                _set_entry(root, cfg.source_name, cfg.split, {
                    **base_entry, "status": "in_progress", "updated_at": utc_now_iso(),
                    "examples_consumed": consumed, "stats": stats, "shards": shards},
                    tokenizer_fingerprint)
                shards_this_run += 1
                if _stop_after_shards is not None and shards_this_run >= _stop_after_shards \
                        and not final:
                    raise _StopForTest()

    # ---------------------------------------------------- main loop
    budget_hit = cfg.max_tokens is not None and stats["kept_tokens"] >= cfg.max_tokens
    for i in range(start, n_total if not budget_hit else start):
        ex = examples[i]
        stats["examples_seen"] += 1
        ids, mask, reason = _prepare_example(encoder, ex["messages"], L)
        consumed = i + 1
        if reason:
            stats["dropped"][reason] += 1
        else:
            n_train = int(sum(mask))
            stats["examples_kept"] += 1
            stats["kept_tokens"] += len(ids)
            stats["trained_tokens"] += n_train
            _bump_source(stats, ex.get("source") or "unknown", len(ids), n_train)
            if measure_only:
                lengths.append(len(ids))
            packer.add(ids, mask)
        emit_ready()
        if cfg.max_tokens is not None and stats["kept_tokens"] >= cfg.max_tokens:
            budget_hit = True
            break

    packer.flush()
    emit_ready(final=True)

    if measure_only:
        arr = np.array(lengths) if lengths else np.zeros(1)
        total_slots = stats["rows"] * L
        return {
            "source_info": source_info or {},
            **stats,
            "budget_reached": budget_hit,
            "length_percentiles": {p: int(np.percentile(arr, p)) for p in (50, 90, 99, 100)},
            "padding_fraction": (stats["padding_tokens"] / total_slots) if total_slots else 0.0,
            "trained_fraction_of_kept": (stats["trained_tokens"] / stats["kept_tokens"])
            if stats["kept_tokens"] else 0.0,
        }

    final = writer.close()
    if final is not None:
        record_shard(final)
    entry = {**base_entry, "status": "complete", "created_at": utc_now_iso(),
             "examples_consumed": consumed, "budget_reached": budget_hit,
             "total_tokens": stats["rows"] * L, "stats": stats, "shards": shards}
    _set_entry(root, cfg.source_name, cfg.split, entry, tokenizer_fingerprint)
    ckpt = out_dir / CHECKPOINT_NAME
    if ckpt.exists():
        ckpt.unlink()
    return entry


def load_hf_split(cfg: SFTPrepConfig):
    """smol-smoltalk only; kept for compatibility. Use data.sft_sources.load_source."""
    from data.sft_sources import load_smol_smoltalk
    return load_smol_smoltalk(cfg.split, seed=cfg.seed)[0]


# ===================================================================== checks
def validate_sft_split(dataset_root, source: str, split: str,
                       vocab_size: int = 49_152, check_hashes: bool = True) -> dict:
    """Full write-time validation; raises ValueError on the first problem."""
    root = Path(dataset_root)
    entry = _entry_path(read_manifest(root), source, split)
    if not entry:
        raise ValueError(f"no manifest entry for sft/{source}/{split}")
    if entry.get("status") != "complete":
        raise ValueError(f"sft/{source}/{split} status is {entry.get('status')!r}")
    out_dir = root / CATEGORY / source / split
    L = entry["sequence_length"]
    sp = entry["special_ids"]
    tot = Counter()
    for s in entry["shards"]:
        n = s["num_tokens"]
        if n % L:
            raise ValueError(f"shard {s['index']}: {n} tokens is not a multiple of {L}")
        for stream in ("tokens", "mask", "spans"):
            p = out_dir / s["files"][stream]
            if not p.exists():
                raise ValueError(f"missing {p}")
            if check_hashes and sha256_file(p) != s["sha256"][stream]:
                raise ValueError(f"checksum mismatch: {p}")
        ids = np.fromfile(out_dir / s["files"]["tokens"], dtype=np.uint16)
        mask = np.fromfile(out_dir / s["files"]["mask"], dtype=np.uint8)
        spans = np.load(out_dir / s["files"]["spans"])
        if len(ids) != n or len(mask) != n:
            raise ValueError(f"shard {s['index']}: length mismatch")
        if ids.max(initial=0) >= vocab_size:
            raise ValueError(f"shard {s['index']}: token ID >= {vocab_size}")
        if mask.max(initial=0) > 1:
            raise ValueError(f"shard {s['index']}: mask values outside {{0,1}}")
        if len(spans) != s["num_examples"]:
            raise ValueError(f"shard {s['index']}: span count mismatch")
        covered = np.zeros(n, dtype=bool)
        prev_end = 0
        for st, ln in spans.tolist():
            if st < prev_end or ln <= 0 or st + ln > n:
                raise ValueError(f"shard {s['index']}: bad span ({st},{ln})")
            if st // L != (st + ln - 1) // L:
                raise ValueError(f"shard {s['index']}: span ({st},{ln}) crosses a row")
            if ids[st] != sp["im_start"]:
                raise ValueError(f"shard {s['index']}: span at {st} does not start with im_start")
            if not mask[st:st + ln].any():
                raise ValueError(f"shard {s['index']}: span at {st} has no trained tokens")
            covered[st:st + ln] = True
            prev_end = st + ln
        if mask[~covered].any() or (ids[~covered] != sp["pad"]).any():
            raise ValueError(f"shard {s['index']}: non-pad data outside spans")
        tot["tokens"] += n
        tot["kept"] += int(covered.sum())
        tot["trained"] += int(mask.sum())
        tot["examples"] += len(spans)
    st = entry["stats"]
    for k, want in (("kept", st["kept_tokens"]), ("trained", st["trained_tokens"]),
                    ("examples", st["examples_kept"]), ("tokens", entry["total_tokens"])):
        if tot[k] != want:
            raise ValueError(f"total {k}: shards have {tot[k]}, manifest says {want}")
    return dict(tot)


def decode_row(dataset_root, source: str, split: str, tokenizer,
               shard: int = 0, row: int = 0) -> str:
    """Decode one row, wrapping trained tokens in ⟦ ⟧ and showing padding as
    a count, for eyeballing the mask."""
    root = Path(dataset_root)
    entry = _entry_path(read_manifest(root), source, split)
    s = entry["shards"][shard]
    L = entry["sequence_length"]
    out_dir = root / CATEGORY / source / split
    ids = np.memmap(out_dir / s["files"]["tokens"], dtype=np.uint16, mode="r")[row * L:(row + 1) * L]
    mask = np.memmap(out_dir / s["files"]["mask"], dtype=np.uint8, mode="r")[row * L:(row + 1) * L]
    spans = np.load(out_dir / s["files"]["spans"])
    pad = entry["special_ids"]["pad"]
    end = max((st + ln - row * L for st, ln in spans.tolist()
               if row * L <= st < (row + 1) * L), default=0)
    out, i = [], 0
    while i < end:
        j = i
        while j < end and mask[j] == mask[i]:
            j += 1
        text = tokenizer.decode(ids[i:j].tolist())
        out.append(f"⟦{text}⟧" if mask[i] else text)
        i = j
    out.append(f"\n[{L - end} pad tokens (id {pad})]")
    return "".join(out)
