# data/dedup.py
"""Exact and near-duplicate detection.

* ``hash_text`` / ``deduplicate_documents``: exact SHA-256 dedup (original helpers).
* ``normalize_for_hashing``: lowercase, strip punctuation, collapse whitespace, so
  copies that differ only in formatting hash the same.
* ``minhash_signature`` + ``near_duplicate_clusters``: MinHash (5-word shingles,
  128 permutations) with LSH banding (16 bands x 8 rows) and a verification step
  (estimated Jaccard >= threshold, default 0.8). Implemented with NumPy instead
  of datasketch's per-shingle Python loop: the legal build hashes ~1.5M long
  documents, and this is ~20x faster with the same parameters.
* ``ngram_hashes``: hashed word n-grams (13-grams for decontamination).
"""

import hashlib
import re
import zlib

import numpy as np


def hash_text(text: str) -> str:
    """Produce a fingerprint for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate_documents(documents: list[str]) -> list[str]:
    """
    Remove exact-duplicate documents, keeping the first occurrence of each.
    """
    seen_hashes = set()
    unique_documents = []

    for doc in documents:
        fingerprint = hash_text(doc)
        if fingerprint not in seen_hashes:
            seen_hashes.add(fingerprint)
            unique_documents.append(doc)

    return unique_documents


_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_for_hashing(text: str) -> str:
    """Lowercase, replace punctuation/symbol runs with a space, collapse whitespace."""
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def exact_key(text: str) -> int:
    """64-bit hash of the normalised text (exact dedup key)."""
    return int.from_bytes(hashlib.blake2b(normalize_for_hashing(text).encode(), digest_size=8).digest(), "little")


# ---------------------------------------------------------------- MinHash
NUM_PERM = 128
SHINGLE = 5
_PRIME = np.uint64(4294967311)          # smallest prime above 2**32
_rng = np.random.default_rng(1)
_A = _rng.integers(1, 2**31, NUM_PERM, dtype=np.uint64)   # a < 2**31 keeps a*x+b < 2**64
_B = _rng.integers(0, 2**31, NUM_PERM, dtype=np.uint64)
_MIX = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                 0x27D4EB2F165667C5, 0xFF51AFD7ED558CCD], dtype=np.uint64)


def _word_hashes(normalized: str) -> np.ndarray:
    words = normalized.split()
    return np.fromiter((zlib.crc32(w.encode()) for w in words), dtype=np.uint64, count=len(words))


def _combine(wh: np.ndarray, n: int, mix: np.ndarray) -> np.ndarray:
    """Order-sensitive 64-bit hash of every window of n consecutive word hashes."""
    m = len(wh) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = (h ^ (wh[k:k + m] * mix[k % len(mix)])) * np.uint64(0x100000001B3)
    return h


def minhash_signature(text: str, normalized: bool = False) -> np.ndarray:
    """uint32[128] MinHash signature over 5-word shingles of the normalised text."""
    norm = text if normalized else normalize_for_hashing(text)
    wh = _word_hashes(norm)
    if len(wh) < SHINGLE:
        wh = np.concatenate([wh, np.zeros(SHINGLE - len(wh), dtype=np.uint64)])
    sh = _combine(wh, SHINGLE, _MIX)
    x = (sh ^ (sh >> np.uint64(32))) & np.uint64(0xFFFFFFFF)
    sig = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(x), 8192):   # bounded memory for very long documents
        blk = x[start:start + 8192]
        vals = (np.outer(_A, blk) + _B[:, None]) % _PRIME
        np.minimum(sig, vals.min(axis=1), out=sig)
    return sig.astype(np.uint32)


def ngram_hashes(text: str, n: int = 13, normalized: bool = False, unique: bool = True) -> np.ndarray:
    """uint64 hashes of word n-grams of the normalised text: sorted unique (default), or one per
    position in text order (``unique=False``; element i is the n-gram starting at word i)."""
    norm = text if normalized else normalize_for_hashing(text)
    wh = _word_hashes(norm)
    if len(wh) < n:
        return np.zeros(0, dtype=np.uint64)
    h = _combine(wh, n, _MIX)
    return np.unique(h) if unique else h


def longest_runs(positions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Per document, the longest run of consecutive integers in its slice of ``positions``
    (each slice sorted ascending; ``offsets`` has n+1 entries). 0 for empty slices."""
    n = len(offsets) - 1
    out = np.zeros(n, dtype=np.int64)
    if not len(positions):
        return out
    doc = np.repeat(np.arange(n), np.diff(offsets))
    # a new run starts at each element that is not previous+1 within the same document
    start = np.ones(len(positions), dtype=bool)
    start[1:] = (np.diff(positions) != 1) | (doc[1:] != doc[:-1])
    idx = np.flatnonzero(start)
    lengths = np.diff(np.append(idx, len(positions)))
    np.maximum.at(out, doc[idx], lengths)
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)   # root = smallest index


def near_duplicate_clusters(signatures: np.ndarray, threshold: float = 0.8, bands: int = 16,
                            exact_keys: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Cluster documents whose estimated Jaccard similarity is >= threshold.

    ``signatures``: uint32[n, 128]. ``exact_keys`` (optional, uint64[n]) also
    joins documents with identical normalised text. Returns (root, stats):
    root[i] is the smallest document index in i's cluster. LSH candidates are
    checked against the first member of each band bucket (no all-pairs blow-up
    on large buckets of boilerplate).
    """
    n, perm = signatures.shape
    rows = perm // bands
    uf = _UnionFind(n)
    stats = {"exact_pairs": 0, "candidate_pairs": 0, "verified_pairs": 0}
    if exact_keys is not None and n:
        order = np.argsort(exact_keys, kind="stable")
        ks = exact_keys[order]
        same = np.flatnonzero(ks[1:] == ks[:-1])
        for i in same:
            uf.union(int(order[i]), int(order[i + 1]))
        stats["exact_pairs"] = int(len(same))
    for b in range(bands):
        block = np.ascontiguousarray(signatures[:, b * rows:(b + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        starts = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
        ends = np.append(starts[1:], n)
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            first = int(order[s])
            members = order[s + 1:e]
            stats["candidate_pairs"] += len(members)
            sim = (signatures[members] == signatures[first]).mean(axis=1)
            for m in members[sim >= threshold]:
                stats["verified_pairs"] += 1
                uf.union(first, int(m))
    roots = np.array([uf.find(i) for i in range(n)], dtype=np.int64)
    return roots, stats
