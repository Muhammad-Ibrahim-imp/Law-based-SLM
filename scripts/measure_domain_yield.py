"""Measure unique pretraining text available for FAMILY, TAX and CRIMINAL law.

Analysis only: reads Hugging Face files (raw .jsonl.xz / parquet / arrow,
never loading scripts), writes nothing under dataset/. Results go to
results/domain_measurement/ (git-ignored). Resumable: every stage writes
results/domain_measurement/<stage>.json (scans also work/<file>.npz) and is
skipped when that file exists. Rerun to continue after an interruption.

    python -m scripts.measure_domain_yield                # all stages
    python -m scripts.measure_domain_yield --stage cl     # one stage
    python -m scripts.measure_domain_yield --stage spot   # precision samples
    python -m scripts.measure_domain_yield --stage summary --force

Stages: pol_tax, title26, state_family, cl, coldcases, hfforlegal,
neardup, summary, spot. Tax follow-up: state_tax, fedreg_tax,
coldcases_state_tax, taxrulings_neardup.
"""

import argparse
import hashlib
import io
import json
import lzma
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/domain_measurement")
WORK = OUT / "work"
POL = "pile-of-law/pile-of-law"
HFL = "HFforLegal/case-law"
COLD = "harvard-lil/cold-cases"
DOWNLOAD_LIMIT = 8.4e9  # 6 GB for the first run (5.37 used) + 3 GB for the tax follow-up stages
SEED = 42

# ---------------------------------------------------------------- keywords
# Keyword lists, matching rules and thresholds live in data/domains.py (shared with SFT prep).
from data.domains import (  # noqa: E402
    DOMAIN_COLS, DOMAINS, NOT_AFTER, TERM_LIST, TERM_RES, TERMS, term_counts,
    PROFILES as THRESHOLDS, classify_counts as classify,
)


# ---------------------------------------------------------------- utilities
def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}  [downloaded {downloaded_bytes() / 1e9:.2f} GB]"
    print(line, flush=True)
    with open(OUT / "progress.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


LEDGER = OUT / "downloads.json"


def downloaded_bytes():
    return sum(v["bytes"] for v in load_json(LEDGER, {}).values())


def record_download(key, nbytes, path=""):
    ledger = load_json(LEDGER, {})
    if key not in ledger:
        ledger[key] = {"bytes": int(nbytes), "path": str(path)}
        save_json(LEDGER, ledger)


_SIZES = {}


def remote_size(repo, filename):
    if repo not in _SIZES:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo, files_metadata=True)
        _SIZES[repo] = {s.rfilename: s.size or 0 for s in info.siblings}
    return _SIZES[repo][filename]


def fetch(repo, filename):
    """Download into the HF cache, enforcing the task download budget."""
    from huggingface_hub import hf_hub_download
    key = f"{repo}/{filename}"
    if key not in load_json(LEDGER, {}):
        size = remote_size(repo, filename)
        if downloaded_bytes() + size > DOWNLOAD_LIMIT:
            raise BudgetExceeded(f"{key} ({size / 1e9:.2f} GB) would exceed the {DOWNLOAD_LIMIT / 1e9:.0f} GB budget")
        log(f"downloading {key} ({size / 1e9:.2f} GB)")
    path = hf_hub_download(repo, filename, repo_type="dataset")
    record_download(key, remote_size(repo, filename), path)
    return path


class BudgetExceeded(RuntimeError):
    pass


def iter_jsonl_xz(path):
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def norm_hash(lower_text):
    return int.from_bytes(hashlib.blake2b(" ".join(lower_text.split()).encode(), digest_size=8).digest(), "little")


WORD_RE = re.compile(r"[a-z0-9]+")


def minhash(lower_text):
    from datasketch import MinHash
    words = WORD_RE.findall(lower_text)
    shingles = {" ".join(words[i:i + 5]).encode() for i in range(max(1, len(words) - 4))}
    m = MinHash(num_perm=128, seed=1, scheme="affine32")
    m.update_batch(list(shingles))
    return m.hashvalues.astype(np.uint64)


def chars_per_token(texts, max_docs=2000, max_chars=400_000_000):
    """Tokenize a random sample with the project tokenizer; returns chars/token."""
    from tokenizer import load_smollm2_tokenizer
    tok = load_smollm2_tokenizer()
    rng = random.Random(SEED)
    sample = texts if len(texts) <= max_docs else rng.sample(texts, max_docs)
    chars = tokens = 0
    for i in range(0, len(sample), 64):
        batch = sample[i:i + 64]
        ids = tok(batch, add_special_tokens=False)["input_ids"]
        chars += sum(map(len, batch))
        tokens += sum(map(len, ids))
        if chars > max_chars:
            break
    return {"chars_per_token": chars / max(tokens, 1), "sample_docs": min(len(sample), i + 64),
            "sample_chars": chars, "sample_tokens": tokens}


def dedup_stats(hashes, lens):
    """Exact dedup: keep the first occurrence of every normalized-text hash."""
    _, first = np.unique(hashes, return_index=True)
    keep = np.zeros(len(hashes), bool)
    keep[first] = True
    return keep


# ---------------------------------------------------------------- scanning
def scan_docs(docs, work_name, minhash_domains=(), snippet_rate=0.05, keep_text_for=None):
    """Keyword-scan an iterable of (doc_id, text). Writes work/<name>.npz.

    Stores per doc: length, normalized hash, term counts. MinHash signatures
    for docs with >=2 distinct terms in any of `minhash_domains`, and a random
    `snippet_rate` sample of matching docs (first 400 chars + matched terms)
    for the precision check.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    npz = WORK / f"{work_name}.npz"
    if npz.exists():
        return np.load(npz, allow_pickle=False)
    rng = random.Random(f"{SEED}-{work_name}")
    ids, lens, hashes, counts, mh_ids, mh_vals, snippets = [], [], [], [], [], [], []
    texts = []  # random text sample for chars/token
    t0 = time.time()
    for n, (doc_id, text) in enumerate(docs):
        lower = text.lower()
        c = term_counts(lower)
        ids.append(doc_id if isinstance(doc_id, int) else n)
        lens.append(len(text))
        hashes.append(norm_hash(lower))
        counts.append(c)
        if c.any():
            if any((c[DOMAIN_COLS[d]] > 0).sum() >= 2 for d in minhash_domains):
                mh_ids.append(n)
                mh_vals.append(minhash(lower))
            if rng.random() < snippet_rate:
                snippets.append({"i": n, "id": doc_id, "len": len(text), "head": text[:400],
                                 "terms": {TERM_LIST[j][1]: int(c[j]) for j in np.flatnonzero(c)}})
            if rng.random() < 0.02 and len(texts) < 3000:
                texts.append(text)
        if n % 20000 == 0 and n:
            log(f"{work_name}: {n} docs, {sum(lens) / 1e9:.2f} G chars, {time.time() - t0:.0f}s")
    cpt = chars_per_token(texts) if texts else {}
    np.savez(npz, ids=np.array(ids, np.int64), lens=np.array(lens, np.int64), hashes=np.array(hashes, np.uint64),
             counts=np.array(counts, np.uint16).reshape(-1, len(TERM_LIST)),
             mh_ids=np.array(mh_ids, np.int64), mh_vals=np.array(mh_vals, np.uint64).reshape(-1, 128))
    with open(WORK / f"{work_name}.snippets.jsonl", "w", encoding="utf-8") as fh:
        for s in snippets:
            fh.write(json.dumps(s) + "\n")
    save_json(WORK / f"{work_name}.cpt.json", cpt)
    log(f"{work_name}: done, {len(lens)} docs in {time.time() - t0:.0f}s")
    return np.load(npz, allow_pickle=False)


def domain_summary(arr, cpt, scale=1.0):
    """Per domain & level: docs, chars, unique (exact-dedup) chars and tokens, scaled."""
    lens, hashes, counts = arr["lens"], arr["hashes"], arr["counts"]
    out = {"docs_scanned": int(len(lens)), "chars_scanned": int(lens.sum()), "scale": scale,
           "chars_per_token": cpt}
    for level in THRESHOLDS:
        masks = {d: classify(counts, lens, d, level) for d in DOMAINS}
        for d in DOMAINS:
            m = masks[d]
            keep = dedup_stats(hashes[m], lens[m])
            uchars = int(lens[m][keep].sum())
            out[f"{d}_{level}"] = {
                "docs": int(m.sum()), "chars": int(lens[m].sum()), "unique_docs": int(keep.sum()),
                "unique_chars": uchars, "unique_tokens_sample": uchars / cpt,
                "unique_tokens_scaled": uchars / cpt * scale,
                "match_rate_docs": float(m.mean()) if len(m) else 0.0,
            }
        out[f"overlap_{level}"] = {
            f"{a}&{b}": {"n": int((masks[a] & masks[b]).sum()),
                         f"pct_of_{a}": float((masks[a] & masks[b]).sum() / max(masks[a].sum(), 1) * 100),
                         f"pct_of_{b}": float((masks[a] & masks[b]).sum() / max(masks[b].sum(), 1) * 100)}
            for a, b in [("family", "criminal"), ("tax", "criminal"), ("family", "tax")]}
    return out


# ---------------------------------------------------------------- stages
def stage_pol_tax():
    """Pile of Law taxrulings + irs_legal_advice_memos: tax by construction."""
    result = {}
    for sub in ["taxrulings", "irs_legal_advice_memos"]:
        texts, lens, hashes, counts = [], [], [], []
        for split in ["train", "validation"]:
            for r in iter_jsonl_xz(fetch(POL, f"data/{split}.{sub}.jsonl.xz")):
                t = r["text"]
                lower = t.lower()
                texts.append(t)
                lens.append(len(t))
                hashes.append(norm_hash(lower))
                counts.append(term_counts(lower))
        lens, hashes = np.array(lens), np.array(hashes, np.uint64)
        counts = np.array(counts, np.uint16)
        keep = dedup_stats(hashes, lens)
        cpt = chars_per_token(texts)
        uchars = int(lens[keep].sum())
        result[sub] = {
            "docs": len(lens), "chars": int(lens.sum()), "unique_docs": int(keep.sum()),
            "unique_chars": uchars, "unique_tokens": uchars / cpt["chars_per_token"], "cpt": cpt,
            "keyword_tax_strict_rate": float(classify(counts, lens, "tax", "strict").mean()),
            "keyword_tax_loose_rate": float(classify(counts, lens, "tax", "loose").mean()),
        }
        # MinHash sample for the cold-cases near-dup check.
        rng = random.Random(SEED)
        idx = rng.sample(range(len(texts)), min(5000, len(texts)))
        np.save(WORK / f"pol_{sub}_minhash.npy", np.array([minhash(texts[i].lower()) for i in idx]))
        log(f"pol_tax {sub}: {result[sub]['unique_tokens'] / 1e6:.1f}M unique tokens")
    return result


SECTION_RE = re.compile(r"\n(?=[ \t]*(?:§|�)+\s*\d)")  # § is sometimes mojibake (U+FFFD)


def stage_title26():
    """Title 26 (Internal Revenue) of uscode and cfr, split into sections for dedup."""
    result = {}
    header = {"uscode": re.compile(r"^\s*Title 26\b", re.M), "cfr": re.compile(r"^\s*Title 26\s*\n\s*Internal Revenue", re.M)}
    for sub in ["uscode", "cfr"]:
        chunks, titles, records = [], Counter(), []
        for split in ["train", "validation"]:
            for r in iter_jsonl_xz(fetch(POL, f"data/{split}.{sub}.jsonl.xz")):
                t = r["text"]
                m = re.search(r"Title (\d+)", t[:2000])
                titles[m.group(1) if m else "?"] += 1
                if header[sub].search(t[:2000]):
                    records.append({"split": split, "url": r.get("url"), "chars": len(t), "head": " ".join(t[:300].split())})
                    chunks.extend(c for c in SECTION_RE.split(t) if c.strip())
        lens = np.array([len(c) for c in chunks])
        hashes = np.array([norm_hash(c.lower()) for c in chunks], np.uint64)
        keep = dedup_stats(hashes, lens)
        cpt = chars_per_token(chunks)
        uchars = int(lens[keep].sum())
        result[sub] = {
            "selection": f"record whose first 2000 chars match {header[sub].pattern!r}",
            "records_selected": records, "title_record_counts": dict(titles),
            "sections": len(chunks), "chars": int(lens.sum()), "unique_sections": int(keep.sum()),
            "unique_chars": uchars, "unique_tokens": uchars / cpt["chars_per_token"], "cpt": cpt,
        }
        log(f"title26 {sub}: {len(records)} records, {result[sub]['unique_tokens'] / 1e6:.1f}M unique tokens")
    return result


HEADING_RE = re.compile(
    r"^[ \t]*(title|division|book|subtitle|chapter|part|article)\s+([0-9]+[\w.\-]*|[ivxlc]+\b)[^\n]{0,200}$",
    re.I | re.M)
LEVEL = {"title": 0, "division": 0, "book": 0, "subtitle": 1, "chapter": 2, "part": 3, "article": 4}
FAMILY_HEADING = re.compile(
    r"domestic relations|marriage|marital|married persons|husband and wife|divorce|dissolution of marriage"
    r"|annulment|separate maintenance|alimony|spousal (?:support|maintenance)|child support|support of (?:minor )?children"
    r"|custody of (?:minor )?child|child custody|custody and visitation|parenting|time-sharing|visitation"
    r"|parent and child|parentage|paternity|adoption of (?:minors|children|a child|adults)|adoption code|adoptions?\s*$"
    r"|termination of parental|parental rights|premarital|prenuptial|family law|family code|domestic partner"
    r"|guardianship of minors|uniform interstate family|uniform child custody", re.I)
FAMILY_HEADING_EXCLUDE = re.compile(
    r"therap|counsel|famil(?:y|ies) (?:farm|services|planning|investment|relief|child care|support services|resource)"
    r"|single-family|department of|license|tax|offenses against"
    # added after the heading review: education, time-share property, prison/DUI/nursing visitation, gift deduction
    r"|student|educat|real estate time|vacation time|drunk|censorship|home visitation|deduction", re.I)


def split_state_code(text, heading=FAMILY_HEADING, exclude=FAMILY_HEADING_EXCLUDE):
    """Yield (chunk_text, in_domain, setting_heading). A heading that matches
    `heading` (and not `exclude`) marks everything up to the next heading of the
    same or a higher level (title > subtitle > chapter > part > article)."""
    heads = list(HEADING_RE.finditer(text))
    stack = {}  # level -> heading text that set the family flag
    for i, m in enumerate(heads):
        lvl = LEVEL[m.group(1).lower()]
        for k in [k for k in stack if k >= lvl]:
            del stack[k]
        h = m.group(0).strip()
        if heading.search(h) and not exclude.search(h):
            stack[lvl] = h
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        chunk = text[m.start():end]
        setter = stack[min(stack)] if stack else None
        yield chunk, setter is not None, setter


TAX_HEADING = re.compile(
    r"taxation|\btax(?:es)?\b|revenue|assessment of property|ad valorem|tax court|tax appeals|tax tribunal", re.I)
TAX_HEADING_EXCLUDE = re.compile(
    r"revenue bonds?|tax increment|tax-exempt bond|lottery|gaming revenue|famil|marriage|divorce", re.I)


def stage_state_family():
    return state_code_domain("family", FAMILY_HEADING, FAMILY_HEADING_EXCLUDE)


def stage_state_tax():
    """State tax/revenue/taxation titles of state_code, same method as state_family."""
    return state_code_domain("tax", TAX_HEADING, TAX_HEADING_EXCLUDE)


def near_dup_chars(items):
    """Sequential MinHash-LSH (threshold 0.8) over (text, n_chars): chars of items
    whose near-duplicate was seen earlier."""
    from datasketch import MinHash, MinHashLSH
    lsh, removed = MinHashLSH(threshold=0.8, num_perm=128), 0
    for i, (text, n) in enumerate(items):
        mh = MinHash(num_perm=128, seed=1, scheme="affine32", hashvalues=minhash(text.lower()))
        if lsh.query(mh):
            removed += n
        else:
            lsh.insert(str(i), mh)
    return removed


def state_code_domain(p, heading, exclude):
    lens, hashes, fam, setters, fam_texts = [], [], [], Counter(), []
    fam_unique, seen = [], set()  # exact-unique domain chunks, for near-dedup
    rng = random.Random(SEED)
    per_record = []
    for split in ["train", "validation"]:
        for ri, r in enumerate(iter_jsonl_xz(fetch(POL, f"data/{split}.state_code.jsonl.xz"))):
            t = r["text"]
            rec_fam = 0
            for chunk, is_fam, setter in split_state_code(t, heading, exclude):
                lens.append(len(chunk))
                hashes.append(norm_hash(chunk.lower()))
                fam.append(is_fam)
                if is_fam:
                    if hashes[-1] not in seen:
                        seen.add(hashes[-1])
                        fam_unique.append((chunk, len(chunk)))
                    rec_fam += len(chunk)
                    setters[setter[:120]] += len(chunk)
                    if rng.random() < 0.02:
                        fam_texts.append(chunk)
            per_record.append({"split": split, "record": ri, "chars": len(t), f"{p}_chars": rec_fam,
                               "head": " ".join(t[:120].split())})
            log(f"state_code {split} record {ri}: {len(t) / 1e6:.0f}M chars, {p} {rec_fam / 1e6:.2f}M")
    lens, hashes, fam = np.array(lens), np.array(hashes, np.uint64), np.array(fam)
    keep_all = dedup_stats(hashes, lens)
    keep_fam = dedup_stats(hashes[fam], lens[fam])
    cpt = chars_per_token(fam_texts)
    uchars = int(lens[fam][keep_fam].sum())
    # The file holds several yearly editions of some states, so near-dedup domain chunks too.
    near_dup = near_dup_chars(fam_unique)
    return {
        f"{p}_near_dup_chars_removed": near_dup,
        f"{p}_near_unique_chars": uchars - near_dup,
        f"{p}_near_unique_tokens": (uchars - near_dup) / cpt["chars_per_token"],
        "method": "split at heading lines (Title/Division/Book/Subtitle/Chapter/Part/Article + number); "
                  f"{p} if the heading or an enclosing higher-level heading matches {p.upper()}_HEADING "
                  f"and not {p.upper()}_HEADING_EXCLUDE; exact dedup per chunk (removes repeated code years)",
        "records": len(per_record), "chunks": len(lens), "chars": int(lens.sum()),
        "unique_chars_all": int(lens[keep_all].sum()),
        f"{p}_chunks": int(fam.sum()), f"{p}_chars": int(lens[fam].sum()),
        f"{p}_unique_chars": uchars, f"{p}_unique_tokens": uchars / cpt["chars_per_token"], "cpt": cpt,
        f"top_{p}_headings_by_chars": setters.most_common(300), "per_record": per_record,
    }


CL_FILES = ["train.courtlisteneropinions.0", "train.courtlisteneropinions.9", "train.courtlisteneropinions.5"]


def cl_all_files():
    return [f"data/{s}.courtlisteneropinions.{i}.jsonl.xz" for s, n in [("train", 16), ("validation", 6)] for i in range(n)]


def cl_opinion_id(record):
    m = re.search(r"(\d+)/?$", record.get("url") or "")
    return int(m.group(1)) if m else -1


def stage_cl():
    total = sum(remote_size(POL, f) for f in cl_all_files())
    done, rates = [], []
    for name in CL_FILES:
        if len(done) == 2:  # third file only if the two match rates disagree by >30% relative
            r = np.array(rates)  # shape (files, domains) strict match rate
            rel = np.abs(r[0] - r[1]) / np.maximum(r.max(0), 1e-9)
            if rel.max() <= 0.30:
                break
            log(f"cl: match rates differ by {rel.max():.0%} relative -> processing a third file")
        try:
            path = fetch(POL, f"data/{name}.jsonl.xz")
        except BudgetExceeded as e:
            log(f"cl: skipping {name}: {e}")
            break
        # doc id = CourtListener opinion id from the url (shows which id range each file covers)
        arr = scan_docs(((cl_opinion_id(r), r["text"]) for r in iter_jsonl_xz(path)), name,
                        minhash_domains=("family", "tax"))
        done.append(name)
        rates.append([classify(arr["counts"], arr["lens"], d, "strict").mean() for d in DOMAINS])
    sample = sum(remote_size(POL, f"data/{n}.jsonl.xz") for n in done)
    return {"files": done, "compressed_sample_bytes": sample, "compressed_total_bytes": total,
            "scale": total / sample, "strict_match_rates": dict(zip(done, rates))}


def merge_npz(names):
    arrs = [np.load(WORK / f"{n}.npz") for n in names]
    return {k: np.concatenate([a[k] for a in arrs]) for k in ["lens", "hashes", "counts"]}


def mean_cpt(names):
    cpts = [load_json(WORK / f"{n}.cpt.json") for n in names]
    return sum(c["sample_chars"] for c in cpts) / sum(c["sample_tokens"] for c in cpts)


class CountingFile(io.RawIOBase):
    """Read-only wrapper around an fsspec file that counts bytes actually read."""

    def __init__(self, f):
        self.f, self.nbytes = f, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence=0):
        return self.f.seek(pos, whence)

    def tell(self):
        return self.f.tell()

    def readinto(self, b):
        data = self.f.read(len(b))
        b[:len(data)] = data
        self.nbytes += len(data)
        return len(data)

    @property
    def size(self):
        return self.f.size


COLD_COURTS = {
    "tax_court": re.compile(r"^united states tax court$|^u\.?s\.? tax court$|^tax court of the united states$", re.I),
    "board_of_tax_appeals": re.compile(r"board of tax appeals", re.I),
    "court_of_federal_claims": re.compile(r"court of federal claims", re.I),
    "court_of_claims": re.compile(r"^(?:united states )?court of claims$", re.I),
}
COLD_TEXT_BUDGET = 1.6e9


def stage_coldcases():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    remote_size(COLD, "README.md")
    files = sorted(f for f in _SIZES[COLD] if f.endswith(".parquet"))
    scan_path = WORK / "coldcases_courts.json"
    scan = load_json(scan_path, {"files": {}, "courts": {}, "bytes": 0})
    for f in files:
        if f in scan["files"]:
            continue
        with fs.open(f"datasets/{COLD}/{f}", "rb", block_size=1 << 16, cache_type="none") as raw:
            cf = CountingFile(raw)
            pf = pq.ParquetFile(io.BufferedReader(cf, 1 << 16))
            rows, courts = [], Counter()
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=["id", "court_full_name", "court_short_name", "date_filed"]).to_pydict()
                for j, (cid, full, short, date) in enumerate(zip(t["id"], t["court_full_name"], t["court_short_name"], t["date_filed"])):
                    courts[full or short or "?"] += 1
                    for key, rx in COLD_COURTS.items():
                        if rx.search(full or "") or rx.search(short or ""):
                            rows.append([rg, j, cid, key, date.year if date else None])
            md = pf.metadata
            scan["files"][f] = {"rows": rows, "row_groups": md.num_row_groups, "opinions_bytes": [
                sum(md.row_group(rg).column(c).total_compressed_size for c in range(md.num_columns)
                    if md.row_group(rg).column(c).path_in_schema.startswith("opinions."))
                for rg in range(md.num_row_groups)]}
            for k, v in courts.items():
                scan["courts"][k] = scan["courts"].get(k, 0) + v
            scan["bytes"] += cf.nbytes
        save_json(scan_path, scan)
        log(f"coldcases court scan {f[:11]}: {len(rows)} matching rows, {cf.nbytes / 1e6:.1f} MB read")
    record_download(f"{COLD}/court+date columns", scan["bytes"])

    # Where are the matching rows?
    by_rg = Counter()
    for f, info in scan["files"].items():
        for rg, *_ in info["rows"]:
            by_rg[(f, rg)] += 1
    total_rgs = sum(info["row_groups"] for info in scan["files"].values())
    n_match = sum(by_rg.values())
    rg_bytes = {(f, rg): scan["files"][f]["opinions_bytes"][rg] for f, rg in by_rg}
    needed_all = sum(rg_bytes.values())
    full = needed_all <= COLD_TEXT_BUDGET and downloaded_bytes() + needed_all <= DOWNLOAD_LIMIT
    # Otherwise take the densest row groups until ~500 matching rows (sample estimate).
    order = sorted(by_rg, key=lambda k: -by_rg[k] / max(rg_bytes[k], 1))
    chosen, got = [], 0
    for k in order:
        if full or got < 500:
            chosen.append(k)
            got += by_rg[k]
    text_path = WORK / "coldcases_text.npz"
    docs = load_json(WORK / "coldcases_docs.json", {"done": [], "docs": []})
    for f, rg in chosen:
        if f"{f}:{rg}" in docs["done"]:
            continue
        if downloaded_bytes() + rg_bytes[(f, rg)] > DOWNLOAD_LIMIT:
            log(f"coldcases: budget reached, stopping text download at {len(docs['done'])} row groups")
            break
        want = {j: (key, year) for rgi, j, _, key, year in scan["files"][f]["rows"] if rgi == rg}
        with fs.open(f"datasets/{COLD}/{f}", "rb", block_size=1 << 16, cache_type="none") as raw:
            cf = CountingFile(raw)
            pf = pq.ParquetFile(io.BufferedReader(cf, 1 << 20))
            ops = pf.read_row_group(rg, columns=["opinions"]).column(0).to_pylist()
        for j, (key, year) in want.items():
            text = "\n\n".join(o["opinion_text"] or "" for o in (ops[j] or []))
            docs["docs"].append({"court": key, "year": year, "text": text})
        docs["done"].append(f"{f}:{rg}")
        record_download(f"{COLD}/{f}#rg{rg}/opinions", cf.nbytes)
        save_json(WORK / "coldcases_docs.json", docs)
        log(f"coldcases text {f[:11]} rg{rg}: {len(want)} rows, {cf.nbytes / 1e6:.0f} MB")

    texts = [d["text"] for d in docs["docs"] if d["text"].strip()]
    cpt = chars_per_token(texts)
    lowers = [t.lower() for t in texts]
    counts = np.array([term_counts(t) for t in lowers], np.uint16).reshape(-1, len(TERM_LIST))
    lens = np.array([len(t) for t in texts])
    hashes = np.array([norm_hash(t) for t in lowers], np.uint64)
    courts_s = [d["court"] for d in docs["docs"] if d["text"].strip()]
    rng = random.Random(SEED)
    idx = rng.sample(range(len(texts)), min(5000, len(texts)))
    np.save(WORK / "coldcases_minhash.npy", np.array([minhash(lowers[i]) for i in idx]).reshape(-1, 128))
    np.save(WORK / "coldcases_minhash_court.npy", np.array([courts_s[i] for i in idx]))

    per_court = {}
    all_rows = [(key, year) for info in scan["files"].values() for _, _, _, key, year in info["rows"]]
    for key in COLD_COURTS:
        m = np.array([c == key for c in courts_s], bool)
        n_rows = sum(1 for k, _ in all_rows if k == key)
        if not m.any():
            per_court[key] = {"rows": n_rows}
            continue
        keep = dedup_stats(hashes[m], lens[m])
        tax_strict = classify(counts[m], lens[m], "tax", "strict")
        tax_loose = classify(counts[m], lens[m], "tax", "loose")
        avg_chars = float(lens[m].mean())
        dup_factor = keep.mean()
        per_court[key] = {
            "rows": n_rows, "sample_docs": int(m.sum()), "avg_chars": avg_chars,
            "exact_dup_rate_in_sample": float(1 - dup_factor),
            "tax_strict_rate": float(tax_strict.mean()), "tax_loose_rate": float(tax_loose.mean()),
            "est_tokens_all": n_rows * avg_chars * dup_factor / cpt["chars_per_token"],
            # Tax Court / BTA are tax by jurisdiction; Claims courts only for keyword-matched cases.
            "est_tax_tokens_strict": n_rows * dup_factor / cpt["chars_per_token"] * float(
                lens[m].mean() if key in ("tax_court", "board_of_tax_appeals") else (lens[m] * tax_strict).mean()),
            "est_tax_tokens_loose": n_rows * dup_factor / cpt["chars_per_token"] * float(
                lens[m].mean() if key in ("tax_court", "board_of_tax_appeals") else (lens[m] * tax_loose).mean()),
        }
    by_year = defaultdict(Counter)
    for key, year in all_rows:
        by_year[key][year] += 1
    return {
        "matched_rows": n_match, "row_groups_total": total_rgs, "row_groups_with_matches": len(by_rg),
        "text_mode": "full" if full else "sample", "text_row_groups_read": len(docs["done"]),
        "text_docs": len(texts), "cpt": cpt, "per_court": per_court,
        "by_year": {k: dict(sorted(v.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))) for k, v in by_year.items()},
        "court_names_seen": dict(Counter(scan["courts"]).most_common()),
        "court_name_matches": sorted({k for k in scan["courts"] if any(rx.search(k) for rx in COLD_COURTS.values())}),
    }


FEDREG_TREASURY = re.compile(r"^\s*DEPARTMENT OF THE TREASURY\b")
FEDREG_IRS = re.compile(r"Internal Revenue Service|\b26 CFR\b")


def stage_fedreg_tax():
    """Treasury documents of Pile of Law federal_register. No agency field: each record
    is one FR document whose text starts with the agency header, so match that."""
    agencies, docs = Counter(), {"irs": [], "treasury_other": []}
    for split in ["train", "validation"]:
        for r in iter_jsonl_xz(fetch(POL, f"data/{split}.federal_register.jsonl.xz")):
            t = r["text"]
            agencies[" ".join(t[:60].split())[:30]] += 1
            if FEDREG_TREASURY.search(t[:300]):
                docs["irs" if FEDREG_IRS.search(t[:600]) else "treasury_other"].append(t)
    out = {"records": sum(agencies.values()), "top_header_prefixes": agencies.most_common(30)}
    for k, texts in docs.items():
        lens = np.array([len(t) for t in texts], np.int64)
        keep = dedup_stats(np.array([norm_hash(t.lower()) for t in texts], np.uint64), lens)
        cpt = chars_per_token(texts)
        out[k] = {"docs": len(texts), "chars": int(lens.sum()), "unique_chars": int(lens[keep].sum()),
                  "unique_tokens": int(lens[keep].sum()) / cpt["chars_per_token"], "cpt": cpt,
                  "heads": [" ".join(t[:200].split()) for t in texts[:10]]}
    return out


def stage_taxrulings_neardup():
    """Within-source near-dedup of taxrulings and IRS memos (templated letters)."""
    out = {}
    for sub in ["taxrulings", "irs_legal_advice_memos"]:
        items, seen = [], set()
        for split in ["train", "validation"]:
            for r in iter_jsonl_xz(fetch(POL, f"data/{split}.{sub}.jsonl.xz")):
                h = norm_hash(r["text"].lower())
                if h not in seen:
                    seen.add(h)
                    items.append((r["text"], len(r["text"])))
        uchars = sum(n for _, n in items)
        removed = near_dup_chars(items)
        cpt = load_json(OUT / "pol_tax.json")[sub]["cpt"]["chars_per_token"]
        out[sub] = {"exact_unique_docs": len(items), "exact_unique_chars": uchars, "near_dup_chars_removed": removed,
                    "near_dup_share": removed / max(uchars, 1), "near_unique_tokens": (uchars - removed) / cpt}
        log(f"taxrulings_neardup {sub}: {out[sub]['near_unique_tokens'] / 1e6:.1f}M near-unique tokens")
    return out


COLD_STATE_TAX = re.compile(r"^(?:oregon|new jersey|indiana|arizona) tax court$|^massachusetts appellate tax board$"
                            r"|^montana tax appeal board$|board of tax appeals of ohio", re.I)
COLD_STATE_TEXT_BUDGET = 1.5e9


def stage_coldcases_state_tax():
    """State tax courts in cold-cases. coldcases_courts.json kept row positions only for the
    federal courts, so rescan the court column (~86 MB with date) into a separate file."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    remote_size(COLD, "README.md")
    files = sorted(f for f in _SIZES[COLD] if f.endswith(".parquet"))
    scan_path = WORK / "coldcases_state_tax_courts.json"
    scan = load_json(scan_path, {"files": {}, "bytes": 0})
    for f in files:
        if f in scan["files"]:
            continue
        with fs.open(f"datasets/{COLD}/{f}", "rb", block_size=1 << 16, cache_type="none") as raw:
            cf = CountingFile(raw)
            pf = pq.ParquetFile(io.BufferedReader(cf, 1 << 16))
            rows = []
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=["court_full_name", "date_filed"]).to_pydict()
                for j, (full, date) in enumerate(zip(t["court_full_name"], t["date_filed"])):
                    if COLD_STATE_TAX.search(full or ""):
                        rows.append([rg, j, full, date.year if date else None])
            md = pf.metadata
            scan["files"][f] = {"rows": rows, "opinions_bytes": [
                sum(md.row_group(rg).column(c).total_compressed_size for c in range(md.num_columns)
                    if md.row_group(rg).column(c).path_in_schema.startswith("opinions."))
                for rg in range(md.num_row_groups)]}
            scan["bytes"] += cf.nbytes
        save_json(scan_path, scan)
        log(f"coldcases state-tax scan {f[:11]}: {len(rows)} rows, {cf.nbytes / 1e6:.1f} MB read")
    record_download(f"{COLD}/court+date columns (state tax rescan)", scan["bytes"])

    by_rg = Counter((f, r[0]) for f, info in scan["files"].items() for r in info["rows"])
    rg_bytes = {k: scan["files"][k[0]]["opinions_bytes"][k[1]] for k in by_rg}
    full = sum(rg_bytes.values()) <= COLD_STATE_TEXT_BUDGET
    # Otherwise the densest row groups that fit in the text budget.
    chosen, spent = [], 0
    for k in sorted(by_rg, key=lambda k: -by_rg[k] / max(rg_bytes[k], 1)):
        if spent + rg_bytes[k] <= COLD_STATE_TEXT_BUDGET:
            chosen.append(k)
            spent += rg_bytes[k]
    docs_path = WORK / "coldcases_state_tax_docs.json"
    docs = load_json(docs_path, {"done": [], "docs": []})
    for f, rg in chosen:
        if f"{f}:{rg}" in docs["done"]:
            continue
        if downloaded_bytes() + rg_bytes[(f, rg)] > DOWNLOAD_LIMIT:
            log("coldcases_state_tax: download budget reached")
            break
        want = {r[1]: (r[2], r[3]) for r in scan["files"][f]["rows"] if r[0] == rg}
        with fs.open(f"datasets/{COLD}/{f}", "rb", block_size=1 << 16, cache_type="none") as raw:
            cf = CountingFile(raw)
            pf = pq.ParquetFile(io.BufferedReader(cf, 1 << 20))
            ops = pf.read_row_group(rg, columns=["opinions"]).column(0).to_pylist()
        for j, (court, year) in want.items():
            docs["docs"].append({"court": court, "year": year,
                                 "text": "\n\n".join(o["opinion_text"] or "" for o in (ops[j] or []))})
        docs["done"].append(f"{f}:{rg}")
        record_download(f"{COLD}/{f}#rg{rg}/opinions", cf.nbytes)
        save_json(docs_path, docs)
        log(f"coldcases state-tax text {f[:11]} rg{rg}: {len(want)} rows, {cf.nbytes / 1e6:.0f} MB")

    d = [x for x in docs["docs"] if x["text"].strip()]
    lowers = [x["text"].lower() for x in d]
    lens = np.array([len(t) for t in lowers])
    hashes = np.array([norm_hash(t) for t in lowers], np.uint64)
    counts = np.array([term_counts(t) for t in lowers], np.uint16).reshape(-1, len(TERM_LIST))
    cpt = chars_per_token([x["text"] for x in d])
    keep = dedup_stats(hashes, lens)
    # Cross-source near-dups against the CourtListener sample index, as in stage_neardup.
    cl = load_json(OUT / "cl.json")
    cl_mh = np.concatenate([np.load(WORK / f"{n}.npz")["mh_vals"] for n in cl["files"]])
    mh = np.array([minhash(lowers[i]) for i in np.flatnonzero(keep)]).reshape(-1, 128)
    r_cl, hits = lsh_dup_rate(mh, cl_mh)
    cl_dup = min(1.0, r_cl * cl["scale"])
    all_rows = [r for info in scan["files"].values() for r in info["rows"]]
    by_decade = defaultdict(Counter)
    for _, _, court, year in all_rows:
        by_decade[court][str(year // 10 * 10) if year else "?"] += 1
    sample_courts = np.array([x["court"] for x in d])
    per_court = {}
    for court, n in Counter(r[2] for r in all_rows).items():
        m = sample_courts == court
        per_court[court] = {"rows": n, "sample_docs": int(m.sum())}
        if m.any():
            per_court[court].update(
                avg_chars=float(lens[m].mean()),
                tax_strict_rate=float(classify(counts[m], lens[m], "tax", "strict").mean()),
                est_tokens=n * float(lens[m].mean()) * float(keep[m].mean()) / cpt["chars_per_token"])
    est = sum(v.get("est_tokens", 0) for v in per_court.values())
    return {"rows": len(all_rows), "row_groups_with_rows": len(by_rg), "text_mode": "full" if full else "sample",
            "text_row_groups_read": len(docs["done"]), "text_docs": len(d), "sample_chars": int(lens.sum()),
            "exact_dup_rate": float(1 - keep.mean()), "cpt": cpt,
            "vs_cl": {"query_docs": int(len(mh)), "hits": hits, "observed_rate": r_cl, "scaled_dup_rate": cl_dup},
            "per_court": per_court, "est_tokens_all": est, "est_tokens_new_vs_cl": est * (1 - cl_dup),
            "by_court_decade": {k: dict(sorted(v.items())) for k, v in by_decade.items()}}


HFL_FILES = ["data/us-00000-of-00020.arrow", "data/us-00011-of-00020.arrow"]


def stage_hfforlegal():
    import pyarrow as pa
    states = Counter()

    def docs():
        for f in HFL_FILES:
            path = fetch(HFL, f)
            with pa.memory_map(path) as src:
                reader = pa.ipc.open_stream(src)
                for batch in reader:
                    d = batch.to_pydict()
                    for i in range(batch.num_rows):
                        states[d["state"][i]] += 1
                        yield f"{d['state'][i]}|{d['id'][i]}", d["document"][i] or ""

    arr = scan_docs(docs(), "hfforlegal_sample", minhash_domains=("family", "tax"))
    if not states:  # npz cached: recount states cheaply
        for f in HFL_FILES:
            with pa.memory_map(fetch(HFL, f)) as src:
                for batch in pa.ipc.open_stream(src):
                    states.update(batch.column("state").to_pylist())
    total = sum(remote_size(HFL, f"data/us-{i:05d}-of-00020.arrow") for i in range(20))
    sample = sum(remote_size(HFL, f) for f in HFL_FILES)
    return {"files": HFL_FILES, "sample_bytes": sample, "total_bytes": total, "scale": total / sample,
            "states_in_sample": dict(states.most_common())}


def lsh_dup_rate(query, index, threshold=0.8):
    """Fraction of `query` signatures with a near-duplicate (est. Jaccard >= threshold) in `index`."""
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, hv in enumerate(index):
        lsh.insert(str(i), MinHash(num_perm=128, seed=1, scheme="affine32", hashvalues=hv))
    hits = 0
    for hv in query:
        if lsh.query(MinHash(num_perm=128, seed=1, scheme="affine32", hashvalues=hv)):
            hits += 1
    return hits / max(len(query), 1), hits


def stage_neardup():
    cl = load_json(OUT / "cl.json")
    hfl = load_json(OUT / "hfforlegal.json")
    rng = np.random.default_rng(SEED)
    cl_mh = np.concatenate([np.load(WORK / f"{n}.npz")["mh_vals"] for n in cl["files"]])
    cl_frac = 1 / cl["scale"]
    res = {"cl_sample_fraction": cl_frac, "cl_minhash_docs": int(len(cl_mh)),
           "note": "observed rate is against a CourtListener SAMPLE; 'scaled' divides by the sample fraction "
                   "(assumes duplicates are spread evenly over CourtListener files), capped at 100%"}
    h = np.load(WORK / "hfforlegal_sample.npz")["mh_vals"]
    h = h[rng.choice(len(h), min(5000, len(h)), replace=False)]
    rate, hits = lsh_dup_rate(h, cl_mh)
    res["hfforlegal_vs_cl"] = {"query_docs": int(len(h)), "hits": hits, "observed_rate": rate,
                               "scaled_dup_rate": min(1.0, rate / cl_frac)}
    # HFforLegal internal near-dup rate (sanity: should be low)
    rate_self, _ = lsh_dup_rate(h[:2500], h[2500:])
    res["hfforlegal_internal_halves"] = rate_self
    if (WORK / "coldcases_minhash.npy").exists():
        c = np.load(WORK / "coldcases_minhash.npy")
        courts = np.load(WORK / "coldcases_minhash_court.npy")
        tr = np.concatenate([np.load(WORK / f"pol_{s}_minhash.npy") for s in ["taxrulings", "irs_legal_advice_memos"]])
        r_tr, hits_tr = lsh_dup_rate(c, tr)
        r_cl, hits_cl = lsh_dup_rate(c, cl_mh)
        res["coldcases_vs_taxrulings_irs"] = {"query_docs": int(len(c)), "hits": hits_tr, "observed_rate": r_tr,
                                              "index_docs": int(len(tr)), "index_is_sample_of_5k_per_subset": True}
        res["coldcases_vs_cl"] = {"query_docs": int(len(c)), "hits": hits_cl, "observed_rate": r_cl,
                                  "scaled_dup_rate": min(1.0, r_cl / cl_frac)}
        per = {}
        for key in set(courts.tolist()):
            m = courts == key
            r, k = lsh_dup_rate(c[m], cl_mh)
            per[key] = {"query_docs": int(m.sum()), "hits": k, "observed_rate": r, "scaled_dup_rate": min(1.0, r / cl_frac)}
        res["coldcases_vs_cl_per_court"] = per
    return res


def stage_summary():
    cl = load_json(OUT / "cl.json")
    hfl = load_json(OUT / "hfforlegal.json")
    out = {"thresholds": THRESHOLDS}
    out["cl"] = domain_summary(merge_npz(cl["files"]), mean_cpt(cl["files"]), cl["scale"])
    out["cl"]["per_file"] = {n: domain_summary(merge_npz([n]), mean_cpt([n]), 1.0) for n in cl["files"]}
    out["hfforlegal"] = domain_summary(merge_npz(["hfforlegal_sample"]), mean_cpt(["hfforlegal_sample"]), hfl["scale"])
    return out


def stage_spot():
    """Write precision-check samples: 25 strict matches and 15 near misses per domain."""
    cl = load_json(OUT / "cl.json")
    rng = random.Random(SEED)
    snippets = []
    for n in cl["files"] + ["hfforlegal_sample"]:
        with open(WORK / f"{n}.snippets.jsonl", encoding="utf-8") as fh:
            snippets += [dict(json.loads(l), src=n) for l in fh]
    lines = []
    for d in DOMAINS:
        terms = set(TERMS[d])
        def stats(s):
            c = {t: v for t, v in s["terms"].items() if t in terms}
            return len(c), sum(c.values()) / max(s["len"], 1) * 1e4, c
        strict = [s for s in snippets if stats(s)[0] >= THRESHOLDS["strict"][0] and stats(s)[1] >= THRESHOLDS["strict"][1]]
        miss = [s for s in snippets if s not in strict and stats(s)[0] >= THRESHOLDS["loose"][0] and stats(s)[1] >= THRESHOLDS["loose"][1]]
        for label, pool, k in [("STRICT", strict, 25), ("NEAR-MISS (loose only)", miss, 15)]:
            lines.append(f"\n# {d.upper()} — {label} ({len(pool)} in snippet pool)\n")
            for s in rng.sample(pool, min(k, len(pool))):
                nd, dens, c = stats(s)
                lines.append(f"## {s['src']} #{s['i']} len={s['len']} distinct={nd} density={dens:.2f}\n"
                             f"terms={c}\nother={ {t: v for t, v in s['terms'].items() if t not in terms} }\n"
                             f"> {' '.join(s['head'].split())}\n")
    (OUT / "spotcheck.md").write_text("\n".join(lines), encoding="utf-8")
    return {"written": str(OUT / "spotcheck.md")}


STAGES = {"pol_tax": stage_pol_tax, "title26": stage_title26, "state_family": stage_state_family,
          "cl": stage_cl, "coldcases": stage_coldcases, "hfforlegal": stage_hfforlegal,
          "neardup": stage_neardup, "summary": stage_summary, "spot": stage_spot,
          "state_tax": stage_state_tax, "fedreg_tax": stage_fedreg_tax,
          "coldcases_state_tax": stage_coldcases_state_tax, "taxrulings_neardup": stage_taxrulings_neardup}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), action="append")
    ap.add_argument("--force", action="store_true", help="rerun the stage even if its result file exists")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    for name in args.stage or list(STAGES):
        path = OUT / f"{name}.json"
        if path.exists() and not args.force:
            log(f"{name}: already done, skipping")
            continue
        log(f"{name}: start")
        try:
            save_json(path, STAGES[name]())
            log(f"{name}: written {path}")
        except BudgetExceeded as e:
            log(f"{name}: SKIPPED, {e}")
        except Exception as e:  # keep going with the other stages; the error is in the log
            log(f"{name}: FAILED {type(e).__name__}: {e}")
            if args.stage:
                raise


if __name__ == "__main__":
    main()
