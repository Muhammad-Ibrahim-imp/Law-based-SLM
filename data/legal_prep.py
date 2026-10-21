"""Legal pretraining corpora: FAMILY LAW (SLM 1) and CRIMINAL LAW (SLM 2).

Output categories (same format and manifest schema as general_train /
general_validation, read by ``ShardedTokenDataset`` and checked by
``general_prep.validate_shards``):

    family_law_train, family_law_validation, criminal_law_train, criminal_law_validation

Each is ``dataset/<category>/shard_XXXXX.bin``: raw uint16 token IDs, EOS after
every document, 2**28 tokens per shard (last shard partial).

Sources (Hugging Face raw files, never loading scripts; Pile of Law train AND
validation files are both used):

    courtlistener  Pile of Law courtlisteneropinions (22 files): family "loose", criminal "strict"
    hfforlegal     HFforLegal/case-law split us (20 arrow files): same rules; near-duplicates
                   of CourtListener lose to the CourtListener copy
    state_code     Pile of Law state_code: family / criminal titles by heading
                   (data.domains.split_state_code), grouped per heading, <= 60k chars per doc
    uscode         Pile of Law uscode: Title 18 + Title 18 Appendix (criminal), grouped sections
    fre            Pile of Law fre: Federal Rules of Evidence (criminal)

Stages (each resumable; intermediates under ``work``):

    extract   per input file, in parallel: classify, clean, quality-filter
              -> work/candidates/<job>.jsonl.gz + <job>.stats.json
    evalsets  13-gram hashes of CaseHOLD validation+test contexts and LegalBench test
    features  per candidate file, in parallel: exact key, MinHash, decontamination hits
    dedup     global clusters (exact + near, 0.8 Jaccard); per domain keep one document per
              cluster (CourtListener first, then longest), drop contaminated ones; split
              0.5% validation by a stable hash of the cluster, so a cluster never straddles
              train/validation and a document has the same split in both domains
    tokenize  per candidate file, in parallel: SmolLM2 tokens + EOS for the kept documents
    write     per category: seeded shuffle, criminal train capped at 4.7B tokens (all statutes,
              then a seeded random sample of case law), ShardWriter (atomic), manifest entry
              (locked update)
    validate  validate_shards, ShardedTokenDataset checks, decoded sample windows
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import lzma
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data.cleaner import clean_legal_text, quality_issue
from data.domains import (
    CRIMINAL_HEADING, CRIMINAL_HEADING_EXCLUDE, FAMILY_HEADING, FAMILY_HEADING_EXCLUDE, SECTION_RE,
    classify_counts, split_state_code, term_counts,
)
from data.shard_writer import ShardWriter, StreamSpec, update_manifest, utc_now_iso

POL = "pile-of-law/pile-of-law"
HFL = "HFforLegal/case-law"
DOMAINS = ("family", "criminal")
CASE_PROFILE = {"family": "loose", "criminal": "strict"}
SEED = 42
VALIDATION_RATIO = 0.005
SHARD_TOKENS = 2 ** 28          # matches general's shard_tokens_target in the local manifest
SEQUENCE_LENGTH = 8192          # 200M training context (metadata only; shards are flat)
CRIMINAL_TRAIN_CAP = 4_700_000_000
MIN_WORDS = 150
STATUTE_DOC_CHARS = 60_000      # state-code / US Code groups are cut into docs of at most this
DOWNLOAD_LIMIT = 35e9
SOURCE_RANK = {"courtlistener": 0, "hfforlegal": 1}   # cluster representative: lower rank first
LICENCES = {"courtlistener": "Pile of Law, CC BY-NC-SA 4.0", "state_code": "Pile of Law, CC BY-NC-SA 4.0",
            "uscode": "Pile of Law, CC BY-NC-SA 4.0", "fre": "Pile of Law, CC BY-NC-SA 4.0",
            "hfforlegal": "HFforLegal/case-law, CC-BY-4.0"}


# ---------------------------------------------------------------- jobs and paths
def all_jobs(pilot: bool = False) -> list[dict]:
    """One job per input file. ``pilot`` keeps one CourtListener + one HFforLegal file."""
    jobs = [{"job": f"courtlistener.{s}.{i}", "source": "courtlistener", "repo": POL,
             "file": f"data/{s}.courtlisteneropinions.{i}.jsonl.xz"}
            for s, n in (("train", 16), ("validation", 6)) for i in range(n)]
    jobs += [{"job": f"hfforlegal.{i:02d}", "source": "hfforlegal", "repo": HFL,
              "file": f"data/us-{i:05d}-of-00020.arrow"} for i in range(20)]
    for sub in ("state_code", "uscode", "fre"):
        jobs += [{"job": f"{sub}.{s}", "source": sub, "repo": POL, "file": f"data/{s}.{sub}.jsonl.xz"}
                 for s in ("train", "validation")]
    if pilot:
        keep = {"courtlistener.train.0", "hfforlegal.00"}
        jobs = [j for j in jobs if j["job"] in keep or j["source"] not in ("courtlistener", "hfforlegal")]
    return jobs


class Paths:
    def __init__(self, work: str | Path):
        self.work = Path(work)
        self.cand = self.work / "candidates"
        self.feat = self.work / "features"
        self.tok = self.work / "tokens"
        self.eval = self.work / "evalsets"
        for p in (self.cand, self.feat, self.tok, self.eval):
            p.mkdir(parents=True, exist_ok=True)


LOG_PATH = Path("results/legal_prep/progress.log")


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    for attempt in range(10):   # Windows refuses os.replace while another process has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def _read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


# ---------------------------------------------------------------- downloads
LEDGER = Path("results/legal_prep/downloads.json")


def ensure_downloaded(job: dict) -> str:
    """Path of the job's file in the HF cache, downloading it (within the budget) if needed."""
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    cached = try_to_load_from_cache(job["repo"], job["file"], repo_type="dataset")
    if isinstance(cached, str):
        return cached
    ledger = _read_json(LEDGER, {})
    size = next(s.size for s in HfApi().dataset_info(job["repo"], files_metadata=True).siblings
                if s.rfilename == job["file"])
    used = sum(v["bytes"] for v in ledger.values())
    if used + size > DOWNLOAD_LIMIT:
        raise RuntimeError(f"download budget: {job['file']} ({size / 1e9:.2f} GB) would exceed "
                           f"{DOWNLOAD_LIMIT / 1e9:.0f} GB (used {used / 1e9:.2f} GB)")
    log(f"download {job['repo']}/{job['file']} ({size / 1e9:.2f} GB; {used / 1e9:.2f} GB so far)")
    for attempt in range(1, 6):   # the Hub resets long unauthenticated downloads now and then
        try:
            path = hf_hub_download(job["repo"], job["file"], repo_type="dataset")
            break
        except Exception as e:
            if attempt == 5:
                raise
            log(f"download {job['file']}: attempt {attempt} failed ({type(e).__name__}: {str(e)[:120]}); "
                f"retrying in {60 * attempt}s")
            time.sleep(60 * attempt)
    ledger[f"{job['repo']}/{job['file']}"] = {"bytes": size, "path": os.path.realpath(path)}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    _write_json(LEDGER, ledger)
    return path


def downloaded_gb() -> float:
    return sum(v["bytes"] for v in _read_json(LEDGER, {}).values()) / 1e9


# ---------------------------------------------------------------- extract
def _iter_jsonl_xz(path):
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            yield i, json.loads(line)


def _iter_case_law(job: dict, path: str):
    """(doc_id, raw_text) for CourtListener / HFforLegal."""
    if job["source"] == "courtlistener":
        for i, r in _iter_jsonl_xz(path):
            m = re.search(r"(\d+)/?$", r.get("url") or "")
            yield f"cl:{m.group(1) if m else job['job'] + ':' + str(i)}", r.get("text") or ""
    else:
        import pyarrow as pa
        with pa.memory_map(path) as src:
            for batch in pa.ipc.open_stream(src):
                d = batch.to_pydict()
                for k in range(batch.num_rows):
                    yield f"hfl:{d['state'][k]}:{d['id'][k]}", d["document"][k] or ""


def _group(chunks, limit=STATUTE_DOC_CHARS):
    """Concatenate consecutive chunks into documents of at most ``limit`` chars."""
    buf, n = [], 0
    for c in chunks:
        if buf and n + len(c) > limit:
            yield "".join(buf)
            buf, n = [], 0
        buf.append(c)
        n += len(c)
    if buf:
        yield "".join(buf)


def _iter_statutes(job: dict, path: str):
    """(doc_id, raw_text, domains) for state_code / uscode / fre."""
    split = job["file"].split("/")[1].split(".")[0]
    src = job["source"]
    for ri, r in _iter_jsonl_xz(path):
        text = r.get("text") or ""
        if src == "fre":
            yield f"fre:{split}:{ri}", text, ("criminal",)
        elif src == "uscode":
            if re.search(r"^\s*Title 18\b", text[:2000], re.M):   # Title 18 and Title 18 Appendix
                for k, doc in enumerate(_group(c for c in SECTION_RE.split(text) if c.strip())):
                    yield f"uscode:{split}:{ri}:{k}", doc, ("criminal",)
        else:  # state_code: one record = one state's code (some states in several yearly editions)
            for dom, inc, exc in (("family", FAMILY_HEADING, FAMILY_HEADING_EXCLUDE),
                                  ("criminal", CRIMINAL_HEADING, CRIMINAL_HEADING_EXCLUDE)):
                k, group, setter_now = 0, [], None
                for chunk, in_dom, setter in split_state_code(text, inc, exc):
                    if setter != setter_now and group:
                        for doc in _group(group):
                            yield f"state_code:{split}:{ri}:{dom}:{k}", doc, (dom,)
                            k += 1
                        group = []
                    setter_now = setter
                    if in_dom:
                        group.append(chunk)
                for doc in _group(group):
                    yield f"state_code:{split}:{ri}:{dom}:{k}", doc, (dom,)
                    k += 1


def extract_job(job: dict, path: str, cand_dir: str) -> dict:
    """Classify, clean and filter one input file. Writes <job>.jsonl.gz and <job>.stats.json."""
    out = Path(cand_dir) / f"{job['job']}.jsonl.gz"
    stats_path = Path(cand_dir) / f"{job['job']}.stats.json"
    if stats_path.exists():
        return _read_json(stats_path)
    t0 = time.time()
    st = {"job": job["job"], "source": job["source"], "docs_read": 0, "chars_read": 0,
          **{d: {"matched": 0, "matched_chars": 0, "kept": 0, "kept_chars": 0, "dropped": Counter()}
             for d in DOMAINS}}
    samples = []   # a few before/after pairs for the report
    rng = random.Random(f"{SEED}-{job['job']}")
    tmp = out.with_suffix(".gz.tmp")
    statute = job["source"] not in ("courtlistener", "hfforlegal")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3) as fh:
        if statute:
            items = ((i, t, doms) for i, t, doms in _iter_statutes(job, path))
        else:
            items = ((i, t, None) for i, t in _iter_case_law(job, path))
        for doc_id, raw, doms in items:
            st["docs_read"] += 1
            st["chars_read"] += len(raw)
            if doms is None:
                c = term_counts(raw.lower())[None]
                lens = np.array([len(raw)])
                doms = tuple(d for d in DOMAINS if classify_counts(c, lens, d, CASE_PROFILE[d])[0])
            if not doms:
                continue
            for d in doms:
                st[d]["matched"] += 1
                st[d]["matched_chars"] += len(raw)
            text = clean_legal_text(raw, layout=not statute)
            why = quality_issue(text, MIN_WORDS)
            if why:
                for d in doms:
                    st[d]["dropped"][why] += 1
                continue
            for d in doms:
                st[d]["kept"] += 1
                st[d]["kept_chars"] += len(text)
            if len(samples) < 3 and rng.random() < 0.002 and len(raw) < 20000:
                samples.append({"id": doc_id, "raw": raw[:1500], "clean": text[:1500]})
            fh.write(json.dumps({"id": doc_id, "source": job["source"], "kind": "statute" if statute else "case",
                                 "family": "family" in doms, "criminal": "criminal" in doms,
                                 "chars": len(text), "text": text}) + "\n")
    os.replace(tmp, out)
    for d in DOMAINS:
        st[d]["dropped"] = dict(st[d]["dropped"])
    st["seconds"] = round(time.time() - t0)
    st["samples"] = samples
    _write_json(stats_path, st)
    return st


def _iter_candidates(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# ---------------------------------------------------------------- evaluation 13-grams
def build_evalsets(paths: Paths) -> dict:
    """13-gram hashes of evaluation text: CaseHOLD validation+test contexts, LegalBench test."""
    from data.dedup import ngram_hashes
    from data.sft_sources import _read_casehold_parquet
    info = _read_json(paths.eval / "info.json")
    if info:
        return info
    info = {}
    grams = []
    n = 0
    for split in ("validation", "test"):
        for row in _read_casehold_parquet(split):
            ctx = (row.get("citing_prompt") or "").replace("<HOLDING>", " ")
            grams.append(ngram_hashes(ctx))
            n += 1
    arr = np.unique(np.concatenate(grams)) if grams else np.zeros(0, np.uint64)
    np.save(paths.eval / "casehold.npy", arr)
    info["casehold"] = {"examples": n, "ngrams": int(len(arr)), "splits": ["validation", "test"],
                        "text": "citing_prompt (opinion excerpt), <HOLDING> removed"}
    try:
        import csv
        from huggingface_hub import HfApi, hf_hub_download
        files = [f for f in HfApi().list_repo_files("nguha/legalbench", repo_type="dataset")
                 if f.startswith("data/") and f.endswith("/test.tsv")]
        grams, n = [], 0
        csv.field_size_limit(1 << 30)
        for f in files:
            p = hf_hub_download("nguha/legalbench", f, repo_type="dataset")
            with open(p, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    text = " ".join(v for k, v in row.items() if isinstance(v, str) and k not in ("index", "idx"))
                    grams.append(ngram_hashes(text))
                    n += 1
        arr = np.unique(np.concatenate(grams))
        np.save(paths.eval / "legalbench.npy", arr)
        info["legalbench"] = {"tasks": len(files), "examples": n, "ngrams": int(len(arr)), "split": "test"}
    except Exception as e:  # recorded, build continues without it
        info["legalbench"] = {"error": f"{type(e).__name__}: {e}"}
    _write_json(paths.eval / "info.json", info)
    return info


def _load_evalsets(paths: Paths) -> dict[str, np.ndarray]:
    return {name: np.load(paths.eval / f"{name}.npy") for name in ("casehold", "legalbench")
            if (paths.eval / f"{name}.npy").exists()}


# ---------------------------------------------------------------- features
def features_job(job_name: str, work: str) -> int:
    from data.dedup import minhash_signature, ngram_hashes, normalize_for_hashing
    paths = Paths(work)
    out = paths.feat / f"{job_name}.npz"
    if out.exists():
        return 0
    evals = _load_evalsets(paths)
    ids, src, kind, fam, crim, chars, exact, sigs = [], [], [], [], [], [], [], []
    hits = {k: [] for k in evals}   # per document: (positions, hashes) of evaluation 13-grams it contains
    for rec in _iter_candidates(paths.cand / f"{job_name}.jsonl.gz"):
        norm = normalize_for_hashing(rec["text"])
        ids.append(rec["id"])
        src.append(rec["source"])
        kind.append(rec["kind"])
        fam.append(rec["family"])
        crim.append(rec["criminal"])
        chars.append(rec["chars"])
        exact.append(int.from_bytes(hashlib.blake2b(norm.encode(), digest_size=8).digest(), "little"))
        sigs.append(minhash_signature(norm, normalized=True))
        g = ngram_hashes(norm, 13, normalized=True, unique=False)   # in text order
        for k, ev in evals.items():
            if not len(ev) or not len(g):
                hits[k].append((np.zeros(0, np.int32), np.zeros(0, np.uint64)))
                continue
            pos = np.searchsorted(ev, g)
            pos[pos == len(ev)] = 0
            where = np.flatnonzero(ev[pos] == g)
            hits[k].append((where.astype(np.int32), g[where]))
    tmp = out.with_suffix(".tmp.npz")
    matched = {}
    for k, v in hits.items():   # ragged -> flat arrays + offsets
        matched[f"match_{k}"] = np.concatenate([h for _, h in v]) if v else np.zeros(0, np.uint64)
        matched[f"match_{k}_pos"] = np.concatenate([p for p, _ in v]) if v else np.zeros(0, np.int32)
        matched[f"match_{k}_off"] = np.concatenate([[0], np.cumsum([len(h) for _, h in v])]).astype(np.int64)
    np.savez(tmp, ids=np.array(ids, dtype=object).astype(str), source=np.array(src), kind=np.array(kind),
             family=np.array(fam, bool), criminal=np.array(crim, bool), chars=np.array(chars, np.int64),
             exact=np.array(exact, np.uint64), sigs=np.array(sigs, np.uint32).reshape(-1, 128), **matched)
    os.replace(tmp, out)
    return len(ids)


# ---------------------------------------------------------------- dedup + split
def split_of(cluster_key: str) -> int:
    """0 = train, 1 = validation (0.5%), stable across runs and domains."""
    d = hashlib.sha256(f"{SEED}\0{cluster_key}".encode()).digest()
    return int(int.from_bytes(d[:8], "big") < VALIDATION_RATIO * (1 << 64))


DECONTAM_MAX_DF = 20     # eval 13-grams occurring more often than this in the candidates are formulae
DECONTAM_MIN_RUN = 50    # contaminated: >= 50 consecutive rare matching 13-grams (~62 words of eval text)


def dedup_and_split(paths: Paths, jobs: list[str], decontam_min_run: int = DECONTAM_MIN_RUN,
                    decontam_max_df: int = DECONTAM_MAX_DF) -> dict:
    """Global near-dup clusters, then per-domain selection, decontamination and split.

    Writes features/<job>.select.npz with per-document ``family`` / ``criminal``
    arrays: -1 dropped, 0 train, 1 validation.
    """
    from data.dedup import near_duplicate_clusters
    done = paths.work / "dedup.json"
    if done.exists():
        return _read_json(done)
    parts = [np.load(paths.feat / f"{j}.npz", allow_pickle=False) for j in jobs]
    sizes = [len(p["ids"]) for p in parts]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "source", "kind", "family", "criminal",
                                                               "chars", "exact")}
    sigs = np.concatenate([p["sigs"] for p in parts])
    n = len(cat["ids"])
    # Decontamination. Shared 13-grams are mostly legal formulae and citations quoted in thousands of
    # opinions ("any rational trier of fact could have found ...", "Miranda v. Arizona, 384 U.S. 436"):
    # those occurring more than ``decontam_max_df`` times in the candidates are ignored, and a document
    # counts as contaminated only if it holds a run of ``decontam_min_run`` consecutive rare matches,
    # i.e. it contains a passage of evaluation text, not a quotation.
    from data.dedup import longest_runs
    benches = sorted({k[6:] for p in parts for k in p.files
                      if k.startswith("match_") and not k.endswith(("_off", "_pos"))})
    hits, decon_info = {}, {}
    for b in benches:
        flat = np.concatenate([p[f"match_{b}"] for p in parts])
        pos = np.concatenate([p[f"match_{b}_pos"] for p in parts])
        per_doc = np.concatenate([np.diff(p[f"match_{b}_off"]) for p in parts])
        uniq, occ = np.unique(flat, return_counts=True)       # occurrences (~document frequency)
        rare = ~np.isin(flat, uniq[occ > decontam_max_df])
        rare_per_doc = np.bincount(np.repeat(np.arange(n), per_doc)[rare], minlength=n)
        offsets = np.concatenate([[0], np.cumsum(rare_per_doc)])
        hits[b] = longest_runs(pos[rare], offsets)
        decon_info[b] = {"matched_eval_ngrams": int(len(uniq)), "common_ngrams_ignored": int((occ > decontam_max_df).sum()),
                         "docs_with_any_match": int((per_doc > 0).sum()),
                         "docs_with_rare_match": int((rare_per_doc > 0).sum()),
                         **{f"docs_with_run_>={r}": int((hits[b] >= r).sum()) for r in (10, 20, 50, 100)}}
    log(f"dedup: {n} candidate documents, clustering")
    roots, cstats = near_duplicate_clusters(sigs, 0.8, exact_keys=cat["exact"])
    rank = np.array([SOURCE_RANK.get(s, 0) for s in cat["source"]])
    split = np.array([split_of(cat["ids"][r]) for r in roots], np.int8)
    stats = {"candidates": n, "clustering": cstats, "decontam_min_run": decontam_min_run,
             "decontam_max_df": decontam_max_df, "decontamination": decon_info}
    select = {}
    for d in DOMAINS:
        idx = np.flatnonzero(cat[d])
        sel = np.full(n, -1, np.int8)
        # stage counts in report order: exact dedup, then near dedup, then decontamination
        exact_keep = {}
        for i in idx[np.lexsort((-cat["chars"][idx], rank[idx]))]:
            exact_keep.setdefault(int(cat["exact"][i]), int(i))
        after_exact = np.array(sorted(exact_keep.values()), dtype=np.int64)
        rep = {}
        for i in after_exact[np.lexsort((-cat["chars"][after_exact], rank[after_exact]))]:
            rep.setdefault(int(roots[i]), int(i))
        after_near = np.array(sorted(rep.values()), dtype=np.int64)
        contaminated = np.zeros(n, bool)
        for k in benches:
            contaminated |= hits[k] >= decontam_min_run
        final = after_near[~contaminated[after_near]]
        sel[final] = split[final]
        select[d] = sel
        per_src = defaultdict(lambda: Counter())
        for name, arr in (("candidates", idx), ("after_exact", after_exact), ("after_near", after_near),
                          ("final", final)):
            for s, k in zip(*np.unique(cat["source"][arr], return_counts=True)):
                per_src[s][name] = int(k)
            for s in np.unique(cat["source"][arr]):
                m = cat["source"][arr] == s
                per_src[s][name + "_chars"] = int(cat["chars"][arr][m].sum())
        stats[d] = {
            "candidates": int(len(idx)), "exact_duplicates_removed": int(len(idx) - len(after_exact)),
            "near_duplicates_removed": int(len(after_exact) - len(after_near)),
            "decontaminated": int(len(after_near) - len(final)),
            "decontaminated_by": {k: int((hits[k][after_near] >= decontam_min_run).sum()) for k in benches},
            "final": int(len(final)), "validation_docs": int((sel == 1).sum()),
            "per_source": {s: dict(v) for s, v in per_src.items()},
        }
    both = (select["family"] >= 0) & (select["criminal"] >= 0)
    stats["overlap_docs_in_both"] = int(both.sum())
    stats["overlap_chars_in_both"] = int(cat["chars"][both].sum())
    stats["split_consistent_in_both"] = bool((select["family"][both] == select["criminal"][both]).all())
    off = 0
    for j, sz in zip(jobs, sizes):
        np.savez(paths.feat / f"{j}.select.npz", family=select["family"][off:off + sz],
                 criminal=select["criminal"][off:off + sz])
        off += sz
    _write_json(done, stats)
    return stats


# ---------------------------------------------------------------- tokenize
def tokenize_job(job_name: str, work: str) -> int:
    from tokenizer import load_smollm2_tokenizer
    paths = Paths(work)
    out_idx = paths.tok / f"{job_name}.idx.npz"
    if out_idx.exists():
        return 0
    sel = np.load(paths.feat / f"{job_name}.select.npz")
    keep = (sel["family"] >= 0) | (sel["criminal"] >= 0)
    tok = load_smollm2_tokenizer()
    eos = tok.eos_token_id
    bin_tmp = paths.tok / f"{job_name}.bin.tmp"
    rows, offsets, lengths = [], [], []
    pos = 0
    batch, batch_rows = [], []

    def flush(fh):
        nonlocal pos
        for r, ids in zip(batch_rows, tok(batch, add_special_tokens=False)["input_ids"]):
            ids.append(eos)
            fh.write(np.asarray(ids, dtype=np.uint16).tobytes())
            rows.append(r)
            offsets.append(pos)
            lengths.append(len(ids))
            pos += len(ids)
        batch.clear()
        batch_rows.clear()

    with open(bin_tmp, "wb") as fh:
        for r, rec in enumerate(_iter_candidates(paths.cand / f"{job_name}.jsonl.gz")):
            if not keep[r]:
                continue
            batch.append(rec["text"])
            batch_rows.append(r)
            if len(batch) >= 64:
                flush(fh)
        if batch:
            flush(fh)
    os.replace(bin_tmp, paths.tok / f"{job_name}.bin")
    np.savez(out_idx, rows=np.array(rows, np.int64), offsets=np.array(offsets, np.int64),
             lengths=np.array(lengths, np.int64))
    return pos


# ---------------------------------------------------------------- write shards
def write_category(paths: Paths, jobs: list[str], root: str, domain: str, split_name: str,
                   cap: int | None = None) -> dict:
    """Shuffle (seeded) and write one category's shards + manifest entry."""
    category = f"{domain}_law_{split_name}"
    marker = paths.work / f"write.{category}.json"
    if marker.exists():
        return _read_json(marker)
    want = 0 if split_name == "train" else 1
    docs = []   # (job index, offset, length, source, kind)
    for ji, j in enumerate(jobs):
        sel = np.load(paths.feat / f"{j}.select.npz")[domain]
        ix = np.load(paths.tok / f"{j}.idx.npz")
        feat = np.load(paths.feat / f"{j}.npz", allow_pickle=False)
        source, kind = feat["source"], feat["kind"]   # load once: NpzFile re-reads an array on every access
        for r, o, ln in zip(ix["rows"], ix["offsets"], ix["lengths"]):
            if sel[r] == want:
                docs.append((ji, int(o), int(ln), str(source[r]), str(kind[r])))
    rng = random.Random(f"{SEED}-{category}")
    left_out = {"docs": 0, "tokens": 0}
    if cap is not None:   # keep every statute, then a seeded random sample of case law
        statutes = [d for d in docs if d[4] == "statute"]
        cases = [d for d in docs if d[4] != "statute"]
        rng.shuffle(cases)
        total = sum(d[2] for d in statutes)
        chosen = list(statutes)
        for i, d in enumerate(cases):
            if total >= cap:
                left_out = {"docs": len(cases) - i, "tokens": int(sum(x[2] for x in cases[i:]))}
                break
            chosen.append(d)
            total += d[2]
        docs = chosen
    rng.shuffle(docs)
    out_dir = Path(root) / category
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"{out_dir} is not empty; refusing to overwrite (delete it to rebuild)")
    started = utc_now_iso()
    mm = [np.memmap(p, dtype=np.uint16, mode="r") if p.stat().st_size else None   # empty: no kept docs
          for p in (paths.tok / f"{j}.bin" for j in jobs)]
    writer = ShardWriter(out_dir, SHARD_TOKENS, {"tokens": StreamSpec("uint16", ".bin")})
    shards, tok_by_src, doc_by_src = [], Counter(), Counter()
    total = 0

    def record(infos):
        for info in infos:
            shards.append({"file": info.files["tokens"], "tokens": info.num_tokens,
                           "checksum_sha256": info.sha256["tokens"]})
            log(f"write {category}: {info.files['tokens']} ({info.num_tokens:,} tokens)")

    buf, nbuf = [], 0
    for ji, o, ln, src, _ in docs:
        buf.append(mm[ji][o:o + ln])
        nbuf += ln
        tok_by_src[src] += ln
        doc_by_src[src] += 1
        total += ln
        if nbuf >= 16_000_000:
            record(writer.write(tokens=np.concatenate(buf)))
            buf, nbuf = [], 0
    if buf:
        record(writer.write(tokens=np.concatenate(buf)))
    last = writer.close()
    if last:
        record([last])
    return {"category": category, "shards": shards, "total_tokens": total, "docs": len(docs),
            "per_source_token_counts": dict(tok_by_src), "per_source_doc_counts": dict(doc_by_src),
            "left_out_by_cap": left_out, "started_at": started, "finished_at": utc_now_iso()}


def manifest_entry(w: dict, domain: str, split_name: str, dedup_stats: dict, extract_stats: dict,
                   cap: int | None) -> dict:
    d = dedup_stats[domain]
    filt = Counter()
    for st in extract_stats.values():
        filt.update(st[domain]["dropped"])
    return {
        "source": "; ".join(sorted({LICENCES[s].split(",")[0] + f" ({s})" for s in w["per_source_doc_counts"]})),
        "sources": {s: LICENCES[s] for s in sorted(w["per_source_doc_counts"])},
        "domain": f"{domain} law",
        "split": split_name,
        "document_split": {"method": "sha256(seed + near-duplicate cluster key), after dedup",
                           "seed": SEED, "validation_ratio": VALIDATION_RATIO},
        "selection": {"case_law_profile": CASE_PROFILE[domain], "tagger": "data/domains.py",
                      "statutes": "by heading (data.domains)" if domain == "family" else
                                  "by heading + US Code Title 18 + FRE"},
        "sequence_length": SEQUENCE_LENGTH,
        "eos_boundary_handling": ("eos_token_id appended after every document; documents shuffled (seeded); "
                                  "train and validation documents are disjoint"),
        "shard_tokens_target": SHARD_TOKENS,
        "target_tokens": cap if cap is not None else w["total_tokens"],
        "shards": w["shards"],
        "per_source_token_counts": w["per_source_token_counts"],
        "per_source_doc_counts": w["per_source_doc_counts"],
        "total_tokens_written": w["total_tokens"],
        "total_docs_written": w["docs"],
        "total_exact_duplicates_removed": d["exact_duplicates_removed"],
        "total_near_duplicates_removed": d["near_duplicates_removed"],
        "total_decontaminated": d["decontaminated"],
        "decontaminated_by_benchmark": d["decontaminated_by"],
        "filter_counts": dict(filt),
        "left_out_by_token_cap": w["left_out_by_cap"],
        "note": "dedup/filter/decontam counts are per domain over train+validation",
        "started_at": w["started_at"],
        "finished_at": w["finished_at"],
    }


def register_category(root: str, category: str, entry: dict) -> None:
    def fn(m: dict) -> None:
        cats = m.setdefault("categories", {})
        if category in cats:
            raise RuntimeError(f"manifest already has '{category}'; not overwriting")
        cats[category] = entry
    update_manifest(Path(root), fn)
