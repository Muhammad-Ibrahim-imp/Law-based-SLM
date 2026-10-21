# Legal pretraining data (family law and criminal law)

These are the domain pretraining corpora for **SLM 1 (family law)** and **SLM 2 (criminal law)**, built on 20 Oct 2026. This is written mainly for Member B, who runs pretraining. Prepared by Member A (data and tokenization).

Code: `data/legal_prep.py` (logic) and `scripts/prepare_legal_data.py` (CLI). The full build log and every decision are in `results/legal_prep/REPORT.md` (git-ignored).

## Categories

| Category | Tokens | Documents | Shards |
|---|---:|---:|---:|
| `family_law_train` | 1,075,030,691 | 233,230 | 5 |
| `family_law_validation` | 4,970,536 | 1,110 | 1 |
| `criminal_law_train` | 4,681,271,524 | 1,010,523 | 18 |
| `criminal_law_validation` | 24,089,895 | 5,102 | 1 |

- **Family law** is data-limited. Its unique train data (1.075B tokens) is stored once, so reaching the 4.7B budget means **~4.4 passes** at training time. It's 96% CourtListener, plus 37M tokens of HFforLegal opinions not already in CourtListener and 9M tokens of state family codes.
- **Criminal law:** all criminal case law and statutes that passed the filters came to 4.68B tokens, **just under the 4.7B cap**, so nothing was left out (about 1.0 pass). The train set has 4.50B tokens of CourtListener, 0.14B of HFforLegal, 37M of state criminal codes, 2M of US Code Title 18, and the Federal Rules of Evidence.
- **Overlap:** documents that are both family and criminal (e.g. child abuse, domestic violence, custody fights involving a conviction) are in both corpora on purpose. There are 13,202 such documents (~115M tokens, ~11% of family), and they have the same train/validation split in both.

## Sources and licences

| Source | Used for | Selection | Licence |
|---|---|---|---|
| Pile of Law `courtlisteneropinions` (all 22 files) | family, criminal | keyword tagger `data/domains.py`: family "loose", criminal "strict" | CC BY-NC-SA 4.0 |
| HFforLegal/case-law, split `us` (all 20 files) | family, criminal | same tagger; near-duplicates of CourtListener opinions are dropped in favour of the CourtListener copy | CC-BY-4.0 |
| Pile of Law `state_code` | family, criminal | family / criminal titles by heading (`data/domains.py`), near-deduplicated across yearly editions | CC BY-NC-SA 4.0 |
| Pile of Law `uscode`, Title 18 + Title 18 Appendix | criminal | by title | CC BY-NC-SA 4.0 |
| Pile of Law `fre` (Federal Rules of Evidence) | criminal | all | CC BY-NC-SA 4.0 |

**CC BY-NC-SA means non-commercial use only**, with attribution and share-alike. That's fine for this academic project. Any release of models or data must respect it.

## Format

The format is identical to `general_train` / `general_validation`, so the existing loaders work unchanged:
- **Shards:** `dataset/<category>/shard_XXXXX.bin`, raw uint16 token IDs (SmolLM2 tokenizer, vocabulary 49,152).
- **Document boundaries:** EOS (`<|endoftext|>`, `tokenizer.eos_token_id`) after every document.
- **Order:** documents are shuffled with a fixed seed before writing, so shards aren't grouped by source or court.
- **Shard size:** 2^28 = 268,435,456 tokens per shard; the last shard is partial.
- **Manifest:** `dataset/manifest.json` → `categories.<category>`. It uses the same keys as general: `shards: [{file, tokens, checksum_sha256}]`, `total_tokens_written`, `total_docs_written`, `sequence_length` (8,192), `shard_tokens_target`, `split`, `source`. It also adds per-source token/document counts, dedup, decontamination and filter counts, the token cap, and UTC timestamps.
- **Split:** train and validation are document-disjoint. 0.5% of documents go to validation by a stable hash, assigned after deduplication so that near-duplicate copies never straddle the split.

**Checks already run on all four categories:**
- `data.general_prep.validate_shards()` passes (token ranges, counts and SHA-256 checksums).
- `ShardedTokenDataset(root, category, 8192)` loads them, returning `torch.long` tensors with `labels[:-1] == input_ids[1:]`.

## How to train on them

```python
from data.shard_dataset import ShardedTokenDataset

train = ShardedTokenDataset("dataset", "family_law_train", 8192)        # or criminal_law_train
val = ShardedTokenDataset("dataset", "family_law_validation", 8192)
```

**Repeated passes come from the sampler, not from duplicated data.** Set `target_train_tokens` to 4,700,000,000. `experiments.data.StatefulBatchSampler` starts a new seeded epoch whenever it runs out of windows:
- family runs **~4.4 epochs**;
- criminal runs about one epoch (4.68B tokens).

Checkpoints store the sampler's epoch and position, so resume works across epochs.

**Runner change needed:** `experiments/runner.py` hard-codes the general categories (lines 145–149: `ShardedTokenDataset(..., "general_train", ...)` and `"general_validation"`). For legal pretraining it needs the category names as parameters, e.g. new config fields `train_category` / `validation_category` that default to the general names:

| SLM | train category | validation category |
|---|---|---|
| SLM 1 (family) | `family_law_train` | `family_law_validation` |
| SLM 2 (criminal) | `criminal_law_train` | `criminal_law_validation` |

That file is Member B's, so I haven't changed it.

Continuing from the 200M general checkpoint (Gemma 3, sequence length 8,192) uses the same tokenizer and context, so no data changes are needed.

## Decontamination (evaluation stays clean)

Pretraining documents that contain evaluation text were removed. The method:
1. Compute 13-gram overlap (lowercased, punctuation stripped) with **CaseHOLD validation + test** (the opinion excerpts) and the **LegalBench test** splits (all 162 tasks).
2. Ignore 13-grams that are common in the corpus (legal formulae and citations such as "any rational trier of fact could have found…").
3. Drop a document if it shares a passage of at least 50 consecutive rare 13-grams with an evaluation example (about 62 words or more).

Removed: 430 family and 2,442 criminal documents, almost all for CaseHOLD. **CaseHOLD (test and validation) and LegalBench can be used for evaluation.** CaseHOLD's test split was never used in SFT either (`reports/sft_data.md`). Short quotations of standard legal tests remain, as they do in any legal corpus.

**Limit of the check:** documents were removed only for runs of at least 50 consecutive rare 13-grams (about 62 words). LegalBench items shorter than about 62 words therefore couldn't be checked for contamination: even a verbatim copy of such an item in the corpus would not be removed. Many LegalBench tasks have short inputs (one-sentence clauses, hearsay or rule-application questions), so treat results on those tasks with that in mind. CaseHOLD contexts are longer (~200 words), so the check covers them.

## Known cleaning issue in the current shards

The shards were built before a cleaning fix (21 Oct). Hyphenated compounds split across PDF line breaks were merged without their hyphen ("well-\nreasoned" → "wellreasoned"). `data/cleaner.py` now keeps the hyphen unless the merged word also appears elsewhere in the document. The current shards still have the merged forms, which affects only case law with PDF line wraps. Rebuilding (extract stage onward) would pick up the fix.

## Getting the data onto Modal

The four categories, a filtered `manifest.json` and a dataset card are in our **private** Hugging Face dataset repo **`Law-based-slms/legal-pre-training-data`**. The upload was verified file by file against the shard SHA-256s. `scripts/modal_fetch_data.py` copies the data from there onto the training account's volume `law-slm-data` (mounted at `/vol/data`, dataset root `/vol/data/dataset`, the same as `modal_training.py`).

**Until `a/data-prepare` is merged into `main`, check out that branch first.** The scripts only exist there:

```bash
git fetch origin
git checkout a/data-prepare
```

One-time setup on the training account: create a Modal Secret named `huggingface` that holds a Hugging Face token with read access to the repo. A read token is enough; ask Member A to add you to the `Law-based-slms` organisation if you can't see the repo.

```bash
modal profile activate <training-account-profile>
modal secret create huggingface HF_TOKEN=hf_your_read_token
```

Then fetch all four categories (detached, so closing your terminal doesn't stop it):

```bash
modal run --detach scripts/modal_fetch_data.py
```

What it does:
1. Downloads only the requested categories plus `manifest.json` into `/vol/data/_incoming/`.
2. Checks every shard's size and SHA-256 against the manifest.
3. Refuses if a category already exists on the volume.
4. Moves the folders into `/vol/data/dataset/`.
5. Merges the four entries into the existing `manifest.json`. Other categories (`general_train`, `sft`, …) are left untouched, and it refuses on a tokenizer mismatch.
6. Runs `validate_shards` on each category.
7. Commits the volume and deletes `_incoming`.

Options:
- Only some categories: `--categories family_law_train,family_law_validation`.
- Replace categories that are already there: `--overwrite`.
- Progress and errors are in the Modal dashboard under the app `law-slm-fetch-data`. It downloads ~11 GB and takes about 15–30 minutes.

## Rebuilding

```bash
python -m scripts.prepare_legal_data --root dataset --work dataset/_work/legal
```

- The build is resumable: finished files and stages are skipped.
- The first run downloads ~25 GB (Pile of Law CourtListener + HFforLegal).
- `--pilot` runs everything on 1 CourtListener file + 1 HFforLegal file + the statute sources, into its own output root.
- Expect several hours on a 12-core PC.
