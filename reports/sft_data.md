# SFT data handoff

This is the supervised fine-tuning data for both SLMs, built and validated locally on 19 Oct 2026. It's written mainly for Member C, who owns SFT, and covers:
- what the data contains;
- how it is formatted;
- how to load the final mix;
- how to upload it to Modal.

Prepared by Member A (data and tokenization).

## Decisions behind it

- **SLM 1 stays Family Law** (tax law was considered and rejected on 19 Oct 2026). The reasons, from the domain measurement:
  - family law has more unique pretraining data (~1.4B tokens precision-weighted vs ~1.23B for tax);
  - its estimate has a tighter uncertainty;
  - SFT coverage is better;
  - the real family/criminal overlap exercises the JEV "both" route.
- **SFT budget raised from 0.3B to ~0.4B**, confirmed by the team on 19 Oct 2026. Pretraining budgets are unchanged, so the total budget rises by the same amount (README: 14.8B).
- The conversational part stays at the size originally planned (~248M tokens). All of the increase goes to legal data, which is seen twice.
- **Legal SFT data is identical for both SLMs.**

## Sources

| Source | What it is | Train | Test | Licence |
|---|---|---|---|---|
| `smol_smoltalk` | `HuggingFaceTB/smol-smoltalk`: general multi-turn chat (80% `smol-magpie-ultra-short`). Gives the model conversational behaviour. | its `train`, capped at 248.4M kept tokens | its `test`, uncapped | Apache-2.0 |
| `casehold` | `casehold/casehold` (Harvard Caselaw Access Project, Zheng et al. 2021): a US opinion excerpt with one cited holding replaced by `<HOLDING>`. Half the rows are "state the holding" and half are 5-option multiple choice (answer = letter + holding text). Teaches reading case law, not conversation. | CaseHOLD `train` | CaseHOLD `validation` | No licence on the HF card; verify before any release |
| `lawinstruct` | `lawinstruct/lawinstruct` (Niklaus et al. 2023), six English US subsets; see the list below. | 99% by content hash, RedditLegalQA capped at a seeded 50k sample | the other 1% | Each subset keeps its own licence; Reddit/StackExchange are not cleared for commercial use, so academic use only |
| `reddit_domain` | RedditLegalQA posts that `lawinstruct`'s 50k cap leaves out, kept only if the domain tagger marks them **family and/or criminal** | the complement of `lawinstruct`'s 50k sample (disjoint from it) | the family/criminal-tagged part of the Reddit 1% test split (a subset of `lawinstruct` test) | Reddit content, academic use only |

The six `lawinstruct` subsets:
- RedditLegalQA (r/legaladvice Q&A: lay questions, informal, all areas of law, mixed jurisdictions, sometimes wrong);
- StackExchangeQuestionsLegal (better answers, cites law);
- CaseBriefs (US case briefs: facts + issue → analysis);
- OLCMemos (DOJ Office of Legal Counsel memos: formal, federal/executive);
- MultiLexSum long_to_short (civil-rights case summaries);
- MCExamsLaw explain (exam explanations).

**Kept clean for evaluation:**
- CaseHOLD's own `test` split is never used. It stays clean for benchmarking.
- `nguha/legalbench` is evaluation-only. It's not in any training data; most tasks are contracts, civil procedure and evidence.

### Changes to the LawInstruct data, and why

- **Removed `ValidWills`:** its answers are yes/no, about 5 tokens each.
- **Removed `CABarExamEssays`:** each of its 62 records is a whole exam booklet in `answer` with an empty prompt, and every one is far longer than 8,192 tokens. There is no question field to recover.
- **Non-English instructions replaced:** 95% of LawInstruct instructions are machine-translated into 23 other EU languages (the prompts and answers are English). Each one is replaced by one of the same subset's English instructions, chosen by a stable hash.
- **CaseBriefs:** the facts and question were inside the answer ("Question: …\nAnalysis: …"). They now form the user turn, and the analysis is the assistant turn.
- **OLCMemos:** the memo's title and headnote form the research question; the rest of the memo is the answer. With English instructions, 171 duplicate memos were also removed.
- **RedditLegalQA:** 4,093 posts with an empty "Question:" are dropped.

The full details and counts are in `data/sft_sources.py` and each split's manifest `source_info`.

## Format: chat template `chatml-v1` (`tokenizer/chat.py`)

- Special tokens: `<|im_start|>` = 1, `<|im_end|>` = 2. The pad token is the EOS token (`<|endoftext|>`).
- One turn is `<|im_start|>{role}\n{content}<|im_end|>\n`, assembled as token IDs, never by tokenizing a rendered string.
- **No default system prompt.** None of these sources has a system turn.
- **Trained tokens:** assistant content plus its closing `<|im_end|>`, so the model learns to stop. Role headers, user turns and padding are not trained.
- **Inference:** build the prompt with `ChatEncoder(tokenizer).encode(messages, add_generation_prompt=True)` (it ends with `<|im_start|>assistant\n`), and **stop on token ID 2**. Training and inference must both use `ChatEncoder` / `encode_chat`.
- Any change to the format bumps `CHAT_FORMAT_VERSION` and must be agreed with Members B and C.

## Storage and loading

- **Sequence length:** 8,192, the 200M pretraining context. Conversations longer than that are dropped, never truncated.
- **Packing:** whole conversations are best-fit packed into rows of exactly 8,192 tokens. A conversation never crosses a row. Leftover space is pad with mask 0. There is no attention reset between packed conversations; per-shard `*.spans.npy` files record each conversation's (start, length) if document masking is wanted later.
- **Files:** `dataset/sft/<source>/<split>/shard_XXXXX.bin` (uint16 IDs), `.mask.bin` (uint8, 1 = trained) and `.spans.npy`.
- **Manifest:** `manifest.json` → `categories.sft.sources.<source>.<split>`, holding status, sequence length, shards with SHA-256, stats and `source_info`.
- **Labels:** `data/sft_dataset.py::SFTDataset` builds `labels[t] = input_ids[t+1]` where `mask[t+1] == 1`, and `-100` everywhere else. Both losses in `training/` ignore -100.

**The final mix:** smol_smoltalk once, and each legal source twice, by listing it twice:

```python
from data.sft_dataset import SFTDataset

mix = [("smol_smoltalk", "train")] + [
    ("casehold", "train"), ("lawinstruct", "train"), ("reddit_domain", "train"),
] * 2
train_ds = SFTDataset("dataset", mix)          # 46,503 rows of 8,192 tokens
test_ds = SFTDataset("dataset", [(s, "test") for s in
                                 ("smol_smoltalk", "casehold", "lawinstruct", "reddit_domain")])
```

**Checks already run:**
- `--validate` passes on all 8 splits.
- `SFTDataset` loads the mix above.
- A batch from `training.dataloader.create_dataloader(train_ds, batch_size=2)` passes `training.batch.validate_batch` (shape 2 × 8,192).

## Final mix (train)

| Source | Examples | Unique tokens | Trained tokens | Passes | Tokens seen | Trained tokens seen |
|---|---:|---:|---:|---:|---:|---:|
| smol_smoltalk | 302,729 | 248,366,379 | 197,356,878 | 1 | 248,366,379 | 197,356,878 |
| casehold | 42,002 | 17,375,498 | 1,419,816 | 2 | 34,750,996 | 2,839,632 |
| lawinstruct | 65,500 | 34,258,339 | 10,837,971 | 2 | 68,516,678 | 21,675,942 |
| reddit_domain | 30,102 | 14,032,294 | 2,459,117 | 2 | 28,064,588 | 4,918,234 |
| **Total** | | **314,032,510** | **212,073,782** | | **379,698,641** | **226,790,686** |

- **Total SFT tokens seen: 0.38B.** The smol_smoltalk cap is 300M minus casehold train and lawinstruct train.
- **Legal share: 34.6% of tokens seen and 13.0% of trained tokens.** At the old 0.3B plan it was ~18% and ~6%.
- Legal trained tokens are much lower than legal tokens because CaseHOLD prompts are ~400 tokens with ~30-token answers, and Reddit answers are short.
- Padding is under 1% in every split.

Test splits, in kept tokens:

| Split | Kept tokens |
|---|---:|
| smol_smoltalk test | 19,645,338 |
| casehold test | 2,152,603 |
| lawinstruct test | 905,328 |
| reddit_domain test | 204,942 |

LawInstruct per subset (train, unique):

| Subset | Examples | Tokens | Trained tokens |
|---|---:|---:|---:|
| RedditLegalQA | 50,000 | 21.76M | 3.86M |
| StackExchangeQuestionsLegal | 10,068 | 6.21M | 3.85M |
| MultiLexSum | 2,190 | 2.62M | 0.35M |
| OLCMemos | 578 | 2.29M | 2.22M |
| CaseBriefs | 2,595 | 1.34M | 0.55M |
| MCExamsLaw | 69 | 0.03M | 0.01M |

281 examples are dropped as longer than 8,192 tokens: 280 OLC memos (858 kept by the loader → 578 built) and 1 other.

## Domain shares (from `results/sft_domains/REPORT.md`)

Tagging uses the shared tagger `data/domains.py` (short profile for user turns up to 4,000 characters), on the user turn. Shares of **legal trained tokens**, one pass; the mix repeats every legal source twice, so the shares are the same:

| Domain | Examples | Tokens | Trained tokens | Share of legal trained |
|---|---:|---:|---:|---:|
| Family only | 17,433 | 8.25M | 1.53M | 10.4% |
| Criminal only | 34,520 | 16.28M | 3.21M | 21.8% |
| Both | 2,704 | 1.53M | 0.26M | 1.8% |
| Other legal | 82,947 | 39.61M | 9.71M | 66.0% |

- Family including "both" is 12.2%; criminal including "both" is 23.6%.
- Short-profile precision on Reddit/StackExchange samples (30 per label, after one adjustment): **family 80%, criminal 83%, other 83%**.
- reddit_domain is what raises the family share; without it family would be 5.3% of legal trained tokens.

## Decoded examples

`⟦` marks where the trained span starts. `[...]` marks text cut for length here.

**casehold**

```text
<|im_start|>user
The following excerpt from a U.S. court opinion cites a case, but the holding of the cited case has been replaced with <HOLDING>.

of serious physical harm,” Gamer, 471 U.S. at 11, 105 S.Ct. at 1701, to Ponseti, or to others if the suspects were allowed to flee. [...] 3 . But see Harper v. Harris County, 21 F.3d 597, 600 (5th Cir.1994) (<HOLDING>). 4 . Smith also observes: “Furthermore, the

What is the holding of the cited case?<|im_end|>
<|im_start|>assistant
⟦Holding that the significant injury requirement of johnson v morel 876 f2d 477 5th cir1989 a case involving a fourth amendment violation is no longer valid in the wake of hudson v mcmillianus 112 sct 995 117 led2d 156 1992 in which the supreme court held that a showing of a significant injury was not required to prove an eighth amendment violation.<|im_end|>
```

CaseHOLD holdings are lowercased and stripped of punctuation upstream.

**lawinstruct** (RedditLegalQA)

```text
<|im_start|>user
Answer this legal concern as if you were a Reddit user. Remember to add the IANAL disclaimer if you're not a lawyer.

Question: Pretty much in the title. My nephew primarily lives with his mother, who shuffles him from caretaker to caretaker, sometimes not seeing him for a week or so at a time. [...] Would I be out of line to have my own private talks with his teachers and request they call me with education inquiries? I'm sure I can't give permission for any kind of education intervention. How can I make sure he is safe when he not staying with us?<|im_end|>
<|im_start|>assistant
⟦Analysis:  His teachers are mandated reporters.  If you won’t pick up the phone and report the concern for his living conditions then perhaps if you share them with the school they will.<|im_end|>
```

**reddit_domain** (tagged criminal)

```text
<|im_start|>user
Given this legal issue, provide advice as if you're responding on Reddit. If you aren't a lawyer, include the IANAL disclaimer.

Question: Using a throwaway here for obvious reasons, but here's my situation without giving away too much info:

I live in Missouri, and a local woman has been missing from the area for nearly a year now, she just mysteriously disappeared one day. [...] I'd say I'm 85% sure that this person knows what happened to her, or like I said, is the one behind it completely. [...]
Topic:
Criminal Law<|im_end|>
<|im_start|>assistant
⟦Analysis:  Check your local/state crimestoppers service.<|im_end|>
```

Reddit answers in LawInstruct start with "Analysis:" and are often a single short comment. That's a property of the upstream data.

## Notes for the training code

- **Runner:** `experiments/runner.py` is hard-wired to the general pretraining categories (`ShardedTokenDataset`). SFT needs its own runner path that builds `SFTDataset` with the mix above and loads the 200M general checkpoint.
- **Loss weighting:** `Trainer` averages the loss per micro-batch. Under SFT, the number of trained (non -100) positions varies a lot between rows: a CaseHOLD row has ~8% trained positions, a smol_smoltalk row ~80%. So each trained token's weight depends on which micro-batch it lands in. A per-token average across the whole accumulation step (sum of losses / total trained tokens) would weight every trained token equally.
- **Shuffling:** `SFTDataset` rows are already packed; shuffle rows at the sampler level. Listing a source twice repeats its rows; the two copies are not re-packed differently.
- **Evaluation:** use the test splits above for SFT validation loss, CaseHOLD `test` for holding accuracy, and LegalBench for evaluation only.

## Rebuilding locally

Run from the repo root. Each command resumes after a crash.

```bash
python -m scripts.prepare_sft_data --root dataset --source casehold --split train
python -m scripts.prepare_sft_data --root dataset --source casehold --split test
python -m scripts.prepare_sft_data --root dataset --source lawinstruct --split train
python -m scripts.prepare_sft_data --root dataset --source lawinstruct --split test
python -m scripts.prepare_sft_data --root dataset --source reddit_domain --split train
python -m scripts.prepare_sft_data --root dataset --source reddit_domain --split test
python -m scripts.prepare_sft_data --root dataset --source smol_smoltalk --split train --max-tokens 248366163
python -m scripts.prepare_sft_data --root dataset --source smol_smoltalk --split test
```

- `--max-tokens` = 300,000,000 − (casehold train 17,375,498 + lawinstruct train 34,258,339).
- Add `--validate` to any command to check a split.
- About 0.97 GB on disk (0.33B tokens × 3 bytes, plus spans).

## Building the SFT data on Modal

The SFT data is **not uploaded anywhere**. Like the general data, it's rebuilt on the account that trains, and checked against my reference build. `scripts/modal_build_sft.py` builds all 8 splits into `/vol/data/dataset/sft/` on the volume `law-slm-data`, and adds `categories.sft` to the existing `manifest.json` there. The update is atomic and locked, and refuses another tokenizer.

**Until `a/data-prepare` is merged into `main`, check out that branch first.** The script and the reference file only exist there:

```bash
git fetch origin
git checkout a/data-prepare
modal profile activate <sft-account-profile>
modal run --detach scripts/modal_build_sft.py
```

**How it reproduces the reference build:**
- Every upstream dataset is read at a pinned commit (`data/sft_sources.REVISIONS`):
  - smol-smoltalk `f73fe857…`;
  - CaseHOLD parquet conversion `8a4dbd58…`;
  - LawInstruct `578e1007…`.
- Every split uses the exact config recorded in `reports/sft_expected.json` (same seed, sequence length, LawInstruct subsets and caps, and smol_smoltalk `max_tokens` 248,366,163).
- The image pins the same `transformers==5.17.0` / `datasets==5.0.1` as `modal_training.py`.

**Practicalities:**
- **Runtime:** CPU only (4 CPUs, 16 GB); about **1–2 hours**, mostly smol-smoltalk. It downloads ~1 GB of upstream data.
- **Resumable:** rerun the same command and it skips completed splits and resumes an interrupted one. The volume is committed after each split.
- **Hugging Face token:** optional (all sources are public). If the account has the `huggingface` Modal Secret (see `reports/legal_pretraining_data.md`), it's used to avoid rate limits.

**What "matches the reference build" means:**
- At the end the function runs `validate_sft_split` on every split.
- It compares the result with `reports/sft_expected.json`: config, examples kept, kept and trained tokens, rows, and **the SHA-256 of every shard file** (tokens, mask, spans).
- If everything is identical, the log (Modal dashboard, app `law-slm-build-sft`) ends with **`SFT build matches the reference build`**. The data is then byte-for-byte what was built and checked locally.

**If it doesn't match:**
- The log prints `SFT build DIFFERS from the reference build` and one line per difference, and nothing is deleted.
- Send those lines to Member A, and don't train on the data until it's resolved.
- The usual causes are a different library version in the image or an upstream file that changed despite the pin.
