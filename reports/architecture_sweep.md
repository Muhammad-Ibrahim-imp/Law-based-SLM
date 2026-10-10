# Architecture sweep report

This report covers the fair 12-run comparison of GQA, LFM2.5, Qwen3, and Gemma 3 at 25M, 50M, and 100M parameters, and the later 200M general pre-training of Gemma 3 and LFM2.5. All 12 smaller training runs finished. Both 200M general runs finished. The 100M and 200M multiple-choice benchmarks have not been run.

The selection metric agreed for this sweep is held-out validation loss, with perplexity reported from that loss. On that metric, **Gemma 3 is the architecture to keep.** It led at 50M, 100M, and 200M.

## Who did which part

The work follows the team split in the README.

**Member A — data and the train/validation split**

- Document-level split of SmolLM-Corpus before tokenization.
- Mix: 80% `fineweb-edu-dedup` and 20% `cosmopedia-v2`, mix seed 42. `python-edu` is excluded.
- Split: SHA-256(seed, source, text), validation ratio 0.005, split seed 42.
- Tokenizer: `HuggingFaceTB/SmolLM2-135M`, vocabulary 49,152, EOS between documents.
- Prepared corpus, from `dataset/manifest.json`: general train 1,000,001,749 tokens across 41 shards; general validation 5,026,238 tokens in one shard. Train and validation documents are disjoint.
- Packed training windows use sequence length 2,048. The 100M runs need 1,000,341,504 tokens, so they wrap about 0.08% of the packed tokens into a second epoch. Smaller runs stop earlier. The same seed-42 order is used for every architecture.

**Member B — configs, training, checkpoints, and Modal**

- Twelve configs in `configs/training/{25m,50m,100m}/`.
- Shared training setup: AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1 on parameters with more than one dimension and 0 on the rest, cosine decay to 10% of peak learning rate, 2% warmup, gradient clip 1.0, sequence length 2,048, global batch 524,288 tokens (micro-batch 32, accumulation 8).
- Peak learning rate: 6e-4 at 25M, 4.5e-4 at 50M, 3e-4 at 100M. The same peak is used for every architecture at a given size.
- One H200 per run, BF16 autocast, `torch.compile`, seed 42.
- Checkpoints store the model, optimizer, scheduler, tokens seen, and RNG/sampler state. Resume was required and was used.
- Optimizer correction, which applies to every run in this sweep: the default Adam betas are now (0.9, 0.95), and the one-dimensional parameter group has weight decay 0. Before the fix, weight decay could still be applied to that group and the beta default did not match the agreed schedule.
- Modal profiles: `throat-choker` for the 25M and 50M stages, `shozab-siraiki` for the 100M stage. The volumes `law-slm-data` and `law-slm-training` exist in both workspaces and are not shared. There are no Modal Secrets. The function timeout is 24 hours.
- Two 100M attempts were cancelled because the local machine lost DNS and the attached Modal client cancelled the remote job. GQA had already finished. Qwen3 had no checkpoint on the first cancellation, so that run started again. Gemma 3 later resumed from its step-600 checkpoint. The final launcher spawns the remote call and exits, so a later local disconnect does not cancel the GPU job.

**Member C — validation, benchmarks, and this report**

- Full held-out validation loss and perplexity inside every training run. Validation is 5,023,744 tokens, 77 batches.
- Zero-shot multiple choice for the 25M and 50M checkpoints: HellaSwag validation (10,042), ARC-Easy test (2,376), and PIQA validation (1,838). Both raw accuracy and length-normalized accuracy are stored.
- PIQA is loaded from the official AI2 zip. `datasets` 5.0.1 cannot load the old `ybisk/piqa` script.
- The 100M checkpoints do not yet have `benchmarks.json`.

## Not assigned to Member A, B, or C

Section 21 of the README assigns data to Member A, the model and pre-training to Member B, and evaluation and post-training to Member C. The items below are outside that split. The finished ones are already in the repo, so they do not need to be rebuilt. The unfinished ones are not a current milestone for any member.

### Already done

**GPU choice.** Candidate Modal GPUs were measured, and pre-training uses an NVIDIA H200 SXM, sequence length 2,048, micro-batch 32, gradient accumulation 8, and a global batch of 524,288 tokens in BF16. That decision is recorded in README section 8.

| Path | Purpose |
|---|---|
| `benchmarks/` | Times a ~200M GQA model on B300, B200, H200, H100, A100, and RTX PRO 6000, and records tokens/second, memory, and cost |
| `benchmarks/modal_runner.py` | Starts those GPU jobs on Modal |
| `benchmarks/run_benchmark.py`, `run_real_data.py` | Runs one timing job on dummy data or real tokens |
| `benchmarks/reporting.py`, `merge_results.py` | Turns the job outputs into one table |
| `benchmarks/model_200m.py` | Builds the fixed-size model used only for that timing |
| `benchmarks/report.md`, `benchmarks/README.md` | The measured GPU table and how to repeat it |

**Comparison runner.** This is the machinery that trained the 12 models under one setup. The model code stays in `model/` and `architectures/`. The data code stays in `data/`. The score code stays in `evaluation/`.

| Path | Purpose |
|---|---|
| `experiments/` | Loads one config, builds the model, trains it, validates it, and can resume |
| `configs/training/` | The 12 JSON files: four architectures at 25M, 50M, and 100M |
| `scripts/train_sweep.py` | Starts one run, or the whole size, on the current machine |
| `scripts/modal_training.py` | Starts the same runs on one Modal H200 and keeps going if the laptop disconnects |
| `scripts/evaluate_sweep.py` | Scores a finished checkpoint on HellaSwag, ARC-Easy, and PIQA |
| `scripts/count_architecture_params.py` | Prints each architecture's parameter count |
| `reports/architecture_sweep.md` | This comparison: numbers, the Gemma 3 choice, and this ownership list |

**Shared project files.** These keep the three members working against the same rules.

| Path | Purpose |
|---|---|
| `common/interfaces.py` | The agreed shapes for tokens, batches, model outputs, checkpoints, and evaluation |
| `requirements.txt` | The pinned Python packages, including PyTorch, Transformers, Datasets, and Modal |
| `.gitignore` | Keeps `dataset/`, `results/`, logs, and checkpoint files out of git |
| `tests/` | Checks that the current code still runs. Add a test when you change your own code; do not rewrite the suite |
| Modal volumes `law-slm-data` and `law-slm-training` | Hold the token shards and the 12 checkpoints. The 25M and 50M copies are on profile `throat-choker`. The 100M copies are on profile `shozab-siraiki` |

### Not started, and not a member task

No folder exists for these, and none of them is in the team split.

| Task | What it would be |
|---|---|
| Production release | A packaged service, an API, deployment, monitoring, and a versioned release |
| Desktop app | A desktop window for asking questions |
| Mobile app | A phone app for the same thing |
| Security and jailbreak resistance | Filters, abuse tests, and defenses against prompts that try to bypass the model's limits |
| Accounts and distribution | Logins, user data, and shipping an app to users |

Generation, JEV, legal data, the 100M and 200M benchmarks, and SFT are still member work. They are listed under "What is next" and in README section 21. A chat window is not the same thing as Member B's generation code. The 200M general pre-training of Gemma 3 and LFM2.5 is finished and is recorded below.

## Result

Every run used one compiled H200. Validation loss is the final full-validation number.

### 25M — 100,139,008 tokens, 191 steps

| Architecture | Parameters | Validation loss | Perplexity | Tokens/sec | Train seconds |
|---|---:|---:|---:|---:|---:|
| LFM2.5 | 24,916,736 | 5.8365 | 342.58 | 694,957 | 144.09 |
| Gemma 3 | 24,962,152 | 6.1941 | 489.86 | 313,876 | 319.04 |
| GQA | 24,649,984 | 6.2773 | 532.33 | 500,660 | 200.01 |
| Qwen3 | 25,372,672 | 6.2793 | 533.42 | 491,120 | 203.90 |

### 50M — 500,170,752 tokens, 954 steps

| Architecture | Parameters | Validation loss | Perplexity | Tokens/sec | Train seconds |
|---|---:|---:|---:|---:|---:|
| Gemma 3 | 50,406,544 | 4.0952 | 60.05 | 346,196 | 1,444.76 |
| LFM2.5 | 50,154,368 | 4.1088 | 60.88 | 711,329 | 703.15 |
| Qwen3 | 49,854,848 | 4.2497 | 70.08 | 503,693 | 993.01 |
| GQA | 49,190,784 | 4.3035 | 73.96 | 539,815 | 926.56 |

Gemma 3 at 50M was resumed after a local disconnect. The stored `train_seconds` includes the time from before that disconnect.

### 100M — 1,000,341,504 tokens, 1,908 steps

| Architecture | Parameters | Validation loss | Perplexity | Tokens/sec | Train seconds |
|---|---:|---:|---:|---:|---:|
| Gemma 3 | 98,686,608 | 3.5433 | 34.58 | 233,161 | 4,290.34 |
| Qwen3 | 99,901,696 | 3.5825 | 35.96 | 326,848 | 3,060.57 |
| LFM2.5 | 97,026,304 | 3.5837 | 36.01 | 462,148 | 2,164.55 |
| GQA | 100,558,336 | 3.6555 | 38.69 | 351,715 | 2,844.18 |

At 100M, Qwen3 and LFM2.5 are nearly tied. Gemma 3 is ahead of both by about 0.04 loss. LFM2.5 is about twice as fast as Gemma 3.

### Multiple choice

Chance is 0.25 on HellaSwag and ARC-Easy, and 0.50 on PIQA. The 25M scores sit on that floor. At 50M, ARC-Easy and PIQA move above chance, and LFM2.5 leads those two. HellaSwag stays near 0.26 for every model. These gaps are too small, and the 100M scores are missing, so the multiple-choice results do not override the validation-loss ranking.

Accuracy / length-normalized accuracy:

| Run | HellaSwag | ARC-Easy | PIQA |
|---|---:|---:|---:|
| GQA 25M | 0.2624 / 0.2445 | 0.2727 / 0.2744 | 0.5185 / 0.5071 |
| Qwen3 25M | 0.2648 / 0.2456 | 0.2685 / 0.2677 | 0.5250 / 0.5125 |
| Gemma 3 25M | 0.2620 / 0.2447 | 0.2731 / 0.2790 | 0.5163 / 0.5033 |
| LFM2.5 25M | 0.2597 / 0.2441 | 0.2824 / 0.2769 | 0.5174 / 0.4984 |
| GQA 50M | 0.2594 / 0.2379 | 0.3295 / 0.3165 | 0.5490 / 0.5424 |
| Qwen3 50M | 0.2586 / 0.2411 | 0.3287 / 0.3165 | 0.5484 / 0.5392 |
| Gemma 3 50M | 0.2589 / 0.2452 | 0.3350 / 0.3253 | 0.5430 / 0.5473 |
| LFM2.5 50M | 0.2590 / 0.2415 | 0.3577 / 0.3447 | 0.5582 / 0.5490 |

## Recommendation

Use **Gemma 3** for the family-law and criminal-law models.

It has the best validation loss at 50M, 100M, and 200M. The 200M general run is the closest evidence to the models this project will train next. Gemma 3 finished that run at validation loss 2.9243. LFM2.5 finished at 2.9700. The 25M ranking is different: LFM2.5 led there by a wide margin. That lead did not hold after the longer runs.

LFM2.5 is the alternative if GPU time becomes the binding constraint. At 200M its loss is 2.9700 against Gemma 3's 2.9243, and it trains at 225,894 tokens/sec against 87,102. Qwen3 and GQA were not trained at 200M. At 100M, Qwen3 matched LFM2.5 on loss and was slower, and GQA was last.

This choice is not a claim about legal answering quality. The 200M run used general web and textbook text only. Legal pre-training, supervised fine-tuning, JEV, and the 200M multiple-choice benchmarks have not started.

The later copy-mechanism runs do not change this choice. They are recorded below. Gemma 3 is still the best architecture at 100M with the copy path turned on, but its loss is worse than the same model without the copy path. The 200M runs did not use the copy path.

## Copy mechanism

A second comparison used the same data, batch, learning rates, token budgets, and sequence length. Each model still predicts the next token from the vocabulary. A learned gate mixes that prediction with a second distribution that can copy any earlier token in the current context. These runs use a 2,048-token context, so that is the span the copy path can point at. The model maximum remains 8,192. The gate starts near "generate", so the first steps are close to ordinary next-token loss.

The 50M checkpoints cannot be loaded into the 100M models. The 100M models are wider and deeper, so the weight shapes do not match. The 100M copy runs were trained from the start on four H200s at once.

| Stage | Account | How it ran |
|---|---|---|
| 25M copy | `shozab-siraiki` | One H200, four models in sequence |
| 50M copy | `abdulrehmanriaz29` | One H200, four models in sequence |
| 100M copy | `spiritwolf786` | Four H200s, one architecture each |

Configs are in `configs/training_copy/`. Checkpoints are on that account's `law-slm-training` volume, under `/training_sweep_copy/<name>/`. The code is `training/copy_mechanism.py`. It does not change the architecture files.

### 25M copy — 100,139,008 tokens, 191 steps

| Architecture | Loss without copy | Loss with copy | Change | Perplexity with copy |
|---|---:|---:|---:|---:|
| LFM2.5 | 5.8365 | 6.2456 | +0.409 | 515.76 |
| Gemma 3 | 6.1941 | 6.3736 | +0.180 | 586.19 |
| Qwen3 | 6.2793 | 6.6030 | +0.324 | 737.29 |
| GQA | 6.2773 | 6.6704 | +0.393 | 788.68 |

### 50M copy — 500,170,752 tokens, 954 steps

| Architecture | Loss without copy | Loss with copy | Change | Perplexity with copy |
|---|---:|---:|---:|---:|
| Gemma 3 | 4.0952 | 4.3239 | +0.229 | 75.48 |
| LFM2.5 | 4.1089 | 4.3282 | +0.219 | 75.81 |
| Qwen3 | 4.2497 | 4.3618 | +0.112 | 78.40 |
| GQA | 4.3035 | 4.5367 | +0.233 | 93.38 |

### 100M copy — 1,000,341,504 tokens, 1,908 steps

| Architecture | Loss without copy | Loss with copy | Change | Perplexity with copy | Tokens/sec |
|---|---:|---:|---:|---:|---:|
| Gemma 3 | 3.5433 | 3.7233 | +0.180 | 41.40 | 245,087 |
| LFM2.5 | 3.5837 | 3.7267 | +0.143 | 41.54 | 495,043 |
| Qwen3 | 3.5825 | 3.7814 | +0.199 | 43.88 | 331,395 |
| GQA | 3.6555 | 3.8420 | +0.186 | 46.62 | 347,409 |

At every size, the copy path raised held-out loss. The 200M runs left it off. Gemma 3 remains the architecture to keep. LFM2.5 remains the faster alternative.

## 200M general pre-training

Both runs are finished. This is the README general-corpus stage: 5.0 billion tokens, of which 99.5% is train and 0.5% is validation. It is not the full 14.4 billion token pre-training plan. The family-law corpus, the criminal-law corpus, and the 0.3 billion token SFT set have not been collected, so those stages have not started. Qwen3 and GQA were not trained at 200M.

The copy path is off. The sequence length is 8,192. The global batch stays 524,288 tokens: sequence 8,192, micro-batch 2, accumulation 32. The peak learning rate is 2e-4 for both models. The rest of the schedule matches the smaller sweep: AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1 on parameters with more than one dimension, cosine decay to 10% of the peak, 2% warmup, gradient clip 1.0, seed 42, BF16, and `torch.compile`. One H200 ran each model. The two models were not on one workspace for the whole run, because volumes are not shared across Modal accounts and credits ran out between resumes.

The prepared train split is 4,975,001,703 tokens. The optimizer ran 9,490 steps, which is 4,975,493,120 tokens. The extra tokens are the packed windows needed to finish the last step. Held-out validation at the end is 24,993,792 tokens, 1,526 batches. The first 1,907 steps used the earlier 1 billion token corpus and its smaller validation split of 5,021,696 tokens. From step 2,000 on, validation uses the 5 billion token corpus. A loss change exactly at that switch is a change of validation set, not a training failure.

`train_seconds` is the time stored inside the training loop, including every successful resume. It does not include the hours a job spent stopped while the checkpoint was moved to a new account.

### Final held-out result

| Architecture | Parameters | Validation loss | Perplexity | Tokens/sec | Train seconds | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 3 | 200,069,440 | 2.9243 | 18.62 | 87,102 | 57,122.42 | 25,649,924,608 |
| LFM2.5 | 194,691,072 | 2.9700 | 19.49 | 225,894 | 22,025.76 | 24,017,023,488 |

Gemma 3 is ahead by 0.0457 loss. LFM2.5 is about 2.6 times faster. The final learning rate for both runs is 2e-5, which is the planned 10% floor. Gemma 3's last training loss was 2.9195, against a held-out loss of 2.9243.

Exact summary values:

| Field | Gemma 3 | LFM2.5 |
|---|---:|---:|
| Validation loss | 2.9243141773371413 | 2.970004342650242 |
| Perplexity | 18.621450656696986 | 19.49200424280426 |
| Optimizer steps | 9,490 | 9,490 |
| Train tokens | 4,975,493,120 | 4,975,493,120 |
| Train seconds | 57,122.41636337076 | 22,025.75809436121 |
| Tokens/sec | 87,102.28727632207 | 225,894.29606392395 |
| Validation tokens | 24,993,792 | 24,993,792 |
| Validation batches | 1,526 | 1,526 |

### Gemma 3 shape and loss curve

Config `configs/training/200m/gemma3_200m.json`: width 832, 18 layers, 4 query heads, 1 key-value head, head dimension 256, feed-forward 2,688. That is three repeats of the 5-local plus 1-global pattern. Parameter count 200,069,440.

Selected held-out points from `metrics.jsonl`. Steps through 1,907 use the smaller validation split.

| Step | Tokens seen | Validation loss | Perplexity |
|---:|---:|---:|---:|
| 0 | 0 | 10.9928 | 59,446.72 |
| 1,907 | 999,817,216 | 3.4453 | 31.35 |
| 2,000 | 1,048,576,000 | 3.5151 | 33.62 |
| 4,000 | 2,097,152,000 | 3.1493 | 23.32 |
| 6,000 | 3,145,728,000 | 3.0144 | 20.38 |
| 8,000 | 4,194,304,000 | 2.9467 | 19.04 |
| 9,200 | 4,823,449,600 | 2.9273 | 18.68 |
| 9,400 | 4,928,307,200 | 2.9252 | 18.64 |
| 9,490 | 4,975,493,120 | 2.9243 | 18.62 |

### LFM2.5 shape

Config `configs/training/200m/lfm25_200m.json`: width 768, 12 query heads, 6 key-value heads, raw feed-forward 4,992, and all 16 official layer types. Parameter count 194,691,072. Its finished summary is the held-out result above. The run ended at step 9,490 with validation loss 2.9700.

The finished weights are on different Modal accounts. Gemma 3 is on profile `mibrahimsh2006`. LFM2.5 is on profile `muslimonly234-glitch`. Both paths are `/training_sweep/<name>/` on volume `law-slm-training`. Earlier partial checkpoints on other accounts are older than these files.

At the planning rate of $0.001261/second, the summed training-loop time is 79,148 seconds, about 22.0 hours, or about $100. That figure is Gemma 3's 57,122 seconds plus LFM2.5's 22,026 seconds. It does not include validation time, compiles, the corpus build, or the time spent stopped between accounts.

## What was measured, and what was not

- No NaN or Inf loss stopped a finished run.
- Checkpoint resume works. One resume bug was fixed: RNG tensors must be moved to CPU before restoring them.
- FLOPs were not measured.
- Modal invoice totals were not pulled. The H200 rate used for planning was $0.001261/second, or $4.5396/hour. The summed training-loop time across the 12 finished runs is 17,294 seconds, about 4.80 hours, or about $22 at that rate. The bill is higher. Validation, benchmarks, profiling, compiles, and the cancelled attempts are not included in that $22 figure.

## Where the checkpoints are

They are not in git. `results/`, checkpoints, and `dataset/` are ignored.

- 25M and 50M, no copy path: Modal profile `throat-choker`, volume `law-slm-training`, path `/training_sweep/<name>/`
- 100M, no copy path: Modal profile `shozab-siraiki`, volume `law-slm-training`, path `/training_sweep/<name>/`
- Copy-path runs: `/training_sweep_copy/<name>/` on `shozab-siraiki` (25M), `abdulrehmanriaz29` (50M), and `spiritwolf786` (100M)
- Gemma 3 200M, finished: profile `mibrahimsh2006`, volume `law-slm-training`, path `/training_sweep/gemma3_200m/`
- LFM2.5 200M, finished: profile `muslimonly234-glitch`, volume `law-slm-training`, path `/training_sweep/lfm25_200m/`

Each finished directory contains `summary.json`, `metrics.jsonl`, and `checkpoint.pt`. The 25M and 50M directories also contain `benchmarks.json`.

## What is next

1. Run HellaSwag, ARC-Easy, and PIQA on the 100M checkpoints and on the two finished 200M checkpoints. The 25M and 50M scores are already stored. The 100M and 200M scores are not.
2. Add a generation loop. The models return logits only. There is no `generate()` implementation and no chat command. See below.
3. Collect the family-law and criminal-law corpora, then pre-train Gemma 3, without the copy path, on that text. The general 5.0 billion token stage is done for Gemma 3 and LFM2.5. SFT and JEV have not started.

## Read the repository in this order

Start with the plan, then the data, then one model, then the training loop, then the comparison. Each line is the job of that file. Skip `dataset/`, `results/`, and checkpoint files. They are generated data, not source.

### 1. Plan

| File | Purpose |
|---|---|
| `README.md` | Project plan, team split, GPU choice, and the development order |
| `reports/architecture_sweep.md` | Measured results, the Gemma 3 choice, the copy-path results, and this reading order |
| `requirements.txt` | Pinned packages: PyTorch, Transformers, Datasets, and Modal |
| `.gitignore` | Keeps the corpus, logs, results, and checkpoints out of git |

### 2. Shared contracts

| File | Purpose |
|---|---|
| `common/interfaces.py` | Agreed shapes for tokens, batches, model outputs, checkpoints, and evaluation |

### 3. Data and tokenizer — Member A

| File | Purpose |
|---|---|
| `data/__init__.py` | Marks `data` as a Python package |
| `data/cleaner.py` | Cleans raw text before it is tokenized |
| `data/dedup.py` | Removes duplicate documents |
| `data/general_prep.py` | Mixes FineWeb-Edu and Cosmopedia, splits documents, and writes shards |
| `data/shard_dataset.py` | Reads a shard as packed windows. Labels are the input shifted by one token |
| `scripts/prepare_general_data.py` | Command that runs the general-corpus preparation |
| `scripts/data_prepare.py` | Older data-preparation entry point |
| `scripts/data_peeker.py` | Prints a small sample of prepared text |
| `scripts/export_training_samples.py` | Writes a few training windows out for inspection |
| `tokenizer/__init__.py` | Loads the SmolLM2 tokenizer, vocabulary 49,152 |

### 4. GQA model — Member B

| File | Purpose |
|---|---|
| `model/config.py` | Width, depth, heads, and context length for the GQA model |
| `model/embeddings.py` | Token embedding and the tied output head |
| `model/rmsnorm.py` | RMSNorm used before attention and the feed-forward block |
| `model/rope.py` | Rotary position encoding |
| `model/attention.py` | Grouped-query attention |
| `model/swiglu.py` | SwiGLU feed-forward block |
| `model/transformer_block.py` | One GQA layer: attention, then feed-forward |
| `model/initialization.py` | Starting weight values |
| `model/model.py` | The full GQA decoder. `forward` returns logits |
| `model/parameter_count.py` | Counts parameters, with tied embeddings counted once |

### 5. The other three architectures — Member B

| File | Purpose |
|---|---|
| `architectures/__init__.py` | Marks the package |
| `architectures/registry.py` | Builds a model by name: `lfm25`, `qwen3`, or `gemma3` |
| `architectures/lfm25.py` | LFM2.5-style mix of convolution and attention layers |
| `architectures/qwen3.py` | Qwen3-style decoder |
| `architectures/gemma3.py` | Gemma 3-style decoder, with local and global attention |
| `architectures/custom.py` | Empty placeholder for a later architecture |
| `scripts/count_architecture_params.py` | Prints each architecture's parameter count |

### 6. Training loop — Member B

| File | Purpose |
|---|---|
| `training/config.py` | Shared training settings object |
| `training/batch.py` | Checks that a batch has `input_ids` and `labels` of the same shape |
| `training/loss.py` | Ordinary next-token cross-entropy. It does not shift the labels |
| `training/optimizer.py` | AdamW. Weight decay is 0 on one-dimensional parameters. Betas are (0.9, 0.95) |
| `training/scheduler.py` | Cosine decay to 10% of the peak learning rate, with 2% warmup |
| `training/gradient.py` | Clips the gradient norm |
| `training/trainer.py` | One optimization step, including accumulation, BF16, and the copy loss when it is enabled |
| `training/checkpoint.py` | Saves and loads the model, optimizer, scheduler, step, and RNG state |
| `training/dataloader.py` | Batching helper used by the earlier trainer tests |
| `training/training_step.py` | Single-step helper used outside the sweep runner |
| `training/mixed_precision_training_cuda.py` | BF16 autocast helper |
| `training/copy_mechanism.py` | Mixes the vocabulary prediction with a copy distribution over the current context |

### 7. Experiment configs and runner

| File | Purpose |
|---|---|
| `experiments/__init__.py` | Marks the package |
| `experiments/config.py` | Loads one JSON config and checks the training settings |
| `experiments/models.py` | Builds GQA or one of the three registered architectures and checks the parameter count |
| `experiments/data.py` | Shuffles training windows in a way that can resume at the same example |
| `experiments/runner.py` | Seeds the run, trains, validates, logs, and writes `summary.json` |
| `configs/training/` | Twelve JSON files for the original comparison, plus `200m/gemma3_200m.json` and `200m/lfm25_200m.json` |
| `configs/training_copy/` | The same twelve settings with the copy path turned on |

### 8. Evaluation — Member C

| File | Purpose |
|---|---|
| `evaluation/__init__.py` | Marks the package |
| `evaluation/validation.py` | Held-out loss and perplexity. Uses the copy mixture when that path is on |
| `evaluation/multiple_choice.py` | Zero-shot HellaSwag, ARC-Easy, and PIQA |
| `scripts/evaluate_sweep.py` | Local entry point for those three benchmarks |

### 9. GPU timing and the Modal launcher

| File | Purpose |
|---|---|
| `benchmarks/__init__.py` | Marks the package |
| `benchmarks/model_200m.py` | Fixed ~200M GQA model used only for GPU timing |
| `benchmarks/run_benchmark.py` | Times that model on dummy tokens |
| `benchmarks/run_real_data.py` | Times that model on real tokens |
| `benchmarks/reporting.py` | Formats one timing result |
| `benchmarks/merge_results.py` | Combines timing results into one table |
| `benchmarks/modal_runner.py` | Runs the timing jobs on the candidate Modal GPUs |
| `benchmarks/README.md` | How to repeat the GPU timing |
| `benchmarks/report.md` | The measured GPU table that led to the H200 choice |
| `scripts/train_sweep.py` | Starts one run, or a whole size, on the current machine |
| `scripts/modal_training.py` | Starts the runs on Modal. Several names can run at once, one H200 each, and a disconnect does not cancel them |

### 10. Tests

`tests/` checks the pieces above. `tests/test_copy_mechanism.py` checks the copy mixture. `tests/test_experiment_configs.py` checks the twelve original configs. The other test files match the module they name. Add a test when you change your own code. Do not rewrite the suite.

These folders are named in the plan and do not exist yet. They are still member work: `generation/` for Member B, and `post_training/` for Member C.

## How to continue text from a trained checkpoint

These checkpoints are next-token models, not chat models. A prompt is completed as more text. It will not reliably answer in a question-and-answer style until SFT.

Download the Gemma 3 100M checkpoint:

```powershell
$modal = "$env:APPDATA\Python\Python314\Scripts\modal.exe"
& $modal volume get --profile shozab-siraiki law-slm-training /training_sweep/gemma3_100m/checkpoint.pt gemma3_100m_checkpoint.pt
```

The file includes optimizer state, so it is much larger than the model weights. From the repository root, with the project dependencies installed:

```python
import torch
from experiments.config import load_experiment_config
from experiments.models import build_experiment_model
from tokenizer import load_smollm2_tokenizer

config = load_experiment_config("configs/training/100m/gemma3_100m.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_experiment_model(config).to(device).eval()
checkpoint = torch.load("gemma3_100m_checkpoint.pt", map_location=device, weights_only=False)
state = checkpoint["model_state_dict"]
if state and all(key.startswith("_orig_mod.") for key in state):
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
model.load_state_dict(state)

tokenizer = load_smollm2_tokenizer()
ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    for _ in range(40):
        next_id = torch.argmax(model(ids)[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break
print(tokenizer.decode(ids[0].tolist()))
```

Swap the config path and checkpoint to inspect another run. Greedy argmax is the simplest check that the weights load. It is not a chat product.

The reading order is the section "Read the repository in this order" above.
