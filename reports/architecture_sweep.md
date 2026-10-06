# Architecture sweep report

This report covers the fair 12-run comparison of GQA, LFM2.5, Qwen3, and Gemma 3 at 25M, 50M, and 100M parameters. All 12 training runs finished. The 100M multiple-choice benchmarks have not been run.

The selection metric agreed for this sweep is held-out validation loss, with perplexity reported from that loss. On that metric, **Gemma 3 is the architecture to scale to the 200M models.**

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

Use **Gemma 3** for the 200M family-law and criminal-law models.

It has the best validation loss at both 50M and 100M, which are the scales closest to the 200M target. The 25M ranking is different: LFM2.5 led there by a wide margin. That lead did not hold after the longer runs.

LFM2.5 is the alternative if GPU time becomes the binding constraint. At 100M its loss is 3.5837 against Gemma 3's 3.5433, and it trains at 462,148 tokens/sec against 233,161. Qwen3 matches LFM2.5 on loss and is slower. GQA is last on loss at 50M and 100M.

This choice is not a claim about legal answering quality. The sweep used general web and textbook text, sequence length 2,048, and at most 1B tokens. Legal pre-training, supervised fine-tuning, and JEV have not started.

## What was measured, and what was not

- No NaN or Inf loss stopped a finished run.
- Checkpoint resume works. One resume bug was fixed: RNG tensors must be moved to CPU before restoring them.
- FLOPs were not measured.
- Modal invoice totals were not pulled. The H200 rate used for planning was $0.001261/second, or $4.5396/hour. The summed training-loop time across the 12 finished runs is 17,294 seconds, about 4.80 hours, or about $22 at that rate. The bill is higher. Validation, benchmarks, profiling, compiles, and the cancelled attempts are not included in that $22 figure.

## Where the checkpoints are

They are not in git. `results/`, checkpoints, and `dataset/` are ignored.

- 25M and 50M: Modal profile `throat-choker`, volume `law-slm-training`, path `/training_sweep/<name>/`
- 100M: Modal profile `shozab-siraiki`, volume `law-slm-training`, path `/training_sweep/<name>/`

Each finished directory contains `summary.json`, `metrics.jsonl`, and `checkpoint.pt`. The 25M and 50M directories also contain `benchmarks.json`.

## What is next

1. Run the same three multiple-choice benchmarks on the four 100M checkpoints.
2. Add a generation loop. The models return logits only. There is no `generate()` implementation and no chat command. See below.
3. Scale Gemma 3 to about 200M parameters. The 200M training run should use sequence length 8,192. This sweep deliberately used 2,048.
4. Pre-train on the general corpus, then on the legal corpus. Do not start legal-domain training, SFT, or JEV until the 200M configuration is fixed.

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

## Where to read the code

1. `README.md` for the project plan.
2. `data/general_prep.py` and `scripts/prepare_general_data.py` for Member A's split.
3. `tokenizer/` for the SmolLM2 tokenizer.
4. `model/` for GQA, then `architectures/qwen3.py`, `architectures/gemma3.py`, and `architectures/lfm25.py`.
5. `configs/training/` for the 12 experiment files.
6. `experiments/runner.py` and `training/` for the loop, optimizer, and checkpoints.
7. `evaluation/` for validation and multiple choice.
8. `scripts/modal_training.py` for the H200 launcher.
