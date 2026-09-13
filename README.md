# Law-based-SLM

A from-scratch decoder-only language model project built by a three-member team.

## Target

- Model size: ~500M parameters
- Context length: 8,192 tokens
- Training budget: 10B tokens
- Available dataset target: 12–15B tokens
- Hardware target: NVIDIA B200
- Architecture: decoder-only Transformer
- Training objective: next-token prediction

## Team Responsibilities

### Member A — Data & Tokenization
Responsible for corpus collection, cleaning, filtering, deduplication, BPE tokenizer, vocabulary analysis, tokenization, sequence packing, dataset sharding, and dataset manifests/versioning.

Main folders:
- `data/`
- `tokenizer/`

### Member B — Model & Pretraining
Responsible for Transformer architecture, embeddings, RMSNorm, RoPE, GQA, SwiGLU, Transformer blocks, LM head, initialization, training loop, AdamW, LR scheduling, gradient clipping, mixed precision, checkpointing, training stability, generation, and KV cache.

Main folders:
- `model/`
- `training/`
- `generation/`

### Member C — Evaluation & Post-training
Responsible for evaluation harness, validation loss, perplexity, generation evaluation, benchmarking, experiment tracking, SFT, synthetic data, and preference optimization.

Main folders:
- `evaluation/`
- `post_training/`

## Project Structure

```text
Law-based-SLM/
│
├── common/
│   ├── __init__.py
│   └── interfaces.py
│
├── configs/
│
├── data/
│   └── __init__.py
│
├── tokenizer/
│   └── __init__.py
│
├── model/
│   └── __init__.py
│
├── training/
│   └── __init__.py
│
├── generation/
│   └── __init__.py
│
├── evaluation/
│   └── __init__.py
│
├── post_training/
│   └── __init__.py
│
├── scripts/
│
├── tests/
│   └── test_interfaces.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Folder Purposes

### `common/`
Contains contracts shared between all three members.

`common/interfaces.py` defines:
- tokenizer interface
- batch format
- model output format
- model interface
- generation configuration
- evaluation result format
- evaluator interface
- trainer/checkpoint state

Do not change shared interfaces without discussing the change with the team.

### `configs/`
Stores model and training configuration files.

Examples:
```text
model_500m.json
train_500m.json
```

### `data/`
Owned primarily by Member A.

Contains code for:
- loading raw data
- cleaning
- filtering
- deduplication
- packing
- sharding
- dataset loading

Runtime directories may later include:
```text
data/raw/
data/processed/
```

These should not be committed to Git.

### `tokenizer/`
Owned primarily by Member A.

Contains BPE tokenizer implementation, training, encode/decode, serialization, and vocabulary analysis.

### `model/`
Owned primarily by Member B.

Expected modules may include:
```text
config.py
embeddings.py
rmsnorm.py
rope.py
attention.py
swiglu.py
block.py
transformer.py
```

### `training/`
Owned primarily by Member B.

Expected modules may include:
```text
optimizer.py
scheduler.py
trainer.py
checkpoint.py
state.py
```

### `generation/`
Owned primarily by Member B.

Contains autoregressive generation, temperature/top-k/top-p sampling, and KV cache.

### `evaluation/`
Owned primarily by Member C.

Contains validation, perplexity, benchmark evaluation, generation evaluation, regression tests, and checkpoint comparison.

### `post_training/`
Owned primarily by Member C.

Contains SFT, instruction-data processing, preference datasets, and methods such as DPO.

### `scripts/`
Contains executable entry-point scripts such as:
```text
train_tokenizer.py
prepare_data.py
train.py
evaluate.py
generate.py
```

Reusable implementation code should remain inside the appropriate package.

### `tests/`
Contains automated tests.

Run:
```bash
pytest
```

Do not merge code into `main` if tests fail.

## Runtime Directories

These directories are created when needed and should normally not be committed:

```text
data/raw/
data/processed/
checkpoints/
```

Example checkpoint:

```text
checkpoints/
└── checkpoint_000100/
    ├── model.pt
    ├── optimizer.pt
    ├── scheduler.pt
    ├── model_config.json
    ├── train_config.json
    ├── trainer_state.json
    └── tokenizer.json
```

Scripts should create runtime directories automatically when required.

## Shared Tensor Conventions

```text
B  = batch size
T  = sequence length
D  = model dimension
H  = number of attention heads
Dh = head dimension
V  = vocabulary size
```

Typical shapes:

```text
Token IDs:      [B, T]
Hidden states:  [B, T, D]
Logits:         [B, T, V]
```

## Standard Training Batch

Member A should provide:

```python
{
    "input_ids": Tensor[B, T],
    "labels": Tensor[B, T],
}
```

Both should use `dtype=torch.long`.

Causal shift:

```text
tokens:
[A, B, C, D, E]

input_ids:
[A, B, C, D]

labels:
[B, C, D, E]
```

Only one component should perform this shift.

## Git Workflow

Do not develop directly on `main`.

Use short-lived feature branches.

Examples:

```text
a/tokenizer-bpe
a/data-cleaning

b/b00-model-config
b/b01-embeddings
b/b02-rmsnorm

c/eval-harness
c/sft-pipeline
```

Workflow:

```text
main
  ↓
create feature branch
  ↓
implement feature
  ↓
run tests
  ↓
push branch
  ↓
open pull request
  ↓
review
  ↓
merge into main
```

After merging, create the next branch from the latest `main`.

## Collaboration Rules

Coordinate before changing:
- tokenizer interface
- special token IDs
- batch format
- model forward interface
- model output format
- checkpoint format
- generation interface
- evaluation interface
- config field names

Shared contracts live in `common/interfaces.py`.

## Development Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Important Git Rules

Do not commit:

```text
.venv/
__pycache__/
.pytest_cache/
data/raw/
data/processed/
checkpoints/
*.pt
*.pth
*.ckpt
```

## Current Development Stage

The shared repository structure and interfaces are established.

The team can begin in parallel:

```text
Member A → Data + Tokenization
Member B → Model + Pretraining
Member C → Evaluation + Post-training
```

Each member should create their first feature branch from the latest `main`.
