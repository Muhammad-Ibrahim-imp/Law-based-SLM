# 199.4M SLM GPU benchmarks

This directory benchmarks the repository's decoder-only GQA model with:

- 199,398,144 trainable parameters
- vocabulary 49,152; context limit 8,192
- 24 layers, width 768, 12 query heads, 4 KV heads, head width 64
- SwiGLU width 2,240, RMSNorm epsilon `1e-5`, RoPE theta 500,000
- tied token embedding/output weights, no biases, no dropout
- PyTorch SDPA forced to its FlashAttention backend
- BF16 autocast, causal next-token loss, backward, and fused AdamW updates

## Recommended setup

The Modal runner is the reproducible path used for the reported results. It
uses Python 3.12 and pins the current dependency releases:

```text
torch==2.14.0
transformers==5.17.0
datasets==5.0.1
modal==1.6.0
```

Install and authenticate the local Modal client:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install modal==1.6.0
modal token new
modal profile current
```

The GPU container installs the PyTorch wheel and its bundled CUDA 13 runtime.
Do not separately install the CUDA toolkit for the Modal path. A manual cloud
machine must have an NVIDIA driver compatible with that bundled runtime;
confirm with `nvidia-smi` and:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
```

The separate `flash-attn` package is not installed: the implementation uses
PyTorch SDPA's fused FlashAttention backend and refuses to silently fall back
to math attention on CUDA.

## Run all six Modal GPUs

From the repository root:

```bash
modal run benchmarks/modal_runner.py \
  --gpu all \
  --warmup-steps 2 \
  --timed-steps 5
```

The H100 selector is `H100!`, which disables Modal's automatic H200 upgrade.
The other exact selectors are `B300`, `B200`, `H200`, `RTX-PRO-6000`, and
`A100-80GB`. Results are downloaded to `benchmarks/results/`.

For a low-cost smoke run:

```bash
modal run benchmarks/modal_runner.py \
  --gpu h100 \
  --warmup-steps 1 \
  --timed-steps 1 \
  --sequence-lengths 2048 \
  --micro-batch-sizes 2
```

## Merge GPU results

```bash
python -m benchmarks.merge_results \
  benchmarks/results/b300.csv \
  benchmarks/results/b200.csv \
  benchmarks/results/h100.csv \
  benchmarks/results/h200.csv \
  benchmarks/results/rtx-pro-6000.csv \
  benchmarks/results/a100-80gb.csv \
  --output benchmarks/results/all_gpus.csv
```

This creates a combined CSV and `all_gpus.md`, which identifies both the
fastest configuration and the lowest cost per billion tokens. “Steps per
dollar” in these files explicitly means optimizer micro-steps per dollar;
tokens per micro-step are included so unlike-sized rows are not confused.

## TinyStories stability and global batch sweep

After choosing a sequence length and micro-batch from the dummy sweep:

```bash
modal run benchmarks/modal_runner.py \
  --mode real-data \
  --gpu h100 \
  --sequence-length 2048 \
  --micro-batch-size 8 \
  --global-batch-tokens 262144,524288,1048576 \
  --steps 200
```

This uses `HuggingFaceTB/SmolLM2-135M` only for its 49,152-entry tokenizer and
streams `roneneldan/TinyStories`. It records every optimizer-step loss and
compares gradient accumulation, loss stability, throughput, and cost.
