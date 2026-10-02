# 199.4M SLM GPU benchmark report

## Scope and implementation

The repository model was updated to the requested decoder-only architecture:
49,152 vocabulary entries, 8,192 maximum context, 24 layers, model width 768,
12 query heads, 4 KV heads, head width 64, SwiGLU width 2,240, RMSNorm epsilon
`1e-5`, and RoPE theta 500,000. Linear layers have no bias, dropout is zero,
and token embedding/output weights are tied.

The exact trainable parameter count is **199,398,144**, matching the requested
approximately 199.4M total.

GPU attention used **PyTorch SDPA's forced FlashAttention backend** rather than
the separate `flash-attn` package. Every benchmark used BF16 autocast and real
causal forward, cross-entropy, backward, and fused AdamW optimizer steps.

## Dummy-data benchmark

The complete 3 × 7 grid ran on exact Modal selectors `B300`, `B200`, `H100!`,
`H200`, `RTX-PRO-6000`, and `A100-80GB`, with 2 warmups and 5 timed steps per
successful configuration. The environment was Python 3.12, PyTorch
2.14.0+cu130, and CUDA runtime 13.0.

| GPU | Fastest successful config | Tokens/s | Cost/1B tokens | Peak GiB |
|---|---|---:|---:|---:|
| Nvidia B300 | seq 2,048; micro-batch 64 | 233,954 | $8.4290 | 183.82 |
| Nvidia B200 | seq 2,048; micro-batch 32 | 219,950 | $7.8927 | 92.53 |
| Nvidia H200 SXM | seq 2,048; micro-batch 32 | 158,810 | $7.9403 | 92.53 |
| Nvidia H100 SXM5 | seq 2,048; micro-batch 16 | 132,930 | $8.2525 | 46.87 |
| Nvidia RTX PRO 6000 | seq 2,048; micro-batch 8 | 98,833 | $8.5194 | 24.04 |
| Nvidia A100 80GB | seq 2,048; micro-batch 16 | 68,738 | $10.0962 | 46.83 |

B200 sequence-specific maximum grid micro-batches were 32 at 2,048, 16 at
4,096, and 8 at 8,192. B300 maxima were 64, 32, and 16. H200 maxima were 32,
16, and 8. H100 and A100 maxima were 16, 8, and 4. RTX PRO 6000 maxima were 32,
16, and 8.

An intermediate-batch tuning pass found B200's compute-bound optimum at
micro-batch 56 (223,157 tokens/s and $7.7793/1B tokens), H200's largest working
micro-batch at 40, and B300's best cost point at micro-batch 80
($8.4030/1B tokens).

All 126 requested grid rows, including direct OOM and post-OOM skipped rows, are in
`benchmarks/results/all_gpus.csv`. The complete dual ranking is in
`benchmarks/results/all_gpus.md`.

“Steps per dollar” in the outputs means **optimizer micro-steps per dollar**.
This metric naturally ranks small micro-batches highly even when they process
fewer tokens. Tokens per dollar and cost per 1B tokens are the appropriate
metrics for comparing differently sized micro-steps. Cost per 1B tokens is not
a fixed property of a GPU: it changes with sequence length, micro-batch size,
timing duration, and whether input-pipeline time is included.

## TinyStories run and global batch comparison

The real-data run streamed `roneneldan/TinyStories` and used the
`HuggingFaceTB/SmolLM2-135M` tokenizer, whose measured size was exactly 49,152.
It ran on H200 with sequence 2,048 and micro-batch 32.

| Global tokens | Accumulation | Tokens trained | Initial loss | Final loss | Final-20 mean ± std | Tokens/s | Cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 262,144 | 4 | 52.43M | 10.9477 | 2.7291 | 2.8749 ± 0.1162 | 124,179 | $0.532 |
| 524,288 | 8 | 104.86M | 10.9319 | 2.7293 | 2.7234 ± 0.0750 | 124,439 | $1.063 |
| 1,048,576 | 16 | 209.72M | 10.9249 | 2.8412 | 2.8847 ± 0.0856 | 124,877 | $2.118 |

All three completed 200 optimizer steps without NaN or Inf. Loss decreased
substantially in every run. The 1.049M run had a transient loss spike at step
21 and recovered immediately.

These training-loss endpoints are not a controlled model-quality comparison:
larger global batches consumed more tokens, and each final loss came from a
different training batch. A held-out validation run at a fixed token budget is
required to claim a statistically best global batch. Operationally, 524,288
tokens is the best tested compromise: it had the lowest final-window mean and
variance, while 1.049M provided less than 0.4% additional throughput.

## Recommendation

For the current end-to-end pipeline, use H200:

- Main pre-training sequence length: **2,048**
- Micro-batch size: **32**
- Gradient accumulation: **8**
- Global batch: **524,288 tokens**
- Expected dummy-data throughput: about **158.8K tokens/s**
- Measured TinyStories throughput: about **124.4K tokens/s**

Sequence 2,048 is the measured throughput/cost optimum, but exclusively
training at 2,048 will not establish strong 8,192-token behavior. Use a
long-context curriculum or a later training phase at sequence 8,192,
micro-batch 8, and accumulation 8; this preserves the same 524,288-token global
batch.

The B200 compute-only result is 2–3% cheaper than H200 when data is already
tokenized, packed, prefetched, and keeps the GPU saturated. In the matched
TinyStories run, however, B200 processed 152,455 tokens/s at $11.39/1B while
H200 processed 124,439 tokens/s at $10.13/1B. H200 therefore used about 11%
fewer credits end to end because the CPU-side tokenization/data path reduced
B200 utilization.

For a fully pretokenized production pipeline, B200 can be used at sequence
2,048 and micro-batch 48 with accumulation 5 (491,520 global tokens). This is
close to the tested 524,288-token global-batch optimum, retains about 40 GiB of
memory headroom, and costs only about 0.2% more than the memory-edge
micro-batch-56 result. B300 is fastest but not cheapest.

If configuration portability between H200 and 80GB GPUs matters, use
micro-batch 16 and accumulation 16 at sequence 2,048.

## Files changed or created

Existing files changed:

- `.gitignore`
- `model/attention.py`
- `model/config.py`
- `model/initialization.py`
- `model/model.py`
- `model/rmsnorm.py`
- `model/rope.py`
- `tests/test_config.py`
- `tests/test_model.py`

New benchmark files:

- `benchmarks/model_200m.py`
- `benchmarks/run_benchmark.py`
- `benchmarks/modal_runner.py`
- `benchmarks/merge_results.py`
- `benchmarks/reporting.py`
- `benchmarks/run_real_data.py`
- `benchmarks/README.md`
- `benchmarks/report.md`
- Runtime CSV and Markdown outputs under `benchmarks/results/`

## Commands run

Key commands were:

```bash
modal profile current
modal run benchmarks/modal_runner.py --gpu h100 \
  --warmup-steps 1 --timed-steps 1 \
  --sequence-lengths 2048 --micro-batch-sizes 2
modal run benchmarks/modal_runner.py --gpu all \
  --warmup-steps 2 --timed-steps 5
python3 -m benchmarks.merge_results \
  benchmarks/results/b300.csv benchmarks/results/b200.csv \
  benchmarks/results/h100.csv benchmarks/results/h200.csv \
  benchmarks/results/rtx-pro-6000.csv \
  benchmarks/results/a100-80gb.csv \
  --output benchmarks/results/all_gpus.csv
modal run benchmarks/modal_runner.py --mode real-data --gpu h200 \
  --sequence-length 2048 --micro-batch-size 32 \
  --global-batch-tokens 262144,524288,1048576 --steps 200
modal run benchmarks/modal_runner.py --mode test
```

## Problems and uncertainties

- Modal CLI was initially absent, then its browser authorization required a
  human login. After authorization, profile and GPU runs worked.
- RTX PRO 6000 capacity queued for about eight minutes before running.
- CUDA OOMs were expected grid outcomes and were recorded; larger
  micro-batches at the same sequence length were skipped.
- Static compilation and formatting checks pass.
- The repository test run produced **70 passed, 2 skipped, 4 failed**. All four
  failures are pre-existing `Trainer.train_step` interface mismatches in
  `tests/test_gradient_accumulation.py`; neither the trainer nor those tests was
  changed by this work.
- The recommendation is based on short throughput and stability runs, not a
  full convergence or held-out quality study.
