# Law-based-SLM

A from-scratch legal AI system built around **two specialized Small Language Models (SLMs)** and a **JEV orchestration layer**.

The system will train and evaluate multiple Transformer architecture variants before selecting the architecture for the final SLMs.

---

## 1. Project Idea

The project consists of two specialized legal SLMs:

- **SLM 1 — Family Law**
  - Specialized in family-law knowledge, legal provisions, terminology, documents, and reasoning related to family law.

- **SLM 2 — Criminal Law**
  - Specialized in criminal-law knowledge, legal provisions, terminology, documents, and reasoning related to criminal law.

- **JEV — Routing & Orchestration**
  - Receives the user's legal query.
  - Determines whether the query should be handled by SLM 1, SLM 2, or both.
  - Coordinates the outputs of the selected SLM(s).
  - Produces the final system response.

### Conceptual Flow

```text
                    User Legal Query
                           │
                           ▼
                     ┌───────────┐
                     │    JEV    │
                     │Orchestrator│
                     └─────┬─────┘
                           │
                ┌──────────┼──────────┐
                │          │          │
                ▼          ▼          ▼
           Family Law  Criminal Law  Both
             SLM 1       SLM 2      SLMs
                │          │          │
                └──────────┼──────────┘
                           ▼
                     JEV Coordination
                           │
                           ▼
                     Final Response
```

---

# 2. Final Model Plan

| Component | Target |
|---|---|
| Number of SLMs | 2 |
| SLM 1 | Family Law |
| SLM 2 | Criminal Law |
| Target size | ~200M parameters each |
| Context length | 8,192 tokens |
| Tokenizer | SmolLM2 tokenizer |
| Vocabulary size | 49,152 |
| Architecture | Decoder-only Transformer |
| Training | From scratch |
| Objective | Next-token prediction |
| Orchestration | JEV |
| GPU provider | Modal Labs |
| Final GPU | To be determined experimentally |

The final architecture will **not be assumed in advance**. It will be selected after systematic architecture experiments.

---

# 3. Total 14.7B-Token Data Budget

The project uses a **14.7B-token total training-data budget** covering pre-training and SFT.

The proposed allocation is:

| Data Category | Tokens | Percentage | Purpose |
|---|---:|---:|---|
| General pre-training data | 5.0B | ~34% | General language and knowledge |
| Legal data — Family Law (SLM 1) | 4.7B | ~32% | Legal-domain specialization |
| Legal data — Criminal Law (SLM 2) | 4.7B | ~32% | Legal-domain specialization |
| Conversational SFT data | 0.3B | ~2% | Supervised fine-tuning |
| **Total** | **14.7B** | **100%** | |

Therefore:

```text
Total training-data budget = 14.7B tokens

Pre-training = 14.4B tokens
SFT          = 0.3B tokens
```

## Legal Data Specialization

The **9.4B legal-token portion (4.7B per SLM) will be specialized for the two SLMs**:

```text
                    9.4B Legal Data
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
     4.7B Family Law Data     4.7B Criminal Law Data
          → SLM 1                  → SLM 2
```

Each SLM is trained on a **4.7B-token legal corpus** specific to its own domain.

The 300M conversational SFT tokens are part of the overall 14.7B budget and are used during the post-pre-training SFT stage.

---

# 4. Training Pipeline

The overall training pipeline is:

```text
Data Collection
      │
      ▼
Cleaning & Filtering
      │
      ▼
Deduplication
      │
      ▼
Tokenization
      │
      ▼
Dataset Preparation
      │
      ├─────────────────────────┐
      │                         │
      ▼                         ▼
General Data               Legal Data
      │                    ┌────┴────┐
      │                    │         │
      │                Family Law  Criminal Law
      │                    │         │
      └──────────┬─────────┴─────────┘
                 ▼
          Pre-training
          14.4B tokens
                 │
                 ▼
       Supervised Fine-Tuning
           300M tokens
                 │
                 ▼
        Post-training / Evaluation
                 │
                 ▼
          Final SLMs + JEV
```

---

# 5. Architecture Experimentation

Before training the final ~200M-parameter SLMs, approximately four Transformer architecture variants will be implemented and compared.

### Candidate Architectures

1. **GQA-based Transformer**
2. **DeepSeek-inspired Transformer**
3. **Qwen-inspired Transformer**
4. **LLaMA-inspired Transformer**

The implementations will be developed from scratch while taking architectural ideas from the respective model families.

The goal is not to assume that one architecture is automatically better, but to experimentally determine which design is most suitable for this project.

---

# 6. Architecture Scaling Experiments

Each architecture will initially be tested at smaller model sizes:

- ~25M parameters
- ~50M parameters
- ~100M parameters

Additional scales may be tested if required.

After comparing the architectures, the selected architecture will be scaled to approximately **200M parameters** for the final SLMs.

### Experimental Process

```text
GQA ─────────────┐
DeepSeek ────────┤
Qwen ────────────┼──► 25M / 50M / 100M
LLaMA ───────────┘
                       │
                       ▼
                Compare Results
                       │
                       ▼
                Select Architecture
                       │
                       ▼
                   Scale to ~200M
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Family Law       Criminal Law
            SLM 1             SLM 2
```

---

# 7. Architecture Evaluation Criteria

The architecture experiments will use consistent evaluation procedures wherever possible.

### Model Quality

- Validation loss
- Perplexity
- Generation quality
- Legal-domain evaluation

### Training Performance

- Training throughput
- Tokens/second
- GPU memory usage
- Maximum practical batch size
- Sequence-length scalability
- Gradient accumulation requirements

### Training Stability

- Loss behavior
- Gradient stability
- NaN/Inf detection
- Checkpoint recovery
- Training convergence

### Inference Performance

- Generation speed
- Memory usage
- KV-cache behavior
- Batch-size scalability

### Efficiency

- Parameter efficiency
- Checkpoint size
- Computational cost
- GPU cost
- Training time

The results will be tracked so that the final architecture decision is based on experimental evidence.

---

# 8. GPU Selection

Training will be performed using **Modal Labs**.

Candidate GPUs include:

- NVIDIA H100
- NVIDIA H200
- NVIDIA A100
- NVIDIA RTX PRO 6000

The final GPU will be selected after benchmarking the actual workload.

GPU experiments will consider:

- GPU memory
- Training throughput
- Inference throughput
- Batch-size capacity
- Sequence length
- Mixed-precision performance
- Cost per training step
- Cost per billion tokens
- Overall price/performance

The GPU selection will therefore be based on the project's actual workload rather than theoretical specifications alone.

---

# 9. Model Architecture Components

The Transformer implementation will contain components such as:

- Token embeddings
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- Multi-Head Attention / Grouped Query Attention
- Key-value heads where applicable
- SwiGLU or architecture-specific feed-forward network
- Transformer blocks
- Final normalization
- Language-model head
- Weight tying where appropriate
- Causal attention
- KV cache for generation

Architecture-specific components will vary between the experimental model variants.

---

# 10. Training

The training system will implement:

- Next-token prediction
- AdamW optimizer
- Learning-rate scheduling
- Gradient accumulation
- Gradient clipping
- Mixed-precision training
- Checkpointing
- Training-state recovery
- Validation
- Loss tracking
- Experiment tracking

### Standard Causal Language Modeling

For a sequence:

```text
Tokens:
[A, B, C, D, E]
```

The model receives:

```text
Input:
[A, B, C, D]
```

and predicts:

```text
Labels:
[B, C, D, E]
```

Only one component of the system should perform this causal shift.

---

# 11. SFT

The project includes **300M conversational tokens** for supervised fine-tuning.

SFT will be performed after the pre-training stage.

The SFT data is included within the overall 14.7B-token budget:

```text
14.4B Pre-training
+
0.3B SFT
=
14.7B Total
```

The SFT stage will focus on improving the models' ability to produce useful conversational responses from legal queries.

---

# 12. Post-training

Post-training activities may include:

- Supervised fine-tuning
- Instruction/conversational data processing
- Preference datasets
- Preference optimization where appropriate
- Generation evaluation
- Legal-domain evaluation
- Regression testing

The exact post-training techniques and their data requirements will be finalized during development.

---

# 13. Evaluation

Member C will develop an architecture-independent evaluation framework.

Evaluation will include:

- Validation loss
- Perplexity
- Generation evaluation
- Legal-domain benchmarks
- Regression tests
- Training/inference benchmarks
- Experiment tracking
- Checkpoint comparison

The evaluation framework will not depend on the internal implementation of a particular Transformer architecture.

---

# 14. Shared Model Interface

The model interface should remain standardized across architectures.

### Input

```python
input_ids: Tensor[B, T]
```

### Output

```python
logits: Tensor[B, T, V]
```

Where:

```text
B = batch size
T = sequence length
V = vocabulary size
```

### Standard Training Batch

```python
{
    "input_ids": Tensor[B, T],
    "labels": Tensor[B, T],
}
```

Both tensors use:

```python
torch.long
```

This allows the evaluation and training systems to work across different architecture implementations.

---

# 15. Shared Tensor Conventions

The project uses the following notation:

```text
B  = batch size
T  = sequence length
D  = model dimension
H  = number of attention heads
Dh = attention head dimension
V  = vocabulary size
```

Typical tensor shapes:

```text
Token IDs:       [B, T]
Hidden states:   [B, T, D]
Logits:          [B, T, V]
```

---

# 16. Project Structure

```text
Law-based-SLM/
│
├── common/
│   ├── __init__.py
│   └── interfaces.py
│
├── configs/
│   ├── model_200m.json
│   └── train_200m.json
│
├── data/
│   ├── __init__.py
│   ├── raw/
│   └── processed/
│
├── tokenizer/
│   └── __init__.py
│
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── swiglu.py
│   ├── block.py
│   └── transformer.py
│
├── training/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   ├── checkpoint.py
│   └── state.py
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

---

# 17. Folder Responsibilities

### `common/`

Contains shared contracts and interfaces used across the project.

Examples:

- Tokenizer interface
- Batch format
- Model interface
- Model output format
- Generation configuration
- Evaluation result format
- Evaluator interface
- Trainer/checkpoint state

Shared interfaces should not be changed without coordination between team members.

### `configs/`

Contains model and training configurations.

Examples:

```text
model_200m.json
train_200m.json
```

Experimental model configurations can also be stored here.

### `data/`

Responsible for:

- Data loading
- Cleaning
- Filtering
- Deduplication
- Dataset preparation
- Sequence packing
- Dataset sharding
- Dataset loading

Runtime data directories should not be committed to Git.

### `tokenizer/`

Contains integration with the **SmolLM2 tokenizer**.

Responsibilities include:

- Tokenizer loading
- Encoding
- Decoding
- Special-token handling
- Vocabulary analysis
- Tokenizer serialization/configuration

The project uses:

```text
Vocabulary size = 49,152
```

### `model/`

Contains all Transformer architecture implementations.

### `training/`

Contains the complete pre-training infrastructure.

### `generation/`

Contains autoregressive generation functionality, including:

- Sampling
- Temperature
- Top-k
- Top-p
- KV cache

### `evaluation/`

Contains:

- Evaluation harness
- Metrics
- Benchmarks
- Validation
- Generation evaluation
- Regression tests
- Experiment tracking

### `post_training/`

Contains:

- SFT
- Instruction-data processing
- Preference datasets
- Preference optimization

### `scripts/`

Contains executable project entry points.

Examples:

```text
prepare_data.py
train.py
evaluate.py
generate.py
```

Reusable implementation should remain inside the appropriate package rather than being placed directly in scripts.

### `tests/`

Contains automated tests for the project.

Run:

```bash
pytest
```

---

# 18. Checkpoints

Runtime checkpoints should not normally be committed to Git.

Example:

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

The exact checkpoint contents may evolve as the training system develops.

---

# 19. Git Workflow

Development should not be performed directly on `main`.

Use short-lived feature branches.

Examples:

```text
a/tokenizer-smollm2
a/data-cleaning

b/b00-model-config
b/b01-embeddings
b/b02-rmsnorm
b/b03-rope

c/eval-harness
c/sft-pipeline
c/generation-evaluation
```

Workflow:

```text
main
  │
  ▼
Feature Branch
  │
  ▼
Implementation
  │
  ▼
Tests
  │
  ▼
Push
  │
  ▼
Pull Request
  │
  ▼
Review
  │
  ▼
Merge
```

After a feature is merged, the next feature branch should be created from the latest `main`.

---

# 20. Collaboration Rules

The following shared contracts require coordination before modification:

- Tokenizer interface
- Tokenizer configuration
- Special-token configuration
- Batch format
- Model forward interface
- Model output format
- Checkpoint format
- Generation interface
- Evaluation interface
- Configuration field names
- Shared architecture/training assumptions

The goal is to allow all team members to work independently without breaking integration.

---

# 21. Team Responsibilities

## Member A — Data & Tokenization

Responsible for:

- Legal corpus collection
- General corpus collection
- Data provenance
- Data cleaning
- Filtering
- Deduplication
- Family-law data preparation
- Criminal-law data preparation
- SmolLM2 tokenizer integration
- Tokenization
- Vocabulary analysis
- Sequence packing
- Dataset sharding
- Dataset manifests
- Dataset versioning

Main folders:

```text
data/
tokenizer/
```

---

## Member B — Model, Pre-training & JEV

Responsible for:

- Transformer architecture implementation
- Architecture experiments
- Embeddings
- RMSNorm
- RoPE
- Attention
- GQA
- SwiGLU
- Transformer blocks
- Language-model head
- Model initialization
- Training loop
- AdamW
- Learning-rate scheduling
- Gradient clipping
- Mixed precision
- Checkpointing
- Training stability
- Generation
- KV cache
- Pre-training
- JEV orchestration layer
- Integration of the two SLMs

Main folders:

```text
model/
training/
generation/
```

---

## Member C — Evaluation & Post-training

Responsible for:

- Evaluation harness
- Validation loss
- Perplexity
- Generation evaluation
- Benchmarking
- Experiment tracking
- SFT
- Synthetic data where required
- Preference datasets
- Preference optimization
- Post-training

Member C will also assist Member B with:

- Architecture experiments
- Pre-training implementation
- Training experiments

Main folders:

```text
evaluation/
post_training/
```

---

# 22. Development Strategy

The project will proceed approximately as follows:

1. Establish shared project structure and interfaces.
2. Integrate the SmolLM2 tokenizer.
3. Implement the first GQA-based Transformer.
4. Build unit and integration tests.
5. Implement the training infrastructure.
6. Validate the complete training pipeline on small models.
7. Benchmark candidate Modal Labs GPUs.
8. Test batch sizes, sequence lengths, memory usage, and throughput.
9. Validate checkpointing and recovery.
10. Validate generation and KV-cache functionality.
11. Implement the DeepSeek-inspired architecture.
12. Implement the Qwen-inspired architecture.
13. Implement the LLaMA-inspired architecture.
14. Test architecture variants at approximately 25M, 50M, and 100M parameters.
15. Compare architecture quality, efficiency, stability, and cost.
16. Select the architecture based on experimental results.
17. Scale the selected architecture to approximately 200M parameters.
18. Train the family-law SLM.
19. Train the criminal-law SLM.
20. Apply SFT using the allocated conversational data.
21. Evaluate both SLMs independently.
22. Implement and integrate JEV.
23. Evaluate the complete multi-SLM system.
24. Perform final benchmarking and analysis.

---

# 23. Current Status

Current development is focused on:

- GQA architecture implementation
- Model component testing
- Training infrastructure
- Trainer testing
- Configuration experiments
- GPU benchmarking methodology

Next major stages:

```text
Complete GQA
      ↓
Complete Testing
      ↓
GPU / Training Configuration Experiments
      ↓
DeepSeek / Qwen / LLaMA Variants
      ↓
25M / 50M / 100M Experiments
      ↓
Architecture Comparison
      ↓
Select Architecture
      ↓
~200M Family-Law SLM
~200M Criminal-Law SLM
      ↓
SFT
      ↓
JEV Integration
      ↓
Final Evaluation
```

---

# 24. Development Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
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

---

# 25. Files Not to Commit

The following runtime/generated files should normally be excluded from Git:

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

Large datasets and model checkpoints should be handled through the project's designated storage and data-management workflow.

---

# 26. Project Objective

The final system aims to demonstrate a modular legal AI architecture consisting of:

```text
                    ┌─────────────────────┐
                    │     Legal Query     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        JEV          │
                    │ Routing &           │
                    │ Orchestration       │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             ┌─────────────┐       ┌─────────────┐
             │   SLM 1     │       │   SLM 2     │
             │ Family Law  │       │Criminal Law │
             └─────────────┘       └─────────────┘
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Coordinated Final  │
                    │      Response       │
                    └─────────────────────┘
```

The project combines **from-scratch Transformer development, large-scale pre-training, legal-domain specialization, supervised fine-tuning, architecture experimentation, GPU benchmarking, and multi-model orchestration** into a single legal AI system.