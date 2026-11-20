# Fine-tuned Laya routing model

This repository path contains the preserved, calibrated Stage C epoch 3 Laya legal-routing inference export. It includes the complete model weights, routing and encoder configuration, and tokenizer assets required by the matching Laya inference implementation.

- Inference export: `final-inference/`
- Model: ModernBERT-large encoder with Laya's native four-route decision head.
- Routes: `Criminal Law`, `Family Law`, `Both`, and `None`.
- Selected checkpoint: run `c-20261115-finalist-02`, epoch 3 (151,263 optimizer updates).
- Calibration temperature for the four-option choice: `4.354228496551514`.
- Weights SHA-256: `ea69d918e1a8b7bc7548f3782b534b9d35d03c734afdcb130e7989597d589d65`.
- Upstream Laya code revision: `3a38cbe83f6ef556f3125fb0065fdc5a55d05792`.

The `.safetensors` file is stored with Git LFS. Load the checkpoint directory through Laya's `load_checkpoint` implementation at the recorded code revision. The export contains inference weights and assets; it does not contain optimizer state or resume checkpoints.
