# Staged adaptation ablations (after zero-shot + weak-label metrics)

Run these only when baseline numbers show enough transfer signal to justify compute and noisy-label risk.

## Stage A — Frozen backbone, train heads only

- Load each fold checkpoint or a single reference fold.
- Freeze CNN weights; train LSTM, attention, Task1/Task2 heads on weak labels.
- Optimizer: AdamW with low LR; early stop on cow-held-out validation.

## Stage B — Unfreeze temporal stack

- Unfreeze LSTM + attention; keep CNN frozen one extra epoch for stability.

## Stage C — Partial CNN unfreeze

- Unfreeze last CNN block at very low LR; monitor overfitting on cow ID.

## Optional — Representation alignment

- CORAL / MMD on pooled embeddings before the classifier if domain shift dominates label noise.
- Only add after Stages A–C show insufficient separation under cow-held-out CV.

## Reporting

- Always report cow-held-out metrics; quote weak labels as `painful_condition_proxy` only.
- Compare against zero-shot [`evaluate_holstein_zero_shot_v2.9.py`](evaluate_holstein_zero_shot_v2.9.py) outputs on the same manifest.
