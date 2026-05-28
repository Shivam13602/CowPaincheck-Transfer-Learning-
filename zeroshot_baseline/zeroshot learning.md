# Zero-Shot Learning and Weak-Label Fine-Tuning

This folder archives the completed UCAPS v2.9 to Holstein/Jersey transfer baseline work. It contains the original zero-shot evaluation files, the weak-label cow-held-out fine-tuning outputs, and the older v2.9 baseline/evaluation artifacts that were produced before the new DANN + SSL transfer experiment.

## Folder Contents

- `baseline/`: the original zero-shot package, including Holstein zero-shot evaluator scripts, UCAPS v2.9 loader helpers, checkpoint-preparation utilities, baseline documentation, reference UCAPS test metrics, and the synced zero-shot outputs.
- `holstein_weak_label_cv_outputs_a100/`: A100 freeze-CNN weak-label adaptation run.
- `holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/`: A100 full-backbone weak-label fine-tuning run at learning rate `1e-5`.
- `holstein_weak_label_cv_outputs_smoke/`: small smoke-test split/output audit from development.
- `v2.9/`: older v2.9 evaluation, calibration, checkpoint helper, and baseline result artifacts that predate the DANN + SSL folder split.

The shared raw Holstein/Jersey sequence dataset remains outside this folder at `../cow_face_sequences_10s_250/` because it is an input shared by all experiments.

## Data Used

The target dataset is `cow_face_sequences_10s_250`, built from Canadian Holstein/Jersey dairy cow videos. It contains:

- `250` face-crop sequences.
- `10` seconds per sequence.
- `240` frames per sequence at `24 FPS`.
- `32` unique cows.
- Cow-level labels: `Healthy=105`, `Unhealthy=145`.
- Video-level labels: `Healthy=123`, `Unhealthy=127`.

The labels are weak disease-context proxies. They are not veterinary pain scores at the exact clip time. In this archive:

- `Healthy` was treated as no-pain proxy.
- `Unhealthy` was treated as pain proxy.
- Results should be interpreted as proxy label separation, not validated dairy-cow pain detection.

## Source Model

The source model is UCAPS v2.9, trained on beef cattle facial videos from castration-pain experiments. The model architecture is:

- 2D CNN frame encoder.
- LSTM temporal model.
- Attention pooling over frames.
- Task1 head: binary pain/no-pain.
- Task2 head: pain moment/intensity class.

The zero-shot run used the complete nine-fold UCAPS checkpoint bundle `v2.9_20260222_144752` with `ckpt_kind=task2`.

## Run 1: Zero-Shot, No Fine-Tuning

### Goal

The first question was:

> If we run the already-trained UCAPS v2.9 beef-cattle pain model directly on the Holstein/Jersey face clips, do the predicted Task1 pain probabilities separate healthy vs unhealthy proxy groups?

No target-domain training was performed in this run.

### Command Shape

The run was performed with:

```text
python evaluate_holstein_zero_shot_v2.9.py
  --manifest-csv .../cow_face_sequences_10s_250/completed_manifest.csv
  --sequence-root .../cow_face_sequences_10s_250
  --checkpoint-dir .../checkpoints_v2.9/v2.9_20260222_144752
  --ckpt-kind task2
  --train-py .../baseline/v2.9_training_classification.py
  --out-dir .../baseline/holstein_zero_shot_outputs_144752
  --num-workers 8
  --device cuda
```

### Artifacts

Primary local output folder:

- `baseline/holstein_zero_shot_outputs_144752-20260502T051858Z-3-001/holstein_zero_shot_outputs_144752/`

Important files:

- `holstein_zero_shot_report.md`
- `holstein_zero_shot_predictions_20260502T051222Z.csv`
- `holstein_zero_shot_run_20260502T051222Z.json`
- `group_summary_video_health_20260502T051222Z.csv`
- `group_summary_cow_health_20260502T051222Z.csv`
- `group_summary_condition_20260502T051222Z.csv`
- `group_summary_dataset_20260502T051222Z.csv`

### Results

The zero-shot run used all 9 UCAPS folds, `0..8`, on all `250` Holstein/Jersey sequences.

Mean Task1 `pain_prob` by video-level health context:

| video_health_status | n | mean pain_prob | std |
| --- | ---: | ---: | ---: |
| Healthy | 123 | 0.4306 | 0.0491 |
| Unhealthy | 127 | 0.4367 | 0.0559 |

Mean Task1 `pain_prob` by cow-level health label:

| cow_health_status | n | mean pain_prob | std |
| --- | ---: | ---: | ---: |
| Healthy | 105 | 0.4295 | 0.0454 |
| Unhealthy | 145 | 0.4368 | 0.0573 |

Mean Task1 `pain_prob` by proxy condition:

| health_condition | n | mean pain_prob |
| --- | ---: | ---: |
| fresh cows | 7 | 0.4239 |
| healthy | 38 | 0.4279 |
| healthy folder | 85 | 0.4318 |
| lame | 18 | 0.4313 |
| lameness | 55 | 0.4295 |
| lameness/stiffness | 11 | 0.4469 |
| possible mastitis | 21 | 0.4414 |
| possible metritis | 10 | 0.4764 |
| sudden fall | 4 | 0.4452 |
| unhealthy folder | 1 | 0.3877 |

### Interpretation

Zero-shot transfer was weak. Healthy and unhealthy proxy groups were almost indistinguishable:

- Video-level gap: about `0.006`.
- Cow-level gap: about `0.007`.

The model produced mid-range probabilities around `0.43` to `0.44`, suggesting that the beef-cattle UCAPS source model did not directly generalize to the Holstein/Jersey weak proxy labels. Some condition-level groups had slightly higher means, especially `possible metritis`, but subgroup sizes were small and should only be treated as hypothesis-generating.

## Run 2: Weak-Label Fine-Tuning With Frozen CNN

### Goal

After the weak zero-shot separation, the next run tested whether the UCAPS model could adapt to the Holstein/Jersey weak target by training only the temporal and classification layers while keeping the CNN visual encoder frozen.

### Split Protocol

The protocol was cow-held-out:

- Final test cows: `["363", "403", "404", "408"]`.
- Final test set: `29` sequences.
- Test labels: `13` unhealthy proxy positives, `16` healthy proxy negatives.
- Training pool: remaining `28` cows.
- Inner validation: `7` folds, each holding out `4` cows.

This prevents clip leakage across train, validation, and final test by cow ID.

### Setup

Run folder:

- `holstein_weak_label_cv_outputs_a100/`

Important settings:

- Label column: `video_health_status`.
- Mapping: `Healthy=0`, `Unhealthy=1`.
- Initialization: UCAPS v2.9 checkpoint bundle `v2.9_20260222_144752`.
- Freeze CNN: `True`.

### Validation Results

Validation performance varied widely across folds:

| fold | val cows | val AUC | val F1 | val F1 opt | cow AUC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 374,402,432,433 | 0.7296 | 0.0000 | 0.4255 | 1.0000 |
| 1 | 310,378,417,427 | 0.5029 | 0.1818 | 0.6786 | 0.5000 |
| 2 | 370,394,415,436 | 0.4647 | 0.4000 | 0.6296 | 0.5000 |
| 3 | 323,349,352,439 | 0.5208 | 0.4211 | 0.7500 | 0.5000 |
| 4 | 354,405,421,428 | 0.8163 | 0.6000 | 0.8636 | 1.0000 |
| 5 | 406,425,426,438 | 0.4542 | 0.1739 | 0.7692 | 0.5000 |
| 6 | 255,355,387,446 | 0.7333 | 0.4000 | 0.6977 | 1.0000 |

### Final Test Results

Sequence-level final test metrics:

| metric | value |
| --- | ---: |
| n | 29 |
| positives | 13 |
| negatives | 16 |
| threshold from validation | 0.1571 |
| accuracy | 0.4483 |
| balanced accuracy | 0.5000 |
| F1 | 0.6190 |
| precision | 0.4483 |
| recall | 1.0000 |
| AUC | 0.5481 |

Cow-level final test metrics:

| metric | value |
| --- | ---: |
| n cows | 4 |
| positives | 2 |
| negatives | 2 |
| accuracy | 0.5000 |
| balanced accuracy | 0.5000 |
| F1 | 0.6667 |
| AUC | 0.5000 |
| F1 opt | 0.8000 |

Final test cow aggregates:

| cow_id | target | n sequences | mean pain_prob |
| --- | ---: | ---: | ---: |
| 363 | 1 | 2 | 0.4915 |
| 403 | 1 | 11 | 0.4741 |
| 404 | 0 | 10 | 0.5117 |
| 408 | 0 | 6 | 0.4361 |

### Interpretation

Freezing the CNN did not solve the target-domain generalization problem. The model tended to select a low threshold and classified all final-test positives correctly, but it also misclassified all final-test negatives at that threshold. Balanced accuracy was chance at both sequence and cow level.

## Run 3: Weak-Label Full-Backbone Fine-Tuning

### Goal

The next run tested whether unfreezing the CNN backbone and fine-tuning the whole model with a lower learning rate would adapt better to Holstein/Jersey videos.

### Setup

Run folder:

- `holstein_weak_label_cv_outputs_a100_full_ft_lr1e5/`

Important settings:

- Label column: `video_health_status`.
- Mapping: `Healthy=0`, `Unhealthy=1`.
- Initialization: UCAPS v2.9 checkpoint bundle `v2.9_20260222_144752`.
- Freeze CNN: `False`.
- Learning rate: `1e-5`.
- Epochs: `20`.
- Batch size: `8`.
- Same final test cows and inner CV folds as the freeze-CNN run.

### Validation Results

Validation AUC improved in some folds, but fold-to-fold instability remained:

| fold | val cows | best epoch | val AUC | val F1 | val F1 opt | cow AUC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 374,402,432,433 | 1 | 0.8778 | 0.5882 | 0.7273 | 1.0000 |
| 1 | 310,378,417,427 | 0 | 0.5409 | 0.6522 | 0.6909 | 0.5000 |
| 2 | 370,394,415,436 | 0 | 0.3824 | 0.5714 | 0.6296 | 0.5000 |
| 3 | 323,349,352,439 | 11 | 0.4167 | 0.4348 | 0.7500 | 0.5000 |
| 4 | 354,405,421,428 | 14 | 0.8912 | 0.8261 | 0.9231 | 1.0000 |
| 5 | 406,425,426,438 | 11 | 0.4167 | 0.3871 | 0.7692 | 0.5000 |
| 6 | 255,355,387,446 | 1 | 0.9111 | 0.6977 | 0.7857 | 1.0000 |

### Final Test Results

Sequence-level final test metrics:

| metric | value |
| --- | ---: |
| n | 29 |
| positives | 13 |
| negatives | 16 |
| threshold from validation | 0.3500 |
| accuracy | 0.4483 |
| balanced accuracy | 0.5000 |
| F1 | 0.6190 |
| precision | 0.4483 |
| recall | 1.0000 |
| AUC | 0.4567 |

Cow-level final test metrics:

| metric | value |
| --- | ---: |
| n cows | 4 |
| positives | 2 |
| negatives | 2 |
| accuracy | 0.5000 |
| balanced accuracy | 0.5000 |
| F1 | 0.6667 |
| AUC | 0.5000 |
| F1 opt | 0.8000 |

Final test cow aggregates:

| cow_id | target | n sequences | mean pain_prob |
| --- | ---: | ---: | ---: |
| 363 | 1 | 2 | 0.6677 |
| 403 | 1 | 11 | 0.6131 |
| 404 | 0 | 10 | 0.6798 |
| 408 | 0 | 6 | 0.5550 |

### Interpretation

Full-backbone fine-tuning raised probabilities overall but did not improve final held-out cow separation. In fact, the negative cow `404` received the highest mean pain probability among the four final test cows. This suggests that the model may be learning cow identity, camera/farm context, disease-folder artifacts, or other nuisance signals rather than robust pain expression.

## Overall Conclusion

The completed baseline experiments show:

- Zero-shot UCAPS v2.9 transfer to Holstein/Jersey weak labels is weak.
- Freeze-CNN weak-label fine-tuning does not generalize cleanly to unseen cows.
- Full-backbone fine-tuning at low learning rate improves some validation folds but still fails on the 4-cow final test.
- The weak labels are not strong enough to validate pain detection.
- Better target-domain representation learning, source-target alignment, and veterinary calibration are needed.

These findings motivated the separate `../Dann transfer/` experiment folder, which contains the new DANN + SSL pipeline.

## Important Caveat

Do not describe these results as dairy-cow pain detection accuracy. They are diagnostic transfer-learning results against weak healthy/unhealthy disease-context labels. A veterinary pain-scored calibration set is still required for a scientifically valid pain model on Holstein/Jersey cows.
