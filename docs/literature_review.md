# Literature Review and Research Pipeline for UCAPS-to-Holstein/Jersey Cattle Pain Transfer

## Executive Summary

This document reviews peer-reviewed evidence relevant to transferring a UCAPS v2.9 cattle facial-pain model from beef-cattle castration videos to Holstein/Jersey dairy-cow farm videos. It also defines a complete data preparation, sequence creation, experiment, and evaluation pipeline for the current project.

The central conclusion is conservative: the current Holstein/Jersey labels are weak health-context labels, not pain ground truth. Metrics against `video_health_status` measure separation of Healthy versus Unhealthy proxy clips only. A model can be useful as a transfer-learning diagnostic under these labels, but target-domain pain detection requires a veterinary-scored calibration and evaluation set.

The project should therefore proceed on two tracks:

1. **Weak-label track:** use `video_health_status` for immediate proxy-label experiments, with strict cow-held-out evaluation and careful wording.
2. **Validation track:** collect veterinary pain scores on a small, diverse target set before making any claim about Holstein/Jersey pain detection.

## 1. Problem Statement and Domain Shift

The source domain is UCAPS v2.9, trained on beef cattle, mainly Nelore and Angus bulls, in an acute castration-pain context. The target domain is Canadian Holstein/Jersey dairy-cow video collected in farm settings, where available labels are health or disease-context labels such as healthy, lameness, mastitis, metritis, fresh cows, or sudden fall.

This is not a simple fine-tuning problem. It combines multiple forms of shift:

- **Breed and morphology shift:** source animals are beef bulls; target animals are dairy cows.
- **Clinical-context shift:** source pain is acute postoperative castration pain; target conditions include chronic or condition-level disease contexts.
- **Visual-domain shift:** coat patterns, muzzle appearance, camera angle, lighting, distance, background, and farm environment differ.
- **Label shift:** UCAPS source labels are validated pain labels; target `video_health_status` is only a weak proxy.
- **Sample-size limitation:** the current target set has 250 sequences from 32 cows, with most cows having a pure proxy class.

The completed experiments already show the practical consequence of this shift. Zero-shot UCAPS inference produced weak proxy separation, and cow-held-out weak-label or DANN runs did not yield deployable thresholded classifiers. The strongest completed cow-held-out final sequence AUC was weak-label frozen-CNN fine-tuning at approximately 0.548, while DANN variants remained around 0.466 to 0.481 on final sequence AUC. V1 post-fallback Task1 DANN reached approximately 0.558 sequence AUC, but final thresholded classification still predicted all final-test sequences as positive. These results support a pipeline-readiness claim, not a validated pain-detection claim.

## 2. Veterinary Pain-Label Requirements

### 2.1 UCAPS and Cattle Pain Scales

The UNESP-Botucatu cattle pain scale was validated for acute postoperative pain in cattle after orchiectomy. De Oliveira et al. (2014) reported that the refined scale was valid, reliable, responsive, and had discriminatory ability for postoperative pain assessment in cattle. This paper provides the foundation for source-domain pain labels and for rescue-analgesia thresholding.

Tomacheuski et al. (2024) later evaluated real-time and video-recorded UCAPS assessment in young adult bulls undergoing surgical castration. Their study supports the use of UCAPS in both real-time and video-recorded contexts, but it is still grounded in a specific beef-cattle castration setting. The paper itself states that future work should test broader breeds, procedures, sex, age, and dairy-cattle settings. That limitation is directly relevant here.

Gleerup et al. (2015) developed a Cow Pain Scale for dairy cattle and identified pain-related behavioural signs in dairy cows, including attention toward surroundings, head position, ear position, facial expression, response to approach, and back position. This is important because it shows that dairy-cattle pain expression is observable, but it does not make farm folder health labels equivalent to pain scores.

Feighelstein et al. (2026) provides an important recent reference for automated cattle pain recognition. The study compared deep learning video-based models with trained veterinarians using validated pain scales and subject-held-out evaluation. Its relevance is methodological: automated cattle pain recognition is plausible when trained and evaluated against validated pain labels. It does not justify calling weak Holstein/Jersey health folders pain labels.

### 2.2 Why Target Health Folders Are Not Pain Ground Truth

The target labels in this project are `Healthy` and `Unhealthy` health-context labels. These should not be described as pain labels for four reasons:

1. A cow can be clinically unhealthy without displaying visible facial pain during a specific 10-second clip.
2. A healthy-folder clip can still contain stress, discomfort, occlusion, posture change, or ambiguous expression.
3. Most cows in the current target set are pure under the proxy label, so models can learn cow identity, farm context, source folder, coat pattern, or camera conditions.
4. The final test has only four cows, so cow-level AUC is high variance.

The correct wording is:

- **Allowed:** "Holstein/Jersey weak health-proxy separation."
- **Allowed:** "Transfer diagnostic under cow-held-out proxy labels."
- **Not allowed:** "Validated Holstein/Jersey pain detection."
- **Not allowed:** "UCAPS pain model proves dairy-cow pain from current target labels."

## 3. Existing Project State

### 3.1 Current Target Dataset

The current target dataset is:

```text
Transferlearning/cow_face_sequences_10s_250/
```

It contains:

- 250 sequences.
- 32 cows.
- 10 seconds per sequence.
- 24 FPS stored output.
- 240 stored face-crop frames per sequence.
- 224 x 224 RGB face crops.
- `completed_manifest.csv` as the sequence-level manifest.
- Per-sequence `metadata.json`.
- Per-frame `frames.csv`.

Current manifest summary:

| Field | Value |
| --- | ---: |
| Total sequences | 250 |
| Total cows | 32 |
| Video-health Healthy | 123 |
| Video-health Unhealthy | 127 |
| Cow-health Healthy | 105 |
| Cow-health Unhealthy | 145 |
| Dataset source: Truro | 148 |
| Dataset source: Yashan RAC | 98 |
| Dataset source: Cow 349 sudden fall | 4 |

Detection quality in the current 250-sequence manifest:

| Field | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| Detected frames out of 240 | 236.64 | 109 | 240 |
| Filled frames out of 240 | 3.36 | 0 | 131 |
| Mean detection confidence | 0.915 | 0.767 | 0.948 |
| Minimum detection confidence | 0.817 | 0.601 | 0.937 |

### 3.2 Current Sequence Creation Rules

The current sequences were created by:

```text
yolo_cow_face/create_10s_face_sequences.py
```

Current baseline rules:

- Select readable videos with cow IDs and Healthy/Unhealthy labels.
- Exclude cow `409` by default because it is not classified in the current health list.
- Select videos with a reproducible seed.
- Use round-robin sampling by cow so all classified cows are represented before adding additional clips.
- Use one 10-second center window per selected video.
- Sample frames at 24 FPS, giving 240 stored frames.
- Run a trained YOLO cow-face detector with confidence threshold `0.60`.
- Use crop padding `0.08`.
- If multiple faces are detected, select the largest detected face, interpreted as the nearest cow.
- Save 224 x 224 face crops as `frame_0000.jpg` to `frame_0239.jpg`.
- If a frame has no detection, fill it from the nearest previous or first valid detection and record this in `frames.csv`.

This is a defensible baseline dataset and must be preserved for comparability.

### 3.3 Current Modeling Code

The existing DANN workspace already contains the required core modeling components:

- `v2.9_training_classification.py`: UCAPS CNN-LSTM-attention model with `extract_features()`.
- `holstein_v29_dataset.py`: maps Holstein/Jersey manifest rows into UCAPS-compatible sequence entries.
- `weak_label_adapt_v2.9.py`: weak-label cow-held-out adaptation with BCE, focal, GCE, calibration, diagnostics, sliding inference, and bootstrap options.
- `ssl_pretrain_holstein_v2.9.py`: SimSiam target SSL with fold-train-only leakage control.
- `dann_adapt_v2.9.py`: gradient reversal, domain classifier, source Task1 supervision, optional Task2 auxiliary loss, optional weak target loss, source sanity tracking, calibration, and diagnostics.
- `V2/run_task1_vast.sh`: V2 launcher using 14 folds with 2 validation cows per fold and fixed final test cows.

### 3.4 Completed Result Interpretation

The current completed results show:

- Zero-shot UCAPS on all 250 clips: `video_health_status` AUC about 0.529.
- Frozen-CNN weak fine-tune: final sequence AUC about 0.548.
- Full-backbone weak fine-tune: final sequence AUC about 0.457.
- SSL-initialized weak fine-tune: final sequence AUC about 0.447.
- SSL+DANN with Task2 auxiliary: final sequence AUC about 0.481, source Task1 sanity AUC about 0.536.
- Task1-only SSL+DANN baseline: final sequence AUC about 0.466, source Task1 sanity AUC about 0.458.
- V1 Task1 DANN with proxy fallback/no SSL: final sequence AUC about 0.558, but thresholded final-test predictions remained all-positive.
- V1 weak-label GCE: final sequence AUC about 0.471, also with all-positive final-test threshold behaviour.

The recurring failure mode is threshold collapse. Validation-derived thresholds classify all 29 final-test sequences as positive in the key completed runs, producing recall of 1.0 but zero true negatives and balanced accuracy of 0.5. AUC should therefore be read only as a ranking diagnostic until operating points are fixed on stronger validation evidence.

## 4. Literature-Backed Model Families

### 4.1 Face Detection, Cropping, and Identity Control

Cow face detection and identity recognition are active areas in precision livestock farming. Li et al. (2021) showed that lightweight CNN models can identify dairy cows from images with complex backgrounds. Xu et al. (2022) proposed CattleFaceNet, combining RetinaFace-style detection and ArcFace-style metric learning for cattle identification. Lei, Wen and Li (2024) proposed a multi-target cow-face detection model for complex scenes, and Gao et al. (2025) proposed CFR-YOLO for cow face detection in large-pasture settings.

For this project, these papers justify:

- using a trained detector rather than manual crops;
- recording detection confidence and fill rate;
- auditing multi-face frames;
- treating cow identity as a leakage risk;
- using cow-held-out splits as mandatory, not optional.

The current "largest face" rule is practical, but it should be audited in videos with multiple cows. In future dense sequence creation, store multi-face count, selected box area, second-largest box area, and selected/second area ratio. These fields help identify clips where the detector may have followed the wrong animal.

### 4.2 Temporal Modeling for Pain Videos

The UCAPS v2.9 model uses a 2D CNN per frame, LSTM temporal modeling, attention pooling, and Task1/Task2 heads. This is reasonable for small video datasets because it separates frame-level feature extraction from sequence-level aggregation.

Feighelstein et al. (2026) used video-level aggregation for automated cattle pain recognition and emphasized subject-held-out validation. Their pipeline differs from this project, but it supports three principles:

- video clips should be scored at video or subject level, not only at independent-frame level;
- temporal information matters for animal pain;
- subject-held-out validation is required.

The current code stores 240 frames per 10-second sequence, but the model usually samples fewer frames (`max_frames=32` by default). This distinction must be explicit:

- **Stored sequence density:** 240 frames at 24 FPS.
- **Model input density:** 32, 64, 96, or 128 sampled frames per forward pass.
- **Evaluation densification:** sliding-window inference can scan more stored frames without increasing training tensor size.

### 4.3 Self-Supervised Target Adaptation

Self-supervised learning is useful when labeled target examples are scarce. SimSiam is especially attractive for the current target setting because it avoids explicit negative pairs, large batches, and momentum encoders while using a stop-gradient mechanism to avoid collapse (Chen and He, 2021). This matters because many target clips share cows and conditions; contrastive methods with careless negatives could push visually similar same-cow or same-condition examples apart.

MoCo and MAE are valid peer-reviewed alternatives (He et al., 2020; He et al., 2022), but they should be secondary options. The existing code already implements SimSiam, so the practical priority is to run leakage-safe SimSiam on fold-train cows only and test whether it improves downstream weak-label and DANN performance.

Recommended SSL rule:

- Main claim: SSL on fold-train cows only.
- Research-only ablation: transductive SSL on all target cows, clearly labelled as transductive and not comparable to the main leakage-safe claim.

### 4.4 Domain Adaptation

DANN is well matched to the source-labeled/target-unlabeled setting. Ganin et al. (2016) describe a gradient reversal layer that encourages features to be discriminative for the source task while becoming less informative about source versus target domain. This is the conceptual basis of `dann_adapt_v2.9.py`.

However, domain alignment is not automatically beneficial. Ben-David et al. (2010) show that transfer depends on both source performance and source-target discrepancy. If DANN reduces domain discrepancy while destroying source pain discrimination, the adapted feature space is not useful for pain transfer. This is why source Task1 sanity metrics are essential.

ADDA (Tzeng et al., 2017) is an alternative adversarial approach with separate source and target encoders. It should be considered only after Task1-only DANN is stable, because it adds implementation complexity and may overfit with small target data.

CycleGAN and SPGAN-style appearance transfer can be considered later (Zhu et al., 2017; Deng et al., 2018). They should not be first-line methods because changing visual appearance can also distort pain-relevant facial cues. If used, every generated-data experiment must include semantic-preservation audits.

### 4.5 Noisy Labels and Class Imbalance

The weak target labels are noisy because health context is not equivalent to visible pain. GCE is therefore a sensible loss to test because Zhang and Sabuncu (2018) designed it for robustness to noisy labels. Focal loss is also a reasonable ablation because it down-weights easy examples and emphasizes harder examples (Lin et al., 2017). Class-balanced weighting based on effective sample number can help when class or condition counts are imbalanced (Cui et al., 2019).

These methods must be described as weak-proxy diagnostics, not pain supervision. The correct interpretation is:

- BCE, focal, and GCE test whether the model can separate noisy `video_health_status`.
- They do not convert weak labels into veterinary pain labels.

### 4.6 Calibration and Uncertainty

Temperature scaling is a standard post-hoc calibration method for modern neural networks (Guo et al., 2017). The current code already reports calibrated and raw metrics. Calibration is necessary but not sufficient: it cannot fix an invalid target label or a model whose final-test scores do not separate negative cows at the selected threshold.

Uncertainty should support vet-label selection. MC dropout and Bayesian active learning are peer-reviewed approaches for uncertainty-guided image-data selection (Gal and Ghahramani, 2016; Gal, Islam and Ghahramani, 2017). In this project, uncertainty should be used primarily to choose clips for veterinary scoring, not to make clinical decisions.

### 4.7 Few-Shot and Vet-Label Adaptation

Once a veterinary-scored target subset exists, few-shot methods become relevant. Prototypical Networks learn a metric space in which classification is performed by distance to class prototypes (Snell, Swersky and Zemel, 2017). MAML learns parameters that can adapt quickly to new tasks using few labeled examples (Finn, Abbeel and Levine, 2017).

For this project, few-shot models should be secondary to simpler calibrated baselines. The first vet-label experiments should be:

1. UCAPS feature extractor plus logistic/ordinal regression.
2. Frozen encoder plus small MLP.
3. Prototypical baseline using UCAPS/SSL/DANN embeddings.
4. Fine-tuned sequence model only if the vet-label set is large enough.

## 5. Data Preparation and Sequence Creation Protocol

### 5.1 Preserve Baseline Dataset

The current 250-sequence dataset must remain frozen as:

```text
Transferlearning/cow_face_sequences_10s_250/
```

Do not overwrite it. It is the comparability anchor for all completed experiments.

Required baseline invariants:

- Keep all current sequence folders.
- Keep all current sequence indices.
- Keep final test cows `363,403,404,408`.
- Keep current labels unchanged.
- Keep reports clear that this is `dataset_version=baseline_10s_250`.

### 5.2 Create Dense QA-Filtered Dataset

Create a new dataset later, planned as:

```text
Transferlearning/cow_face_sequences_10s_v2_dense/
```

Do not mix this dataset with baseline results. It should be a separate experiment version.

Dense sequence generation rules:

| Rule | Value |
| --- | --- |
| Candidate window length | 10 seconds |
| Candidate stride | 5 seconds |
| Target FPS | 24 |
| Stored frames per window | 240 |
| Crop size | 224 x 224 |
| YOLO confidence threshold | 0.60 |
| Crop padding | 0.08 initially; test 0.12 as an ablation only |
| Minimum detected-frame rate | >= 90 percent |
| Maximum filled-frame rate | <= 10 percent |
| Minimum mean detection confidence | >= 0.80 |
| Minimum per-window min confidence | >= 0.60 |
| Multi-face policy | largest face by area; store multi-face diagnostics |
| Video selection | all eligible videos, then cow-balanced sampling for supervised training |

Dense dataset output fields:

- `sequence_index`
- `dataset_version`
- `source_video_id`
- `dataset_root`
- `relative_path`
- `video_path`
- `cow_id`
- `cow_health_status`
- `video_health_status`
- `health_condition`
- `source_fps`
- `source_frame_count`
- `source_duration_sec`
- `sequence_seconds`
- `target_fps`
- `frames_per_sequence`
- `crop_size`
- `start_second`
- `end_second`
- `detected_frames`
- `filled_frames`
- `detection_rate`
- `filled_frame_rate`
- `mean_detection_confidence`
- `min_detection_confidence`
- `max_detection_confidence`
- `multi_face_frame_count`
- `multi_face_rate`
- `selected_box_area_mean`
- `selected_box_area_min`
- `selected_second_box_area_ratio_mean`
- `blur_laplacian_mean` if implemented
- `created_utc`

Per-frame `frames.csv` should include:

- `output_frame`
- `source_frame`
- `source_second`
- `status`
- `confidence`
- `bbox_xyxy`
- `n_detections`
- `selected_box_area`
- `second_largest_box_area`
- `selected_second_area_ratio`

### 5.3 Data QA Gates

Before training, produce and save QA tables for both baseline and dense datasets:

- rows per cow;
- rows per `video_health_status`;
- rows per `cow_health_status`;
- rows per `health_condition`;
- rows per dataset source;
- detection-rate histogram;
- filled-frame-rate histogram;
- mean-confidence histogram;
- lowest-quality 25 sequences by detection rate;
- highest-filled-frame 25 sequences;
- duplicate `relative_path` or repeated-video windows;
- cows with mixed labels;
- candidate leakage fields: date, source folder, camera/source, and condition.

Recommended exclusion/down-weighting:

- Exclude from supervised weak-label training if detected-frame rate is below 90 percent.
- Exclude or down-weight if mean confidence is below 0.80.
- Keep excluded rows in an audit file, not deleted from disk.
- For SSL, optionally include lower-quality rows only if they do not harm downstream validation; primary SSL should use QA-passing rows.

### 5.4 Label Handling

Default label for weak-label experiments:

```text
label_column = video_health_status
Healthy = 0
Unhealthy = 1
```

Rules:

- Do not train the main weak-label track on `cow_health_status`.
- Use `cow_health_status` only as an explicitly labelled ablation.
- Use `health_condition` only for subgroup summaries and vet-label sampling.
- Do not infer a numeric pain score from condition text.
- Keep cow IDs out of model inputs.
- Never random-split clips; split by cow.

### 5.5 Cow-Balanced Sampling

In the dense dataset, some cows will have more windows than others. For supervised weak-label training:

- Sample at most `K` windows per cow per epoch, where `K` is the median number of QA-passing windows per cow in the train split.
- Use replacement for cows with fewer than `K` windows if necessary.
- Keep validation and test deterministic, using all QA-passing windows for held-out cows.
- Report both sequence-level and cow-level metrics.

For SSL:

- Use all QA-passing windows from training cows.
- If training becomes dominated by high-volume cows, use cow-balanced batches.
- Do not include validation or test cows in non-transductive SSL.

## 6. Veterinary-Scored Calibration Set

### 6.1 Initial Size

Target:

- Initial set: 50 clips.
- Expansion set: up to 100 clips.

The initial 50 is enough to begin calibration and feasibility evaluation, not enough for a definitive deployable model.

### 6.2 Clip Selection Strategy

Select clips using a mixed strategy:

| Criterion | Purpose |
| --- | --- |
| High predictive entropy | find ambiguous clips |
| High model disagreement | identify uncertain transfer cases |
| Condition coverage | include lameness, mastitis, metritis, fresh cows, healthy, sudden fall |
| Cow coverage | avoid oversampling one cow |
| Source coverage | include Truro, Yashan, and Cow 349 source where appropriate |
| Camera/lighting diversity | prevent calibration to one visual setting |
| Quality coverage | mostly high-quality crops, plus a small number of borderline crops for robustness |

Do not select only high-probability positives. That would bias the calibration set.

### 6.3 Scoring Protocol

Recommended scoring design:

- At least 3 blinded veterinary scorers where possible.
- Each scorer sees randomized clips without folder labels, cow health labels, source folder names, or model predictions.
- Use an agreed scale before scoring:
  - binary pain/no-pain,
  - ordinal pain level, and/or
  - UCAPS-compatible item scores if the target video permits.
- Record scorer confidence and visibility/quality flags.
- Compute inter-rater reliability before model evaluation.
- Define the final label aggregation rule before training:
  - majority vote for binary pain,
  - median for ordinal scores,
  - consensus adjudication for high-disagreement clips.

Vet-label outputs:

- `vet_scored_manifest.csv`
- `scorer_id`
- `clip_id`
- `binary_pain`
- `ordinal_pain_score`
- `ucaps_item_scores` if used
- `visibility_quality`
- `scorer_confidence`
- `scoring_timestamp`
- `consensus_label`

## 7. Locked Experiment Pipeline

### Stage 0: Data QA

Goal: ensure the dataset is scientifically auditable before modeling.

Run for baseline and dense datasets:

- manifest summary;
- cow distribution;
- label distribution;
- condition distribution;
- detection-quality plots;
- filled-frame exclusions;
- duplicate-video/window checks;
- cow/video/session leakage checks;
- final test cow confirmation.

Acceptance:

- no train/validation/test cow overlap;
- no accidental use of final test cows in SSL;
- all metrics labelled with dataset version;
- all excluded sequences listed in an audit CSV.

### Stage 1: Source Sanity

Goal: confirm UCAPS source pain discrimination remains intact.

Run:

- UCAPS v2.9 source validation/test evaluation;
- Task1 pain/no-pain metrics;
- Task2 metrics only as source context.

Required metrics:

- Task1 AUC;
- Task1 F1;
- Task1 balanced accuracy;
- Brier/NLL/ECE if logits available.

Stopping rule:

- If source Task1 AUC is below 0.70 in an adapted run, do not interpret that run as preserved pain transfer.

### Stage 2: Baseline Target Experiments

Dataset:

```text
baseline_10s_250
```

Experiments:

| ID | Experiment | Purpose |
| --- | --- | --- |
| T0 | Zero-shot UCAPS | rank-only transfer baseline |
| T1 | Frozen-CNN weak fine-tune | test whether target head/LSTM adaptation helps |
| T2 | Full-backbone weak fine-tune | test whether full fine-tuning overfits or helps |
| T3 | Weak GCE/focal/BCE comparison | test noisy-label handling |

Split:

- final test cows fixed: `363,403,404,408`;
- V2 inner CV: 14 folds, 2 validation cows per fold;
- report 7-fold previous results separately from V2 results.

Primary interpretation:

- Proxy AUC and cow-level ranking are diagnostic.
- Thresholded classifier is not deployable if it predicts all positives or all negatives.

### Stage 3: Temporal Density and Inference Sweeps

Run on the baseline dataset first:

| Setting | Purpose |
| --- | --- |
| `max_frames=32` | current baseline |
| `max_frames=64` | more temporal coverage |
| `max_frames=96` | moderate high-density run |
| `max_frames=128` | high-density run if VRAM permits |

Batch-size guidance:

- `max_frames=32`: batch 8 or 16.
- `max_frames=64`: batch 4 or 8.
- `max_frames=96/128`: batch 2 or 4.

Sliding-window inference:

- keep training unchanged;
- at evaluation, scan multiple frame spans from the stored 240 frames;
- start with raw span 64 and stride 16;
- test aggregation by mean, trimmed mean, and max;
- choose aggregation only from validation, not final test.

### Stage 4: SSL

Main SSL:

- SimSiam on fold-train cows only.
- Initialize from UCAPS v2.9 checkpoint.
- Exclude validation and final test cows.
- Train one SSL checkpoint per fold.

Downstream comparisons:

| ID | Initialization | Downstream |
| --- | --- | --- |
| SSL-W1 | UCAPS only | weak fine-tune |
| SSL-W2 | UCAPS + SimSiam | weak fine-tune |
| SSL-D1 | UCAPS only | DANN |
| SSL-D2 | UCAPS + SimSiam | DANN |

Stopping rule:

- If SSL improves validation but not final cow-level performance, report it as representation adaptation without deployment value.

### Stage 5: DANN

Main DANN recipe:

```text
ckpt_kind = task1
source_task2_weight = 0.0
target_weak_weight = 0.0
domain_weight = 0.5 initial
source_task1_sanity_floor = 0.70
```

Sweep:

| Domain weight | Purpose |
| ---: | --- |
| 0.10 | weak alignment, preserve source |
| 0.25 | moderate alignment |
| 0.50 | current main setting |

Only test target weak BCE after source sanity is acceptable:

```text
target_weak_weight = 0.05 or 0.10
target_weak_start_epoch = 5
ramp_target_weak = true
```

Required DANN reporting:

- source Task1 AUC/F1/balanced accuracy per fold;
- source sanity pass/fail per fold;
- checkpoint selected from sanity gate or proxy fallback;
- domain accuracy;
- target validation proxy AUC;
- final sequence and cow metrics;
- calibration metrics.

Stopping rules:

- If source Task1 AUC is consistently below 0.70, reduce domain pressure or stop DANN interpretation.
- If final test remains all-positive at validation threshold, do not claim deployable classification.
- If cow-level AUC improves but balanced accuracy stays 0.5, describe it as ranking signal only.

### Stage 6: Dense Dataset Experiments

After creating `cow_face_sequences_10s_v2_dense`, repeat the practical core:

1. zero-shot on dense QA-passing windows;
2. V2 weak BCE/GCE;
3. temporal sweep;
4. SimSiam SSL;
5. Task1-only DANN with domain-weight sweep.

Report dense results separately from baseline results:

```text
dataset_version = baseline_10s_250
dataset_version = dense_10s_stride5_qa
```

Never combine rows across dataset versions in one metric calculation.

### Stage 7: Advanced Ablations

Only run after the practical core is complete:

- ADDA if DANN source sanity fails but source and target encoders may need to diverge.
- Deep Adaptation Network or MMD-style alignment if adversarial training is unstable.
- Supervised contrastive loss on UCAPS source features if Task1 source discrimination needs stronger embedding separation.
- CycleGAN/SPGAN-style appearance transfer only if semantic-preservation checks are defined.

Semantic-preservation checks for image translation:

- face crop remains anatomically plausible;
- pain-relevant facial regions are not warped;
- UCAPS source predictions do not change systematically after style transfer;
- a small manual audit is saved.

### Stage 8: Vet-Label Validation

After scoring 50 to 100 clips:

| Experiment | Purpose |
| --- | --- |
| VET-0 | zero-shot UCAPS on vet labels | direct pain-transfer baseline |
| VET-1 | frozen encoder + logistic/ordinal regression | low-variance calibration |
| VET-2 | SSL/DANN embedding + logistic/ordinal regression | test adapted embeddings |
| VET-3 | prototypical classifier | few-shot baseline |
| VET-4 | small fine-tune | only if labels are sufficient |
| VET-5 | calibrated ensemble | final candidate model |

Vet-label metrics:

- binary pain AUC;
- binary F1 and balanced accuracy;
- ordinal MAE/RMSE;
- Spearman correlation;
- weighted kappa;
- ICC or equivalent scorer-agreement metric;
- calibration ECE/Brier/NLL;
- per-cow and per-condition performance.

This is the only stage that can support target-domain pain claims.

## 8. Metrics and Reporting Rules

### 8.1 Weak-Label Track

Report:

- sequence AUC;
- cow-level AUC;
- balanced accuracy;
- F1;
- precision and recall;
- confusion table;
- validation-derived threshold;
- best validation threshold;
- Brier score;
- NLL;
- ECE;
- bootstrap 95 percent CI for cow-level metrics;
- per-cow final table;
- per-condition table;
- source dataset table.

Interpretation:

- AUC: ranking signal.
- Balanced accuracy: operating-point usefulness.
- Calibration: probability reliability.
- Cow-level metrics: primary target generalization signal.
- Sequence metrics: secondary because cows contribute different sequence counts.

### 8.2 Source Track

Report:

- UCAPS Task1 AUC;
- UCAPS Task1 F1;
- UCAPS Task1 balanced accuracy;
- UCAPS Task1 calibration if available;
- Task2 only when Task2 loss is enabled.

Interpretation:

- Source Task1 is the true pain sanity track.
- Target `video_health_status` is not a source pain sanity metric.

### 8.3 Vet-Label Track

Report:

- binary pain AUC/F1/balanced accuracy;
- ordinal metrics if ordinal scores are collected;
- scorer agreement;
- calibration;
- confidence intervals;
- error analysis with representative false positives and false negatives.

## 9. Stopping Conditions and Decision Rules

Use these rules before deciding whether to spend more GPU time.

| Situation | Decision |
| --- | --- |
| Source Task1 sanity collapses below 0.70 | reduce domain weight or stop DANN claim |
| Final threshold predicts all positives | report ranking only; not deployable |
| Inner validation improves but final cow test fails | suspect cow/context shortcuts |
| Cow AUC high on 4 cows but balanced accuracy 0.5 | describe as high-variance ranking signal |
| SSL improves validation only | keep as ablation, not final result |
| Dense dataset improves sequence AUC but not cow AUC | suspect repeated-window/cow imbalance |
| Weak labels and vet labels disagree strongly | prioritize vet-label interpretation |

## 10. Recommended Next Actions

1. **Write data QA report for baseline 250 sequences.**
   - Confirm cow counts, label counts, detection quality, filled frames, condition distribution, and final test isolation.

2. **Build dense QA-filtered sequence dataset.**
   - Use 10-second windows with 5-second stride.
   - Save the added QA fields.
   - Do not overwrite the baseline dataset.

3. **Run V2 weak-label core on baseline.**
   - 14 folds, 2 validation cows per fold.
   - Fixed test cows `363,403,404,408`.
   - BCE, focal, and GCE.
   - Max-frame sweep.
   - Sliding inference.

4. **Run Task1-only DANN source-sanity sweep.**
   - Domain weights 0.10, 0.25, 0.50.
   - No target weak BCE until source sanity is acceptable.

5. **Select 50 vet-scoring candidates.**
   - Use uncertainty, model disagreement, condition diversity, cow diversity, camera/source diversity, and crop quality.

6. **Collect veterinary labels and run the validation track.**
   - Only after this stage should the project discuss Holstein/Jersey pain detection.

## 11. Defensible Thesis Claims

Current completed evidence supports:

- a reproducible UCAPS-to-Holstein/Jersey transfer-learning pipeline;
- cow-held-out weak-label evaluation;
- target SSL and DANN implementation;
- source sanity diagnostics;
- identification of failure modes, especially threshold collapse and weak source sanity;
- a clear need for veterinary target labels.

Current completed evidence does not support:

- validated Holstein/Jersey pain detection;
- deployable clinical classifier performance;
- reliable calibrated target probabilities;
- strong cross-breed pain generalization from UCAPS alone.

Future evidence needed:

- a veterinary-scored target set;
- subject-held-out validation on target pain labels;
- source sanity preservation;
- calibrated operating thresholds that do not collapse on held-out cows;
- confidence intervals across more than four final-test cows.

## References

Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F. and Vaughan, J.W. (2010) 'A theory of learning from different domains', *Machine Learning*, 79, pp. 151-175. https://doi.org/10.1007/s10994-009-5152-4

Chen, X. and He, K. (2021) 'Exploring Simple Siamese Representation Learning', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 15750-15758. https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html

Cui, Y., Jia, M., Lin, T.Y., Song, Y. and Belongie, S. (2019) 'Class-Balanced Loss Based on Effective Number of Samples', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9268-9277. https://doi.org/10.1109/CVPR.2019.00949

Deng, W., Zheng, L., Ye, Q., Kang, G., Yang, Y. and Jiao, J. (2018) 'Image-Image Domain Adaptation With Preserved Self-Similarity and Domain-Dissimilarity for Person Re-Identification', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. https://openaccess.thecvf.com/content_cvpr_2018/html/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.html

de Oliveira, F.A., Luna, S.P.L., do Amaral, J.B., Rodrigues, K.A., Sant'Anna, A.C., Daolio, M. and Brondani, J.T. (2014) 'Validation of the UNESP-Botucatu unidimensional composite pain scale for assessing postoperative pain in cattle', *BMC Veterinary Research*, 10, Article 200. https://doi.org/10.1186/s12917-014-0200-0

Feighelstein, M., Tomacheuski, R.M., Elias, G., Shashoua, N., van der Linden, D., Luna, S.P.L. and Zamansky, A. (2026) 'Comparing the performance of deep learning video-based models and trained veterinarians in cattle pain assessment', *Scientific Reports*, 16, Article 9318. https://doi.org/10.1038/s41598-026-39604-2

Finn, C., Abbeel, P. and Levine, S. (2017) 'Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks', *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, pp. 1126-1135. https://proceedings.mlr.press/v70/finn17a.html

Gal, Y. and Ghahramani, Z. (2016) 'Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning', *Proceedings of the 33rd International Conference on Machine Learning*, PMLR 48, pp. 1050-1059. https://proceedings.mlr.press/v48/gal16.html

Gal, Y., Islam, R. and Ghahramani, Z. (2017) 'Deep Bayesian Active Learning with Image Data', *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, pp. 1183-1192. https://proceedings.mlr.press/v70/gal17a.html

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M. and Lempitsky, V. (2016) 'Domain-Adversarial Training of Neural Networks', *Journal of Machine Learning Research*, 17(59), pp. 1-35. https://www.jmlr.org/papers/v17/15-239.html

Gao, G., Ma, Y., Wang, J., Li, Z., Wang, Y. and Bai, H. (2025) 'CFR-YOLO: A Novel Cow Face Detection Network Based on YOLOv7 Improvement', *Sensors*, 25(4), Article 1084. https://doi.org/10.3390/s25041084

Gleerup, K.C.B., Andersen, P.H., Munksgaard, L. and Forkman, B. (2015) 'Pain evaluation in dairy cattle', *Applied Animal Behaviour Science*, 171, pp. 25-32. https://doi.org/10.1016/j.applanim.2015.08.023

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) 'On Calibration of Modern Neural Networks', *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, pp. 1321-1330. https://proceedings.mlr.press/v70/guo17a.html

He, K., Fan, H., Wu, Y., Xie, S. and Girshick, R. (2020) 'Momentum Contrast for Unsupervised Visual Representation Learning', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html

He, K., Chen, X., Xie, S., Li, Y., Dollar, P. and Girshick, R. (2022) 'Masked Autoencoders Are Scalable Vision Learners', *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16000-16009. https://doi.org/10.1109/CVPR52688.2022.01553

Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. and Krishnan, D. (2020) 'Supervised Contrastive Learning', *Advances in Neural Information Processing Systems*, 33. https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html

Lei, X., Wen, X. and Li, Z. (2024) 'A multi-target cow face detection model in complex scenes', *The Visual Computer*, 40, pp. 9155-9176. https://doi.org/10.1007/s00371-024-03301-w

Li, S., Fu, L., Sun, Y., Mu, Y., Chen, L., Li, J. and Gong, H. (2021) 'Individual dairy cow identification based on lightweight convolutional neural network', *PLOS ONE*, 16(11), e0260510. https://doi.org/10.1371/journal.pone.0260510

Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollar, P. (2017) 'Focal Loss for Dense Object Detection', *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2980-2988. https://doi.org/10.1109/ICCV.2017.324

Snell, J., Swersky, K. and Zemel, R. (2017) 'Prototypical Networks for Few-shot Learning', *Advances in Neural Information Processing Systems*, 30. https://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning

Tomacheuski, R.M., Oliveira, A.R., Trindade, P.H.E., Lopez-Soriano, M., Merenda, V.R., Luna, S.P.L. and Pairis-Garcia, M.D. (2024) 'Real-time and video-recorded pain assessment in beef cattle: clinical application and reliability in young, adult bulls undergoing surgical castration', *Scientific Reports*, 14, Article 15257. https://doi.org/10.1038/s41598-024-65890-9

Tzeng, E., Hoffman, J., Saenko, K. and Darrell, T. (2017) 'Adversarial Discriminative Domain Adaptation', *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7167-7176. https://openaccess.thecvf.com/content_cvpr_2017/html/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.html

Xu, B., Wang, W., Guo, L., Chen, G., Li, Y., Cao, Z. and Wu, S. (2022) 'CattleFaceNet: A cattle face identification approach based on RetinaFace and ArcFace loss', *Computers and Electronics in Agriculture*, 193, Article 106675. https://doi.org/10.1016/j.compag.2021.106675

Zhang, Z. and Sabuncu, M. (2018) 'Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels', *Advances in Neural Information Processing Systems*, 31. https://papers.nips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networks-with-noisy-labels

Zhu, J.Y., Park, T., Isola, P. and Efros, A.A. (2017) 'Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks', *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2223-2232. https://doi.org/10.1109/ICCV.2017.244
