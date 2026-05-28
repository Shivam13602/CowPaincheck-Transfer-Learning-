
Short executive summary
You have a small-source (UCAPS — Brazilian Nelore/Angus bulls undergoing acute castration pain) → small-target (Canadian Holstein/Jersey dairy cows with chronic pain labels/conditions) transfer problem, plus large breed- and phenotype-driven domain shift and very few labeled target examples (≈13–20). The best practical strategy is multi-stage transfer + domain adaptation + label-efficiency methods: (1) strong preprocessing & landmark-based features, (2) pretrain encoders on UCAPS (source) and self-supervise on unlabeled target video, (3) use a combination of adversarial domain adaptation (DANN/ADDA), image/appearance translation (CycleGAN/SPGAN), and few-shot/meta/few-shot metric methods (Prototypical / MAML-like), (4) calibrate/obtain a small set of vet-scored target samples via active learning for pain-score mapping and ordinal regression, and (5) rigorously evaluate with uncertainty estimation and bootstrapped cross-validation. Below I give an end-to-end pipeline, concrete methods to try, and peer-reviewed papers that support each choice. Key supporting literature: UCAPS validation and cattle grimace literature, DANN/ADDA (domain adversarial), CycleGAN/SPGAN for appearance transfer, self-supervised pretraining for small datasets, and few-shot/meta-learning. (PubMed Central)

End-to-end pipeline (practical, experiment-ready)
1. Data & labelling — make the target labels usable
Gather raw video metadata (timestamp, camera, distance, stall/field, lighting) and keep a per-video manifest.


Weak labels on target: existing clinical conditions (lameness, mastitis) are weak proxies for pain presence but not equal to UCAPS continuous scores. Treat them as binary/weak pain labels initially. Use them for semi-supervised signals (see Step 5). (PubMed Central)


Collect a small expert-scored calibration set: ask 3 veterinary experts to score ~30–100 short 10s clips (spread across breeds, lighting, poses) using UCAPS or an ordinal pain scale. These will be used to calibrate/regress pain scores and for evaluation. Even 30–50 vet-scored clips provides huge value (active learning picks the most informative). (See strategy in active learning & few-shot literature.) (PubMed Central)



2. Preprocessing — robust, breed-agnostic face/landmark extraction
Frame extraction: sample at a fixed frame rate (e.g., 10–15 fps) for your 10s sequences (→ 100–150 frames).


Face detection & crop: use a robust detector (YOLOv8/YOLOv7 or a specialized animal-face model). Train/finetune the detector on UCAPS face bounding boxes, then adapt to Canadian images via a small fine-tuning set + synthetic style transfer (below). A YOLO-pose variant can give landmarks directly. Recent preprints show YOLOv8-Pose or custom YOLO pose nets work well for animal faces. (Preprints)


Landmark detection (26-keypoints): you already use 26 landmarks — continue but make them robust: train a keypoint detector (or finetune a pretrained animal-landmark model) on UCAPS and augment target via image translation/augmentation. Literature shows landmark-driven pipelines generalize well when you use shape-normalization (Procrustes) and geometric features (distances/angles). (MDPI)


Geometric features: compute normalized distances, angles, ratios and PCA of landmark shapes; also compute per-frame local patch deep features (see Step 3). Using both geometric and learned deep features is beneficial (hybrid). (MDPI)



3. Feature extraction (spatial + temporal)
Spatial encoder(s):


A: landmark-driven geometric vector (26 landmarks → vector of normalized coords + engineered features).


B: local patch CNN descriptor: crop eye/nostril/muzzle patches and encode with a pretrained MobileNet/ResNet (pretrained on ImageNet then fine-tuned). Recent cattle micro-expression work uses patch encoders stacked into descriptors. (Preprints)


Temporal modelling:


Concatenate per-frame feature vectors into 10s sequences (e.g., 100×D).


For sequence model: try Bi-LSTM and Transformer or Temporal Convolutional Network (TCN) as options (LSTM is fine; consider Bi-LSTM and a 1D-TCN/Transformer in ablation). Also experiment with 3D CNN or 2-stream 3D CNN if raw pixels are used. For your plan, LSTM/BiLSTM + landmark+patch features is a strong baseline. (MDPI)



4. Pretraining on source (UCAPS) + self-supervised on target
Supervised pretrain: train the spatial encoder + sequence model on UCAPS (source) to predict UCAPS pain scores/timepoints (the source task). This gives a strong starting point for pain-relevant features. UCAPS has demonstrated reliable video-based scoring for castration pain in bulls — so it’s appropriate as a base. (PubMed Central)


Self-supervised pretraining on unlabeled target video: use SimCLR/MoCo-style contrastive learning or masked image modeling (MAE) on unlabeled Canadian cow frames (and video augmentations) to adapt visual features to the target domain before fine-tuning. Multiple studies show self-supervision drastically improves transfer when labeled target data are scarce. (Nature)



5. Transfer strategies (combine several, evaluate)
Because your domain gap involves breed anatomy and acute → chronic pain expression differences, use multiple complementary transfer/adaptation methods and ablate:
A. Simple fine-tuning baseline
Freeze early layers of the encoder trained on UCAPS, fine-tune later layers + LSTM on whatever labeled target you have (weak labels for presence and the small vet-scored calibration set). Evaluate as baseline.


B. Unsupervised Domain Adaptation (feature alignment)
DANN (Domain Adversarial Neural Network): add a domain classifier with a gradient reversal layer so the encoder learns domain-invariant features. Works well when you have labeled source + unlabeled target. Ganin et al. is canonical. (Journal of Machine Learning Research)


ADDA: alternative adversarial discriminative adaptation; decouples source and target encoders if that helps. (Open Access CVF)


C. Image-level appearance transfer (augment source)
Use CycleGAN / SPGAN / task-aware CycleGAN to translate UCAPS source images into Canadian-like appearance while preserving pain semantics, then re-train/finetune on synthetic images. This reduces superficial appearance gaps (lighting, hair patterns, muzzle coloration). Several FER works use CycleGAN variants to generate domain-matched images for training. Be careful to preserve semantic facial muscle cues (use cycle + self-supervision / content preservation losses). (CIBM)


D. Metric–few-shot / Meta-learning
If target vet-scored examples remain very few, try Prototypical Networks (metric-based) or MAML-style meta-learning to make the model quick to adapt with few shots. Prototypical approaches are strong for few-shot classification; MAML helps for rapid fine-tuning for regression/ordinal tasks. (arXiv)


E. Pseudo-labeling + multi-task learning
Use weak condition labels (lameness/mastitis) as auxiliary tasks (pain_proxy classification). Train multi-task network: (i) pain regression/ordinal (from source / vet labels), (ii) condition classification (weak target labels), and (iii) domain discriminator (for DANN). This uses all available signal and has shown value in low-label domains. (PubMed Central)


F. Shape/landmark domain alignment
Align shapes via Procrustes/shape normalization and use landmark-only models that are less sensitive to fur/texture differences. Cross-species feature learning literature suggests learning equivariant cross-species features improves transfer. (ECVA)



6. Producing a pain-score for Canadian cows (practical recipe)
Train a hybrid model on UCAPS (source) → encoder + LSTM that predicts UCAPS continuous score.


Adapt encoder with DANN using unlabeled Canadian frames (domain adversarial training); at the same time fine-tune on the small vet-scored target set (multi-task loss: source supervised + vet calibration supervised + domain adversarial + weak-label classification). Use an ordinal regression loss (or adapted regression with monotonic penalties) because pain scores are ordinal. (Journal of Machine Learning Research)


If only binary weak labels exist for most target samples, use them for a coarse calibration (pain/no-pain). Use the small vet-scored set to calibrate the continuous mapping (e.g., isotonic regression from model output → UCAPS scale).


If vet-scored labels are too few, use prototypical/few-shot: compute prototypes per pain level from vet clips and classify/score target clips by distance to prototypes. (arXiv)



7. Evaluation, uncertainty & deployment
Metrics: use MAE/RMSE for continuous scores, Spearman’s rho, intra-class correlation (ICC) and weighted Cohen’s kappa for ordinal agreement vs vets; AUC/precision for binary detection. Report confidence intervals via bootstrapping. (PubMed Central)


Uncertainty estimation: use MC Dropout or ensembles to flag predictions with high epistemic uncertainty for vet review. Important in small-target settings.


Cross-validation: leave-one-farm / leave-one-cow / subject-wise folds to avoid overoptimistic results.


Active learning loop: prioritize high-uncertainty or high-disagreement clips for vet scoring to maximally improve calibration with minimum labeling cost.



8. Practical tips, augmentations & engineering
Data augmentation: photometric (lighting), geometric (rotation, small scale), occlusion (simulate partial face), and style transfer (texture) are critical. Literature shows style-transferred images improve landmark robustness. (arXiv)


Synthetic data: generate extra target-style images via CycleGAN/SPGAN while preserving facial motion semantics (use content and AU-preserving losses). Carefully inspect outputs to ensure pain cues are not destroyed. (CIBM)


Balance: chronic vs acute expressions may differ in magnitude; consider contrastive pretraining to emphasize subtle temporal patterns. (Nature)



Suggested experimental plan (short timeline of experiments)
Baseline: UCAPS-pretrain → freeze CNN → train Bi-LSTM on UCAPS; test zero-shot on Canadian target (report baseline). (PubMed Central)


Fine-tune: fine-tune last layers on small vet target set (if available).


DANN: train source supervised + domain adversarial with unlabeled target. Compare to baseline. (Journal of Machine Learning Research)


CycleGAN augmentation: create style-transferred UCAPS images → retrain/finetune model. Compare. (CIBM)


Self-supervised pretrain: pretrain encoder on unlabeled target frames (SimCLR/MAE) then finetune. Compare. (Nature)


Few-shot / Prototypical: run prototypical networks on vet-scored clips. (arXiv)


Ensemble + calibration: combine best models and calibrate outputs vs vet set.



Key peer-reviewed papers & resources (starter bibliography)
Tomacheuski RM et al. Real-time and video-recorded pain assessment in beef cattle (UCAPS validation) — Scientific Reports / PMC. Shows UCAPS applicability to video-recorded castration pain in bulls. (PubMed Central)


Ganin Y. & Lempitsky V., Domain-Adversarial Training of Neural Networks (DANN) — foundational DA method; use for learning domain-invariant features. (Journal of Machine Learning Research)


Tzeng E. et al. ADDA: Adversarial Discriminative Domain Adaptation — alternative adversarial adaptation. (Open Access CVF)


Bozorgtabar B. et al., Adversarial domain adaptation for facial expression analysis — shows GAN/translation + adversarial methods for FER. (CIBM)


Wolf D., et al. Self-supervised pre-training for small datasets (medical imaging) — review and evidence that self-supervision helps small labeled target datasets. (Nature)


Snell J., Swersky K., Zemel R., Prototypical Networks for Few-shot Learning — metric method for few-shot classification. (arXiv)


Shi X., Deep Cross-species Feature Learning for Animal Face (ECCV 2020) — cross-species feature learning to improve interspecies transfer. Useful for breed morphology variation. (ECVA)


Müller BR., et al. Facial expression of pain in Nellore and crossbred beef cattle — facial action units and grimace scales in beef cattle. (ScienceDirect)


Martvel G., Automated landmark-based cat facial analysis / Cat pain recognition papers — examples of landmark-based pain recognition in animals and methodology transferable to cattle. (Frontiers)


(If you want, I can compile a formatted reference list in Harvard style with DOIs and direct links for each paper above — you previously asked for Harvard style in other work, I can do that next.)

Final practical recommendations (what to run first)
Create the small vet-scored calibration set now (30–50 short clips). This one action will unlock proper evaluation and calibration.


Pretrain encoder on UCAPS (supervised) and self-supervise on all unlabeled Canadian frames (SimCLR/MAE) — these two plus DANN adaptation are the highest ROI. (PubMed Central)


Run a quick pilot: baseline fine-tune vs DANN vs CycleGAN-augmented training and compare on your calibration set. Use uncertainty and active learning to select further vet labels.



If you want, I’ll do any of the following next (pick one and I’ll implement it immediately in this turn):
(A) Produce a prioritized experiment table with hyperparameters and exact code/sketch (PyTorch) for the encoder + Bi-LSTM baseline + DANN.


(B) Compile the full Harvard-style bibliography for all cited papers (with DOIs).


(C) Draft a short active-learning script + selection strategy to pick the next 50 clips for vet scoring.


(D) Produce a concrete model architecture and training schedule you can paste into a training script.


Pick one and I’ll output it right away.

