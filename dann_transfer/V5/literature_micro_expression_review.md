# V5 Literature Review — Facial Micro-Expression Pain in Cattle: Eyebrow (Orbital) Tightening and Muzzle Tension

This review focuses on the specific facial cues the V5 strategy targets: **tightening around the eye/eyebrow (orbital tightening, tension above the eye)** and **muzzle tension (nostril dilation, chewing-muscle straining, mouth strain)**, and on how video-based micro-expression analysis and region-specific deep learning use these cues. It complements the broader project review in [`../../docs/literature_review.md`](../../docs/literature_review.md) and motivates the V5 experiment ladder in [`README.md`](README.md) and [`experiment_matrix.md`](experiment_matrix.md).

> **Framing.** Our Holstein/Jersey labels are **disease-condition labels, not veterinary pain scores** (`video_health_status` Healthy/Unhealthy and `health_condition`: lameness, mastitis, metritis, fresh cows, sudden fall, healthy). This review explains *which facial signals carry pain information and how to model them*; it does not convert condition labels into pain labels. Because no vet/grimace pain labels are available, V5 reframes the goal as **disease-context discrimination** and uses **lameness — the most facial-pain-relevant condition — as a focused proxy** (stage S10, Track B). This is the closest-to-pain signal obtainable from existing labels; it remains a condition proxy, not a pain score.

---

## 1. Why facial micro-expressions carry pain information

Across mammals, pain produces stereotyped facial muscle contractions that are **reflexive and difficult to suppress voluntarily**. The biological basis is that nociceptive input propagates through cranial nerves **V (trigeminal)** and **VII (facial)**, driving contractions in the **periocular, perinasal, and perioral** musculature within milliseconds of a noxious stimulus (Zhang, Sailunaz and Neethirajan, 2025). Because cattle are prey animals that evolved to mask overt distress, their pain signals are **brief, low-amplitude, and easily missed** by behaviour scoring or sensors — which is exactly why frame-level facial micro-expression analysis is attractive for on-farm triage (Zhang, Sailunaz and Neethirajan, 2025).

This matters for our project for two reasons:

1. The most informative cues are **small, localized, and transient** — so whole-face, sparsely-sampled models can dilute them. This argues for region emphasis (eye, muzzle) and denser/attentive temporal modeling.
2. Reflexive cues are **less context-dependent than posture or behaviour**, which is helpful when transferring across breed (beef → dairy) and context (castration → herd-health), the central difficulty in V1–V4.

---

## 2. The cattle pain face: which action units, and how responsive

### 2.1 Calf Grimace Scale (CGS)

The CGS (Scientific Reports, 2024; 69 Angus calves, castration vs sham) formalizes **six facial action units (FAUs)**:

1. **Ear position**
2. **Orbital tightening** (narrowing/closure around the eye)
3. **Tension above the eye** (the bovine analog of "eyebrow tightening")
4. **Nostril dilation**
5. **Straining of the chewing muscle**
6. **Mouth opening**

CGS scores rose significantly after castration and the scale could distinguish pain from non-pain stress. Critically for V5, the FAUs with the **greatest responsiveness to acute pain were ear position, orbital tightening, and nostril dilation**. Two of the user's target cues map directly onto the most responsive units: orbital tightening / tension-above-the-eye ("eyebrow tightening") and nostril dilation ("muzzle tension").

### 2.2 Cross-species convergence

The same periocular and perinasal/perioral units recur across species' validated grimace scales:

- **Mouse / Rat Grimace Scale:** orbital tightening, nose/cheek bulge, ear and whisker changes (Langford et al., 2010; Sotocinal et al., 2011).
- **Horse Grimace Scale / EquiFACS:** orbital tightening, tension above the eye, strained nostrils and muzzle, chin/lip changes (Dalla Costa et al., 2014; Wathan et al., 2015).
- **Sheep / Lamb Pain Face:** orbital tightening, cheek tightening, abnormal ear and lip/jaw posture (McLennan et al., 2016).

This convergence is why a beef-trained model can plausibly transfer *some* periocular/perinasal signal to dairy cows — the muscle groups and their pain responses are homologous — and why region emphasis on eye + muzzle is a defensible inductive bias.

### 2.3 Reliability caveat — do not bet on one cue

A systematic review of facial-expression scoring in large domestic animals (Müller et al., 2022, "Do not look at me like that") found that **inter-observer reliability varies sharply by FAU**: ear position is typically easy and highly reliable (ICC ~0.81–1.0), whereas **orbital tightening and "tension above the eye" are frequently rated "not able to score"** or show only moderate agreement, especially when the cue is mild, the animal is young, or the painful condition is chronic. The review's recommendation is explicit: **weight more-reliable units and use composite (multi-FAU) scores** rather than relying on a single hard-to-score cue.

**V5 implication:** a region-aware model should **fuse eye + muzzle (+ ear) evidence**, not depend on the eyebrow cue alone. This directly informs the S9 region-fusion design and argues against a single-region model.

---

## 3. Temporal dynamics: pain is in the change, not one frame

Equine work analyzing the facial repertoire across orthopedic-pain intensities (Rashid et al., 2024) identified ~16 AUs/action descriptors predictive of pain and showed that **most facial change occurs in the transition from no pain to mild pain**, and that pain- and stress-related activities can co-occur and confound a single-frame read. The authors conclude that **"one prototypical pain face" is a simplification** — the signal is a dynamic, asymmetric repertoire over time.

**V5 implication:** prefer **temporal modeling that captures change and bursts** over single-frame or final-LSTM-state readouts:

- Attention-based temporal pooling over the sequence (S7) rather than only the LSTM final hidden state.
- Sliding-window inference with burst-sensitive aggregation (mean / trimmed-mean / max), chosen on validation only (S7/S8).
- These match the "micro-burst counting" aggregation that improved precision in the dairy-cattle micro-expression pipeline below.

---

## 4. Region-specific deep learning for facial pain

### 4.1 Dairy-cattle micro-expression pipeline (most directly relevant)

Zhang, Sailunaz and Neethirajan (2025), *AI* 6(9):199 — **from Dalhousie University, Truro, NS, the same farm/region as our Truro Cow Video Data** — is the closest published analog to our target domain. Their two-stage pipeline:

1. **YOLOv8-Pose** detects the cow face and **30 facial landmarks** (mAP@0.50 = 96.9% detection; OKS = 83.8% keypoints).
2. **Cropped eye, ear, and muzzle patches** are encoded with a pretrained **MobileNetV2** into 3840-d descriptors that "capture millisecond muscle twitches".
3. **5-frame sequences → 128-unit LSTM** outputs pain probability.
4. A **hybrid aggregation rule** combines a 30% mean-probability threshold with **micro-burst counting** to temper false alarms.

Results: 99.65% accuracy / F1 0.997 on a held-out validation set of 1700 frames, and **64.3% clip-level accuracy with 83% pain-class precision on 14 unseen barn videos** — explicitly framed as an early proof-of-concept. Named future work: **attention-based temporal pooling, curriculum learning for variable window lengths, domain-adaptive fine-tuning, and multimodal fusion**.

Why this is the key reference for V5:

- It validates **region cropping of eye + muzzle (+ ear)** as the way to expose orbital tightening and nostril/mouth tension to a CNN, instead of feeding a whole face crop.
- Its drop from frame-level to unseen-video accuracy (99.7% → 64.3%) mirrors **our inner-val vs held-out-test gap** and reinforces the V5 emphasis on subject-held-out evaluation and honest CIs.
- Its future-work list (domain-adaptive fine-tuning, attention pooling) is essentially the V5 ladder — we can position V5 as executing those directions on a transfer-from-UCAPS backbone.

### 4.2 Region masking and attention in other species

- **Horse welfare via DL (Kim et al., 2023, *Veterinary Sciences*):** the authors **isolate the eyes, nose, and ears and mask the rest of the face in black**, on the explicit grounds that "most signals come from some regions of the face, such as the mouth and eyes, while others (ears, hair) play little role" and that the network "should only focus on important parts." Orbital constriction / partially closed eyes and fixed gaze are highlighted as pain cues (with the caveat that eye closure can also reflect fatigue/learned helplessness — another reason to fuse cues and use temporal context).
- **Rat Grimace Scale attention heatmaps (Vis. of RGS, VT thesis):** region-specific models with attention show the **eye model attends to the eye shape, consistent with how human graders assess orbital tightening**, validating that region-restricted models learn the intended AU rather than spurious background.
- **Open-Sheep-Face (Feng, Karaskova and Mahmoud, 2023):** crops ear/eye/mouth regions and extracts HOG + geometric features (ear angles, landmark distances) for pain estimation, again confirming the region-decomposition recipe.

### 4.3 Source pain model context (UCAPS)

Our source backbone is grounded in the **UNESP-Botucatu Cattle Pain Scale (UCAPS)** (de Oliveira et al., 2014; Tomacheuski et al., 2024), validated for acute castration pain in beef cattle. UCAPS is a behaviour-plus-facial composite, not a pure FAU detector, but the recent automated-vs-veterinarian study (Feighelstein et al., 2026) shows deep video models can approach trained-veterinarian pain assessment **when trained and evaluated against validated pain labels with subject-held-out splits**. V5 cannot make that pain claim (no pain labels), but it adopts the same subject-held-out rigor for its disease-context and lameness-vs-healthy tracks; a future validated pain-scored set would be required for a pain claim.

---

## 5. Concrete implications for the V5 experiment ladder

| Finding | V5 design choice | Stage |
|---------|------------------|-------|
| Orbital tightening + nostril/mouth tension are the most pain-responsive cues | Bias the whole-face model toward eye + muzzle; build explicit region crops in the stretch arm | S3, S7, S9 |
| Single FAUs (esp. eyebrow) are unreliable; composites are reliable | **Fuse** eye + muzzle (+ ear) regions; never a single-region model | S9 |
| Pain is in the temporal change / bursts, not one frame | Attention-based temporal pooling; burst-sensitive sliding-window aggregation | S7, S8 |
| Region crops + temporal model beat static whole-face (Dalhousie/Truro) | Region-encoder + temporal-attention stretch arm; cite as the on-domain precedent | S9 |
| Region masking focuses the net and avoids background shortcuts | Mask non-region pixels / use landmark-cropped patches; audit attention maps | S9 |
| Frame→video accuracy drop mirrors our val→test gap | Keep subject-held-out 8-cow test + bootstrap CIs; treat inner val as optimistic | S0–S8 |
| Domain-adaptive fine-tuning is the named next step on-domain | DANN/CORAL/CDAN with source retention gating | S4, S5 |
| Eye closure / fixed gaze is ambiguous (pain vs fatigue) | Use temporal context + multi-region fusion; do not over-claim from one cue | S7, S9 |

The eyebrow/muzzle focus does **not** require abandoning the UCAPS CNN-LSTM-attention backbone. Two integration paths:

1. **Whole-face, region-biased (low cost, S3/S7):** keep the current face crop but (a) add spatial attention supervision toward periocular/perinasal regions if landmarks are available, and (b) switch temporal aggregation to attention pooling. No new data pipeline.
2. **Explicit region pipeline (stretch, S9):** add a cow facial-keypoint detector (e.g., YOLOv8-Pose as in the Dalhousie work), crop eye + muzzle (+ ear) patches per frame, encode each region, and fuse with temporal attention. Higher cost (new keypoint infrastructure) but directly targets orbital tightening and muzzle tension.

---

## 6. References

Dalla Costa, E., Minero, M., Lebelt, D., Stucke, D., Canali, E. and Leach, M.C. (2014) 'Development of the Horse Grimace Scale (HGS) as a pain assessment tool in horses undergoing routine castration', *PLOS ONE*, 9(3), e92281. https://doi.org/10.1371/journal.pone.0092281

de Oliveira, F.A., Luna, S.P.L., do Amaral, J.B., Rodrigues, K.A., Sant'Anna, A.C., Daolio, M. and Brondani, J.T. (2014) 'Validation of the UNESP-Botucatu unidimensional composite pain scale for assessing postoperative pain in cattle', *BMC Veterinary Research*, 10, Article 200. https://doi.org/10.1186/s12917-014-0200-0

Feighelstein, M., Tomacheuski, R.M., Elias, G., Shashoua, N., van der Linden, D., Luna, S.P.L. and Zamansky, A. (2026) 'Comparing the performance of deep learning video-based models and trained veterinarians in cattle pain assessment', *Scientific Reports*, 16, Article 9318. https://doi.org/10.1038/s41598-026-39604-2

Feng, Z., Karaskova, M. and Mahmoud, M. (2023) 'Open-Sheep-Face: A comprehensive application for sheep face analysis and pain estimation', *ACII Workshops 2023*. https://doi.org/10.1109/ACIIW59127.2023.10388128

Kim, S., et al. (2023) 'Analysis of various facial expressions of horses as a welfare indicator using deep learning', *Veterinary Sciences*, 10(4), 283. https://doi.org/10.3390/vetsci10040283

Langford, D.J., Bailey, A.L., Chanda, M.L., et al. (2010) 'Coding of facial expressions of pain in the laboratory mouse', *Nature Methods*, 7, pp. 447–449. https://doi.org/10.1038/nmeth.1455

McLennan, K.M., Rebelo, C.J.B., Corke, M.J., Holmes, M.A., Leach, M.C. and Constantino-Casas, F. (2016) 'Development of a facial expression scale using footrot and mastitis as models of pain in sheep', *Applied Animal Behaviour Science*, 176, pp. 19–26. https://doi.org/10.1016/j.applanim.2016.01.007

Müller, B.R., et al. (2022) '"Do not look at me like that": Is the facial expression score reliable and accurate to evaluate pain in large domestic animals? A systematic review', *Frontiers in Veterinary Science* / PMC9763617. https://pmc.ncbi.nlm.nih.gov/articles/PMC9763617/

Rashid, M., et al. (2024) 'Changes in the equine facial repertoire during different orthopedic pain intensities', *Scientific Reports* / PMC10762010. https://pmc.ncbi.nlm.nih.gov/articles/PMC10762010/

Sotocinal, S.G., Sorge, R.E., Zaloum, A., et al. (2011) 'The Rat Grimace Scale: a partially automated method for quantifying pain in the laboratory rat via facial expressions', *Molecular Pain*, 7, 55. https://doi.org/10.1186/1744-8069-7-55

Tomacheuski, R.M., Oliveira, A.R., Trindade, P.H.E., Lopez-Soriano, M., Merenda, V.R., Luna, S.P.L. and Pairis-Garcia, M.D. (2024) 'Real-time and video-recorded pain assessment in beef cattle: clinical application and reliability in young, adult bulls undergoing surgical castration', *Scientific Reports*, 14, Article 15257. https://doi.org/10.1038/s41598-024-65890-9

Wathan, J., Burrows, A.M., Waller, B.M. and McComb, K. (2015) 'EquiFACS: The Equine Facial Action Coding System', *PLOS ONE*, 10(8), e0131738. https://doi.org/10.1371/journal.pone.0131738

(Calf Grimace Scale) (2024) 'Development of the calf grimace scale for pain and stress assessment in castrated Angus beef calves', *Scientific Reports*, 14. https://www.nature.com/articles/s41598-024-77147-6

Zhang, S., Sailunaz, K. and Neethirajan, S. (2025) 'Micro-expression-based facial analysis for automated pain recognition in dairy cattle: an early-stage evaluation', *AI*, 6(9), 199. https://doi.org/10.3390/ai6090199

> Note: author lists and years for a few entries were captured from search-result metadata and the project's existing reference list; verify exact author strings and DOIs against the publisher pages before final thesis submission. The CGS, Müller et al., Rashid et al., and Zhang et al. entries are the load-bearing citations for V5 and should be confirmed first.
