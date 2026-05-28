# Holstein/Jersey zero-shot (UCAPS v2.9)

- Generated (UTC): `20260502T051222Z`
- Checkpoints: `/content/drive/MyDrive/facial_pain_project_v2/checkpoints_v2.9/v2.9_20260222_144752` (`ckpt_kind=task2`)
- Folds used: `[0, 1, 2, 3, 4, 5, 6, 7, 8]`
- Sequences: 250
- Predictions CSV: `holstein_zero_shot_predictions_20260502T051222Z.csv`

## Pain probability by video-level health context

| video_health_status | n | pain_prob_mean | pain_prob_std |
| --- | --- | --- | --- |
| Healthy | 123 | 0.43058302465493115 | 0.04908225202040817 |
| Unhealthy | 127 | 0.4367499515766234 | 0.055900539449718026 |

## Pain probability by cow-level health label

| cow_health_status | n | pain_prob_mean | pain_prob_std |
| --- | --- | --- | --- |
| Healthy | 105 | 0.42945420543352764 | 0.045423952824611584 |
| Unhealthy | 145 | 0.4368018228432228 | 0.057272012277549716 |

## Pain probability by disease/proxy condition label

| health_condition | n | pain_prob_mean | pain_prob_std |
| --- | --- | --- | --- |
| fresh cows | 7 | 0.42391308716365267 | 0.046311029447607434 |
| healthy | 38 | 0.4279092565963143 | 0.05432452275103958 |
| healthy folder | 85 | 0.43177835625760697 | 0.04649836705586638 |
| lame | 18 | 0.43126127786106533 | 0.054814971149582795 |
| lameness | 55 | 0.4294567449526353 | 0.061358127741539564 |
| lameness/stiffness | 11 | 0.446910099549727 | 0.03643453711482932 |
| possible mastitis | 21 | 0.44137319638615563 | 0.0493797089917449 |
| possible metritis | 10 | 0.47638080418109896 | 0.04286525238468361 |
| sudden fall | 4 | 0.44517144560813904 | 0.05457011746438251 |
| unhealthy folder | 1 | 0.38768622279167175 | 0.0 |

## Pain probability by dataset root

| dataset_root | n | pain_prob_mean | pain_prob_std |
| --- | --- | --- | --- |
| Cow 349 - Unhealthy (sudden fall) | 4 | 0.44517144560813904 | 0.05457011746438251 |
| Truro Cow Video Data | 148 | 0.4302456695083025 | 0.05251655716284714 |
| Yashan Dhaliwal RAC Data 2025 | 98 | 0.4384888878890446 | 0.05257807017175168 |
