# GPU-optimized thesis_stride8_v5 extraction (sequential decode + FP16 + larger YOLO batches).
# Tuned for ~4GB laptop GPUs: batch 48. On 8GB+ use --yolo-batch-size 96 or 128.
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\shivp\Downloads\Research\DATASET"

$YoloBatch = if ($env:YOLO_BATCH_SIZE) { $env:YOLO_BATCH_SIZE } else { "48" }

python CowPaincheck-Transfer-Learning/datasets/thesis_stride8_qa/create_thesis_stride8_sequences.py `
  --inventory cow_video_dataset_analysis.csv `
  --dataset-root . `
  --model yolo_cow_face/yolo26s.pt `
  --videos-per-cow 4 `
  --stride-seconds 8 `
  --seed 42 `
  --output Transferlearning/cow_face_sequences_thesis_stride8_v5/output `
  --overwrite `
  --device 0 `
  --yolo-batch-size $YoloBatch `
  --yolo-imgsz 640 `
  --yolo-half `
  --jpeg-quality 90
