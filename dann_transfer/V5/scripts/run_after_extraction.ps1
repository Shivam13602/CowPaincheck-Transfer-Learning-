# Run after Step 1 extraction finishes (completed_manifest.csv exists).
# Step 2: validate metadata + regenerate v5_split.json on the enlarged manifest.
$ErrorActionPreference = "Stop"
$Root = "C:\Users\shivp\Downloads\Research\DATASET"
$Out = Join-Path $Root "Transferlearning\cow_face_sequences_thesis_stride8_v5\output"
$Manifest = Join-Path $Out "completed_manifest.csv"

if (-not (Test-Path $Manifest)) {
    Write-Error "Missing $Manifest — wait for create_thesis_stride8_sequences.py to finish."
}

Set-Location $Root
python CowPaincheck-Transfer-Learning/datasets/thesis_stride8_qa/validate_thesis_metadata.py $Out

Set-Location (Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer\V5")
python make_v5_splits.py `
  --manifest $Manifest `
  --out splits/v5_split.json

Write-Host ""
Write-Host "Step 2 done. Confirm printed summary: 8 test cows, 5x4 folds, train_only includes 349."
Write-Host "Next: scripts/upload_to_rorqual.ps1 then sbatch on Rorqual (see RUNBOOK.md Step 4)."
