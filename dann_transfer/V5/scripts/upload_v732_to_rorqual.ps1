# Upload enlarged 732-seq dataset + v5_split_v732 + code to Rorqual.
# LARGE: sequences folder - use rsync from WSL/Git Bash if scp is too slow.
#
# Usage: powershell -ExecutionPolicy Bypass -File upload_v732_to_rorqual.ps1
# Skip dataset if already on scratch: -SkipDataset

param([switch]$SkipDataset)

$ErrorActionPreference = "Stop"
$Root = "C:\Users\shivp\Downloads\Research\DATASET"
$Remote = "shiv136@rorqual.alliancecan.ca"
$Scr = "/scratch/shiv136/project_data"
$OutLocal = Join-Path $Root "Transferlearning\cow_face_sequences_thesis_stride8_v5\output"

ssh $Remote "mkdir -p ${Scr}/cow_face_sequences_thesis_stride8_v5/output ${Scr}/v5/splits ~/Dann_transfer/dann_transfer/V5/scripts"

if (-not $SkipDataset) {
    Write-Host "Uploading manifest + CSVs (small)..."
    scp "$OutLocal\completed_manifest.csv" "$OutLocal\processing_statistics.json" "${Remote}:${Scr}/cow_face_sequences_thesis_stride8_v5/output/"
    Write-Host "Uploading sequences (LARGE - consider WSL rsync)..."
    scp -r "$OutLocal\sequences" "${Remote}:${Scr}/cow_face_sequences_thesis_stride8_v5/output/"
} else {
    Write-Host "SkipDataset: assuming sequences already on scratch."
}

Write-Host "Uploading v5_split_v732.json + fixed scripts..."
$Split = Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer\V5\splits\v5_split_v732.json"
$Scripts = Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer\V5\scripts"
scp $Split "${Remote}:${Scr}/v5/splits/"
scp "$Scripts\run_v5_matrix_rorqual.sh" "$Scripts\run_v5_matrix_rorqual_v732.sh" "$Scripts\sbatch_v5_matrix_rorqual_v732.sh" "${Remote}:~/Dann_transfer/dann_transfer/V5/scripts/"

Write-Host ""
Write-Host "On Rorqual run:"
Write-Host "  sed -i 's/\r$//' ~/Dann_transfer/dann_transfer/V5/scripts/*.sh"
Write-Host "  sbatch ~/Dann_transfer/dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_v732.sh"
