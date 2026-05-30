# Step 3 — upload enlarged dataset, split JSON, checkpoints, and code to Rorqual scratch.
# Requires: SSH key + Alliance MFA session (interactive login once in a separate terminal).
# Usage: powershell -ExecutionPolicy Bypass -File upload_to_rorqual.ps1

$ErrorActionPreference = "Stop"
$Root = "C:\Users\shivp\Downloads\Research\DATASET"
$Acct = "shiv136"
$Host = "${Acct}@rorqual.alliancecan.ca"
$Scr = "/scratch/${Acct}/project_data"

$Paths = @{
    Dataset = Join-Path $Root "Transferlearning\cow_face_sequences_thesis_stride8_v5\output"
    Splits  = Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer\V5\splits"
    Ckpt    = Join-Path $Root "remote_upload_ucaps\Transferlearning\v2.9\v2.9_20260502_181533-20260528T182044Z-3-001\v2.9_20260502_181533"
    Code    = Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer"
}

foreach ($key in @("Dataset","Splits","Ckpt")) {
    if (-not (Test-Path $Paths[$key])) {
        Write-Error "Missing ${key}: $($Paths[$key])"
    }
}

Write-Host "Creating remote dirs..."
ssh $Host "mkdir -p ${Scr}/cow_face_sequences_thesis_stride8_v5/output ${Scr}/v5/splits ${Scr}/v2.9_20260502_181533 ${Acct}/Dann_transfer/dann_transfer"

Write-Host "Uploading enlarged dataset (this is large; may take a long time)..."
scp -r "$($Paths.Dataset)\*" "${Host}:${Scr}/cow_face_sequences_thesis_stride8_v5/output/"

Write-Host "Uploading V5 splits..."
scp -r "$($Paths.Splits)\*" "${Host}:${Scr}/v5/splits/"

Write-Host "Uploading v2.9 checkpoints (27 files)..."
scp "$($Paths.Ckpt)\*.pt" "${Host}:${Scr}/v2.9_20260502_181533/"

Write-Host "Uploading dann_transfer code..."
scp -r "$($Paths.Code)\*" "${Host}:${Acct}/Dann_transfer/dann_transfer/"

Write-Host "Upload complete. Submit with RUNBOOK Step 4 sbatch command (after MFA ssh login)."
