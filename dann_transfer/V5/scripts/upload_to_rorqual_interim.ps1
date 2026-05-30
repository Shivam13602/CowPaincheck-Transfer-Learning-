# Upload interim V5 assets to Rorqual (549-seq already on scratch).
# Checkpoints: remote_upload_ucaps/.../v2.9_20260502_181533 (27 .pt files).
# Uses ONE ssh for remote setup + ckpt check (avoids a second Duo prompt).
#
# Tip — reuse one MFA for several commands (PowerShell, once per session):
#   notepad $env:USERPROFILE\.ssh\config
#   Host rorqual.alliancecan.ca
#     ControlMaster auto
#     ControlPath ~/.ssh/cm-%r@%h:%p
#     ControlPersist 15m
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File upload_to_rorqual_interim.ps1
#   powershell -ExecutionPolicy Bypass -File upload_to_rorqual_interim.ps1 -ForceCkptUpload
#   powershell -ExecutionPolicy Bypass -File upload_to_rorqual_interim.ps1 -SkipCkptUpload

param(
    [switch]$ForceCkptUpload,
    [switch]$SkipCkptUpload,
    [string]$CkptLocal = ""
)

$ErrorActionPreference = "Stop"
$Root = "C:\Users\shivp\Downloads\Research\DATASET"
$Acct = "shiv136"
$Remote = "${Acct}@rorqual.alliancecan.ca"
$Scr = "/scratch/${Acct}/project_data"
$CkptRemote = "${Scr}/v2.9_20260502_181533"

function Resolve-CkptLocal {
    param([string]$Override)
    if ($Override -and (Test-Path $Override)) { return (Resolve-Path $Override).Path }
    $named = Get-ChildItem -Path (Join-Path $Root "remote_upload_ucaps") -Recurse -Directory -Filter "v2.9_20260502_181533" -ErrorAction SilentlyContinue |
        Where-Object { (Get-ChildItem -Path $_.FullName -Filter "*.pt" -ErrorAction SilentlyContinue).Count -ge 9 } |
        Select-Object -First 1
    if ($named) { return $named.FullName }
    $fallback = Join-Path $Root "remote_upload_ucaps\Transferlearning\v2.9\v2.9_20260502_181533-20260528T182044Z-3-001\v2.9_20260502_181533"
    if (Test-Path $fallback) { return $fallback }
    throw "Cannot find v2.9_20260502_181533 under $Root\remote_upload_ucaps (need 9+ .pt files)."
}

$Paths = @{
    Splits = Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer\V5\splits"
    Ckpt   = Resolve-CkptLocal -Override $CkptLocal
    Code   = Join-Path $Root "CowPaincheck-Transfer-Learning\dann_transfer"
}

$localPt = @(Get-ChildItem -Path $Paths.Ckpt -Filter "*.pt").Count
if ($localPt -lt 9) { throw "Only $localPt .pt files in $($Paths.Ckpt); expected 27." }

foreach ($key in @("Splits", "Code")) {
    if (-not (Test-Path $Paths[$key])) { throw "Missing ${key}: $($Paths[$key])" }
}

Write-Host "== Interim V5 upload (549 on scratch; local v5 extraction untouched) =="
Write-Host "Checkpoints (local): $($Paths.Ckpt)  ($localPt .pt)"
Write-Host "Checkpoints (remote): $CkptRemote"
Write-Host "549 sequences (already on scratch): ${Scr}/cow_face_sequences_thesis_stride8/output"
Write-Host ""

# Single SSH: mkdir + count remote task1 folds (one Duo approval).
$remoteSetup = @"
mkdir -p ${Scr}/v5/splits ${CkptRemote} ${Scr}/runs/logs ${Acct}/Dann_transfer/dann_transfer
ls -1 ${CkptRemote}/best_model_v2.9_task1_fold_*.pt 2>/dev/null | wc -l
"@.Trim()

Write-Host "SSH: create dirs + check remote checkpoints (approve Duo once)..."
$ckptCountRaw = & ssh $Remote $remoteSetup 2>&1 | Select-Object -Last 1
$ckptCount = "$ckptCountRaw".Trim()
Write-Host "Remote task1 fold checkpoints: $ckptCount"

$uploadCkpt = $ForceCkptUpload -or (-not $SkipCkptUpload -and -not ($ckptCount -match '^\d+$' -and [int]$ckptCount -ge 9))
if ($uploadCkpt) {
    Write-Host "Uploading $localPt checkpoint files to $CkptRemote (approve Duo if prompted)..."
    & scp "$($Paths.Ckpt)\*.pt" "${Remote}:${CkptRemote}/"
} else {
    Write-Host "Skipping checkpoint upload (already >= 9 task1 folds on scratch). Use -ForceCkptUpload to replace."
}

Write-Host "Uploading V5 splits..."
& scp -r "$($Paths.Splits)\*" "${Remote}:${Scr}/v5/splits/"

Write-Host "Uploading dann_transfer code..."
& scp -r "$($Paths.Code)\*" "${Remote}:${Acct}/Dann_transfer/dann_transfer/"

Write-Host ""
Write-Host "Done. Submit training (SSH + Duo again unless ControlPersist is enabled):"
Write-Host "  ssh $Remote"
Write-Host "  cd ~/Dann_transfer"
Write-Host "  sbatch --export=ALL,CKPT_DIR=${CkptRemote},NUM_WORKERS=24,WEAK_BATCH_SIZE=128,DANN_BATCH_SIZE=48 dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_549.sh"
