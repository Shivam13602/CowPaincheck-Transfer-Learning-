# Download V5 549-seq reports (S3 + S4) from Rorqual with minimal Duo prompts.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File download_results_549.ps1
#       -> one SSH (tar on cluster) + one scp (default; ~2 Duo approvals)
#
#   powershell -ExecutionPolicy Bypass -File download_results_549.ps1 -ResumeS4Only
#       -> only missing S4 folders, single recursive scp per folder (~1 Duo each)
#
param(
    [switch]$ResumeS4Only,
    [switch]$SkipS3
)

$ErrorActionPreference = "Stop"
$Remote = "shiv136@rorqual.alliancecan.ca"
$RemoteRun = "/scratch/shiv136/project_data/runs/v5_thesis_8cow_549"
$RemoteTar = "/scratch/shiv136/project_data/runs/v5_549_reports_nopt.tgz"
$Local = "C:\Users\shivp\Downloads\Research\DATASET\CowPaincheck-Transfer-Learning\dann_transfer\V5\results\549_interim"
$LocalTar = Join-Path (Split-Path $Local -Parent) "v5_549_reports_nopt.tgz"

$S3 = @("S3_weak_bce", "S3_weak_focal", "S3_weak_gce")
$S4 = @(
    "S4_dann_dw_0.10", "S4_dann_dw_0.25", "S4_dann_dw_0.50",
    "S4_coral_w_0.01", "S4_coral_w_0.05", "S4_coral_w_0.10"
)

New-Item -ItemType Directory -Force -Path $Local | Out-Null

function Test-S4Complete($name) {
    $d = Join-Path $Local $name
    return (Test-Path (Join-Path $d "dann_summary.json")) -and (Test-Path (Join-Path $d "dann_report.md"))
}

function Download-S4Folder($cond) {
    $dst = Join-Path $Local $cond
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    $src = "${Remote}:${RemoteRun}/${cond}/"
    Write-Host "scp -r $cond (one Duo session per folder) ..."
    scp -r "${src}dann_report.md" "${src}dann_summary.json" "${src}dann_diagnostics.json" `
        "${src}dann_splits.json" "${src}dann_fold_summary.csv" `
        "${src}dann_test_predictions.csv" "${src}dann_test_cow_aggregates.csv" `
        "${src}dann_test_video_aggregates.csv" $dst
}

if ($ResumeS4Only) {
    Write-Host "Resume: S4 only (skip complete folders)."
    foreach ($cond in $S4) {
        if (Test-S4Complete $cond) {
            Write-Host "  skip $cond (already local)"
            continue
        }
        Download-S4Folder $cond
    }
    Write-Host "Done. Local: $Local"
    exit 0
}

# --- Bundle mode (recommended): 2 connections total ---
$dirs = @()
if (-not $SkipS3) { $dirs += $S3 }
$dirs += $S4
$dirArgs = ($dirs | ForEach-Object { $_ }) -join " "

Write-Host "Step 1/2: create tarball on Rorqual (approve Duo once) ..."
ssh $Remote @"
set -e
cd '$RemoteRun'
tar czf '$RemoteTar' $dirArgs \
  --exclude='fold_*' --exclude='*.pt' --exclude='best_*.pt' 2>/dev/null || \
tar czf '$RemoteTar' $dirArgs --exclude='fold_*' --exclude='*.pt'
ls -lh '$RemoteTar'
"@

Write-Host ""
Write-Host "Step 2/2: download tarball (approve Duo once) ..."
scp "${Remote}:${RemoteTar}" $LocalTar

Write-Host "Extracting ..."
if (Test-Path $LocalTar) {
    tar -xzf $LocalTar -C $Local
    Remove-Item -Force $LocalTar -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Saved under: $Local"
Write-Host "Analyze: python CowPaincheck-Transfer-Learning/dann_transfer/V5/scripts/build_v5_results_analysis.py"
