# Download V6 autoresearch results from VastAI (single tarball).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File download_results_vast.ps1

param(
    [string]$Host = "157.90.56.162",
    [int]$Port = 31799,
    [string]$RemoteTar = "/root/v6_results_reports.tgz"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$V6 = Join-Path $Root "dann_transfer\V6"
$Local = Join-Path $V6 "results\vast_auto"
$Tar = Join-Path $V6 "results\v6_results_reports.tgz"

New-Item -ItemType Directory -Force -Path $Local | Out-Null

Write-Host "Step 1/2: create tarball on Vast (excludes fold checkpoints) ..."
ssh -p $Port "root@${Host}" @"
set -e
cd /root/runs/v6_auto
tar czf '$RemoteTar' A_s3_focal_g1p5_cb A_s3_focal_g2p5_cb A_s3_gce_q0p6_cb A_s3_gce_q0p8_cb \
  B_s4_dann_dw0p15 B_s4_dann_dw0p20 B_s4_coral_w0p02 stageA.log stageB.log queue.log \
  --exclude='*.pt' 2>/dev/null || true
ls -lh '$RemoteTar'
"@

Write-Host ""
Write-Host "Step 2/2: download and extract ..."
scp -P $Port "root@${Host}:${RemoteTar}" $Tar
tar -xzf $Tar -C $Local
Write-Host "Done. Local: $Local"
