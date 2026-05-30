# Upload + submit (549 interim, ckpt `v2.9_20260502_181533`)

Checkpoint folder on your PC (27 files):

`C:\Users\shivp\Downloads\Research\DATASET\remote_upload_ucaps\Transferlearning\v2.9\v2.9_20260502_181533-20260528T182044Z-3-001\v2.9_20260502_181533`

Remote (Rorqual):

`/scratch/shiv136/project_data/v2.9_20260502_181533`

Training uses `CKPT_DIR` above automatically.

---

## Optional: one Duo approval for several SSH/SCP commands

Add to `%USERPROFILE%\.ssh\config`:

```
Host rorqual.alliancecan.ca
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 15m
```

---

## A) Script (fixed: single SSH for mkdir + ckpt check)

```powershell
cd "C:\Users\shivp\Downloads\Research\DATASET"
powershell -ExecutionPolicy Bypass -File CowPaincheck-Transfer-Learning\dann_transfer\V5\scripts\upload_to_rorqual_interim.ps1
```

Force re-upload checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File CowPaincheck-Transfer-Learning\dann_transfer\V5\scripts\upload_to_rorqual_interim.ps1 -ForceCkptUpload
```

---

## B) Manual (if script still hits Duo too often)

Run each block; approve Duo when asked (`1` = push).

```powershell
$R = "shiv136@rorqual.alliancecan.ca"
$Scr = "/scratch/shiv136/project_data"
$Ckpt = "C:\Users\shivp\Downloads\Research\DATASET\remote_upload_ucaps\Transferlearning\v2.9\v2.9_20260502_181533-20260528T182044Z-3-001\v2.9_20260502_181533"
```

**1) Remote dirs (one SSH)**

```powershell
ssh $R "mkdir -p $Scr/v5/splits $Scr/v2.9_20260502_181533 $Scr/runs/logs ~/Dann_transfer/dann_transfer"
```

**2) Checkpoints (27 .pt)**

```powershell
scp "$Ckpt\*.pt" "${R}:${Scr}/v2.9_20260502_181533/"
```

**3) Split + code**

```powershell
scp -r "C:\Users\shivp\Downloads\Research\DATASET\CowPaincheck-Transfer-Learning\dann_transfer\V5\splits\*" "${R}:${Scr}/v5/splits/"
scp -r "C:\Users\shivp\Downloads\Research\DATASET\CowPaincheck-Transfer-Learning\dann_transfer\*" "${R}:~/Dann_transfer/dann_transfer/"
```

---

## Submit training (H100)

```bash
ssh shiv136@rorqual.alliancecan.ca
cd ~/Dann_transfer

sbatch --export=ALL,\
CKPT_DIR=/scratch/shiv136/project_data/v2.9_20260502_181533,\
NUM_WORKERS=24,\
WEAK_BATCH_SIZE=128,\
DANN_BATCH_SIZE=48,\
DATALOADER_PREFETCH=4,\
DATALOADER_PERSISTENT=1 \
  dann_transfer/V5/scripts/sbatch_v5_matrix_rorqual_549.sh
```

Verify GPU after job starts:

```bash
squeue -u shiv136
tail -f /scratch/shiv136/project_data/runs/logs/v5_549_8cow-*.out
```
