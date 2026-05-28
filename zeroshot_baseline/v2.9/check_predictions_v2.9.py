# ============================================================================
# Check if v2.9 test predictions exist for a run_tag (e.g. after Colab training).
# Run in Colab or locally; set project_dir to where results_v2.9 lives.
# ============================================================================

from pathlib import Path
import sys

RUN_TAG = "v2.9_20260221_223056"
TEST_ANIMALS = "14-17"

# Default: Colab Drive mount. Override with first CLI arg if needed.
PROJECT_DIR = Path("/content/drive/MyDrive/facial_pain_project_v2")
if len(sys.argv) > 1:
    PROJECT_DIR = Path(sys.argv[1])

RESULTS_DIR = PROJECT_DIR / "results_v2.9" / RUN_TAG
CKPT_DIR = PROJECT_DIR / "checkpoints_v2.9" / RUN_TAG

print("Run tag:", RUN_TAG)
print("Project dir:", PROJECT_DIR)
print("Results dir:", RESULTS_DIR)
print("Checkpoint dir:", CKPT_DIR)
print()

if not RESULTS_DIR.exists():
    print("RESULTS dir does not exist. Predictions have not been generated yet.")
    print("Run: evaluate_test_set_v2.9_cli.py --run_tag", RUN_TAG, "--ckpt_kind task2")
    sys.exit(1)

predictions = {}
for ckpt_kind in ["combined", "task1", "task2"]:
    fname = f"test_eval_v2.9_{ckpt_kind}_animals_{TEST_ANIMALS}_predictions.csv"
    path = RESULTS_DIR / fname
    predictions[ckpt_kind] = path.exists()
    status = "OK" if path.exists() else "MISSING"
    print(f"  {ckpt_kind:10} -> {status:8}  {path.name}")

if any(predictions.values()):
    print()
    print("At least one predictions CSV found. You can run:")
    print("  bootstrap_ci_v2.9.py    --run_tag", RUN_TAG, "--ckpt_kind task2")
    print("  calibration_v2.9.py     --run_tag", RUN_TAG, "--ckpt_kind task2")
else:
    print()
    print("No predictions CSV found. Run the test evaluator first:")
    print("  evaluate_test_set_v2.9_cli.py --run_tag", RUN_TAG, "--ckpt_kind task2")

# Checkpoints (optional)
if CKPT_DIR.exists():
    pt = list(CKPT_DIR.glob("best_model_v2.9*.pt"))
    print()
    print("Checkpoints:", len(pt), "files")
else:
    print()
    print("Checkpoint dir not found; ensure run_tag and project path are correct.")
