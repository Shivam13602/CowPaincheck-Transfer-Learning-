# max_frames and batch-size sweeps (Holstein / V2)

Longer temporal context uses more VRAM. **`dann_adapt_v2.9.py`** and **`weak_label_adapt_v2.9.py`** both accept **`--max-frames`** (wired from checkpoint `cfg` defaults unless overridden).

## Environment variables (`run_task1_vast.sh`)

| Variable | Effect |
|----------|--------|
| `MAX_FRAMES` | Passed as `--max-frames` to DANN and weak stages (omit for default, usually **32** from checkpoint). |
| `BATCH_SIZE` | Global batch size for both stages (default **8**). Reduce when raising `MAX_FRAMES`. |
| `TRAIN_LR` | DANN learning rate (default `1e-5`). |
| `WEAK_LR` | Weak-label stage LR (default `1e-4`). |
| `DIAG_BOOTSTRAP_SAMPLES` | If set, passed as `--diag-bootstrap-samples` (`0` disables cow-level bootstrap CIs; omit for default **2000**). |

## Cow-level bootstrap CIs

By default both stages use **`--diag-bootstrap-samples 2000`**: resample held-out **cows** with replacement and record **95% intervals** for cow-level AUC and balanced accuracy (prob ≥ 0.5). Results appear in `*_diagnostics.json` and the report. Set **`DIAG_BOOTSTRAP_SAMPLES=0`** in the environment to skip.

## Suggested grid (start conservative)

1. Baseline: omit `MAX_FRAMES`, `BATCH_SIZE=8`.
2. `MAX_FRAMES=64`, `BATCH_SIZE=4`.
3. `MAX_FRAMES=96` or `128`, `BATCH_SIZE=2` (watch OOM on 24GB).

Always keep **train** and **eval** deterministic for val/test: the dataset uses **linspace** temporal sampling when augmentations are off.

## Sliding-window inference (optional)

Does **not** change training tensor size; only **evaluation** paths (`val` / test ensemble) can scan more video context:

| Variable | Maps to CLI |
|----------|-------------|
| `INFER_SLIDING_RAW_SPAN` | `--infer-sliding-raw-span` (consecutive on-disk frames per window) |
| `INFER_SLIDING_STRIDE` | `--infer-sliding-stride` (optional; defaults to `span//4` in Python if omitted) |
| `INFER_SLIDING_AGGREGATE` | `--infer-sliding-aggregate` (`mean`, `trimmed_mean`, `max`) |

## MC dropout at eval

Set `EVAL_MC_SAMPLES` to a small integer (e.g. `5`) to enable **`--eval-mc-samples`**. Uses **`model.train()`** during those forward passes; batch norm statistics can shift — treat as a diagnostic, not ground-truth uncertainty.
