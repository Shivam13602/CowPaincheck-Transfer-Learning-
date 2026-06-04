# V6 Auto Research

**Results:** see [`../v6.md`](../v6.md) and `../results/vast_auto/`. Workspace

This workspace implements an `autoresearch`-style loop for V6:
- define experiment candidates in one search-space file,
- run reproducible trials with fixed metadata,
- summarize and promote only robust candidates.

## Files
- `program.md`: operating policy for autonomous/semiautonomous sweeps.
- `search_space.json`: declarative trial list and default runtime context.
- `run_autoresearch.py`: executes or prints trial commands.
- `summarize_trials.py`: builds ranked results table from trial outputs.

## Quick usage

From repo root:

```bash
python "dann_transfer/V6/auto research/run_autoresearch.py" --dry-run
```

Run only stage A candidates:

```bash
python "dann_transfer/V6/auto research/run_autoresearch.py" --stage A --dry-run
```

After trial folders exist:

```bash
python "dann_transfer/V6/auto research/summarize_trials.py" --results-root /scratch/shiv136/project_data/runs/v6_auto
```

## Recommended workflow
1. Dry-run commands and verify split/checkpoint paths.
2. Launch small smoke subset (2-3 runs).
3. Launch full stage.
4. Summarize and promote top candidates by:
   - sequence AUC,
   - sequence F1 at primary threshold,
   - no catastrophic threshold collapse,
   - condition-wise sanity (lameness not zero).

## Caution
- Keep held-out test cows fixed.
- Never tune on test-only metrics.
- Save every command and summary artifact.
