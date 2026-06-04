#!/usr/bin/env bash
# Waits for the Stage A autoresearch launcher to exit, then runs Stage B (DANN/CORAL).
set -u

WORKDIR="/root/CowPaincheck-Transfer-Learning/dann_transfer/V6/auto research"
SPACE="search_space.vast.json"
RUNS="/root/runs/v6_auto"
STAGE_A_PAT="run_autoresearch.py --search-space ${SPACE} --stage A --execute"

mkdir -p "${RUNS}"
echo "[queue] waiting for Stage A to finish at $(date -u)" >> "${RUNS}/queue.log"

while pgrep -f "${STAGE_A_PAT}" >/dev/null 2>&1; do
  sleep 60
done

echo "[queue] Stage A finished, launching Stage B at $(date -u)" >> "${RUNS}/queue.log"
cd "${WORKDIR}" || { echo "[queue] cd failed" >> "${RUNS}/queue.log"; exit 1; }
python3 run_autoresearch.py --search-space "${SPACE}" --stage B --execute >> "${RUNS}/stageB.log" 2>&1
echo "[queue] Stage B finished at $(date -u)" >> "${RUNS}/queue.log"
