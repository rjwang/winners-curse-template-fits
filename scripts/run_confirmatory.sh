#!/usr/bin/env bash
# Confirmatory run of the paper (Tables 3-5, Figs. 3-4; Sec. 5 checks).
# 17 workflows/controls per replica, 300 pseudo-experiments x 4 hypotheses each.
# Output is written in chunks of 25 replicas; re-running the same command resumes.
# Recorded: N_MC=2000, K=100 (100 replicas) took 12766 s on 8 cores.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
JOBS=${JOBS:--1}          # number of parallel processes (-1 = all cores)
python src/round2_run.py --configs 2000:1:100,2000:100:100   --jobs "$JOBS" --out results/round2
python src/round2_run.py --configs 10000:1:50,10000:100:100  --jobs "$JOBS" --out results/round2
python src/analyze_round2.py --prefix results/round2 --out results/round2_analysis | tee results/round2_analysis.txt
