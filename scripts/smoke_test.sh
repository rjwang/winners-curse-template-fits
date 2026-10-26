#!/usr/bin/env bash
# End-to-end smoke test of the confirmatory pipeline: 4 replicas each for K = 1 and K = 3,
# 60 pseudo-experiments, small oracle sample. Output goes to a temporary prefix.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
OUT=${OUT:-results_smoke/r2smoke}
mkdir -p "$(dirname "$OUT")"
python src/round2_run.py --configs 2000:1:4,2000:3:4 --n_toys 60 --n_orc_s 30000 --n_orc_b 90000 --out "$OUT"
python src/analyze_round2.py --prefix "$OUT" --out "${OUT}_analysis"
echo "smoke test finished: ${OUT}_analysis_T*.csv"
