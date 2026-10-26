#!/usr/bin/env bash
# Scaling study (Table 6, Fig. 5a; 3920 s as recorded) and Asimov predictions (Fig. 5b; 2788 s).
# Both read the measured shifts of the exploratory run from results/phase4d_raw.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
JOBS=${JOBS:--1}
python src/theory_selection.py --n_mc 2000 5000 10000 --K 1 5 20 100 --n_noise 10 --alpha 0.3 --reps 30 \
    --paired results/phase4d_raw.csv --jobs "$JOBS" --out results/theory
python src/theory_asimov.py --n_mc 2000 10000 --K 1 20 100 --n_noise 10 --alpha 0.3 --reps 30 \
    --paired results/phase4d_raw.csv --jobs "$JOBS" --out results/theory_asimov
