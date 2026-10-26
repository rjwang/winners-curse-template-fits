#!/usr/bin/env bash
# Exploratory runs (seeds 1000r+11): Fig. 2, bootstrap correction R4, comparison of
# likelihoods (Sec. 4), training reuse (Sec. 4), tail rates without irrelevant features (Sec. 7).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
JOBS=${JOBS:--1}
# main exploratory run (9323 s as recorded)
python src/phase4d_honest.py --n_mc 2000 10000 --K 1 20 100 --n_noise 10 --alpha 0.3 --reps 50 \
    --n_toys 300 --n_boot 200 --skip_naive --liks bbp --jobs "$JOBS" --out results/phase4d
# Gaussian-constraint vs background-Poisson likelihood on the same pseudo-data
python src/phase4c_crosssel.py --n_mc 2000 10000 --K 1 20 100 --n_noise 10 --alpha 0.3 --reps 30 \
    --n_toys 300 --n_boot 200 --skip_naive --liks bb bbp --jobs "$JOBS" --out results/phase4c
# training reuse (naive R0) and n_noise = 0
python src/phase4d_honest.py --n_mc 2000 10000 --K 1 20 --n_noise 0 10 --alpha 0.3 --reps 30 \
    --n_toys 300 --n_boot 200 --liks bbp --jobs "$JOBS" --out results/phase4e_naive_bbp
# plug-in Fisher information of a fixed design (preliminary tree-based study, P_fixed)
python src/pilot_A_v2.py --n_mc 2000 10000 --K 1 10 50 --n_noise 0 10 --alpha 0.0 0.3 0.6 \
    --min_leaf 50 100 --b_floor 5 --reps 30 --out results/phase1b_conservative
