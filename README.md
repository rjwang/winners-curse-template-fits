# winners-curse-template-fits

Code, stored results and figure scripts for

> R. Wang, *The winner's curse in machine-learned template fits: selection bias from
> reusing finite simulated samples and how to avoid it*, submitted to Computer Physics
> Communications (2026).

When the same finite simulated sample is used to train classifiers, to choose among
candidate analysis designs (hyperparameters and binning) and to fill the templates of a
binned likelihood fit, the selected design is correlated with the fluctuations of its own
templates. This repository contains a controlled study of the effect: a toy
signal-plus-background model with known expected yields, a family of learned designs
(gradient-boosted classifiers with quantile binning), six workflows that use the
simulation in different ways (standard k-fold selection, hold-out, 50 % hold-out,
fold-back, cross-selection and strictly split cross-selection), a matched control for
every workflow, three calibration procedures, and a vectorised profile-likelihood fitter
that treats the finite simulation statistics exactly.

All numbers, tables and figures of the paper can be regenerated from the stored outputs in
`results/` in about a minute, and from scratch with the run scripts in `scripts/`.

---

## Installation

Python ≥ 3.10. The paper's runs used Python 3.14, NumPy 2.4.3 and scikit-learn 1.9.1.

```bash
git clone https://github.com/rjwang/winners-curse-template-fits.git
cd winners-curse-template-fits
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: NumPy, SciPy, pandas, scikit-learn, joblib, matplotlib (figures only),
pytest (tests only). No compiled code; runs on a laptop or workstation.

Set `OMP_NUM_THREADS=1` for the run scripts: parallelism is over replicas with joblib
processes (`--jobs`), and multithreaded BLAS/OpenMP inside each process only slows this
down.

## Quick start

```bash
# 1. tests of the fitter
python -m pytest tests -q

# 2. regenerate the analysis tables and figures from the stored outputs (~1 min)
bash scripts/analyze_and_plot.sh

# 3. end-to-end smoke test: a few replicas of the full pipeline
bash scripts/smoke_test.sh

# 4. performance benchmark of the fitter (--quick: about 10 s)
python benchmarks/bench_fitter.py --quick
```

## Reproducing the paper from scratch

| Step | Command | Wall clock (as recorded) |
|---|---|---|
| Confirmatory run (Tables 3–5, Figs. 3–4) | `bash scripts/run_confirmatory.sh` | 12 766 s on 8 cores for N_MC = 2000, K = 100 (100 replicas) |
| Exploratory runs (Fig. 2, likelihood comparison, bootstrap, training reuse) | `bash scripts/run_exploratory.sh` | 9323 s for the main run |
| Scaling study and Asimov predictions (Table 6, Fig. 5) | `bash scripts/run_theory.sh` | 3920 s + 2788 s |
| Analysis and figures | `bash scripts/analyze_and_plot.sh` | ~1 min |

The confirmatory run writes its output in chunks of 25 replicas and can be resumed if it
is interrupted. On one core a replica with N_MC = 2000 takes about 66 s for K = 1 and
220 s for K = 100.

## Repository layout

```
src/                      the code that produced the paper (unchanged; see docs/CODE_MAP.md)
  pilot_A_v3.py           toy model: generators, oracle yields, Fisher-information objective
  phase3_fit.py           vectorised profile-likelihood fitter (class Fitter)
  phase4_kfold.py         k-fold design family (HistGradientBoostingClassifier + quantile bins)
  phase4d_honest.py       exploratory workflows R0-R6h incl. bootstrap correction (R4)
  round2_run.py           confirmatory run: all workflows and matched controls, q0 of every toy
  analyze_round2.py       tables T1-T8 of the confirmatory run (shifts, calibration, power, ...)
  theory_selection.py     winner's-curse statistics sigma, tau and the Gaussian prediction
  theory_asimov.py        first-order and exact-Asimov predictions of the mu shift
  check_fitter.py         validation: Newton vs Nelder-Mead, calibration with perfect templates
  make_figures_v02.py     Figs. 1, 2, 5
  make_figures_v03.py     Figs. 3, 4
  (other files)           earlier stages of the study, kept for completeness
scripts/                  run, analysis and smoke-test scripts
benchmarks/bench_fitter.py  timing and accuracy of the fitter against scipy references
tests/                    pytest tests of the fitter
results/                  stored outputs of every run used in the paper (CSV, JSON, NPZ, logs)
docs/CODE_MAP.md          which script and which output file give each table and figure
```

## The fitter

`phase3_fit.Fitter` fits the binned likelihood

    L(mu, nu, beta) = N(z_nu | nu, 1) * prod_i Pois(n_i | mu s_i + beta_i (1 + nu delta_i)) * Pois(m_i | beta_i / w_b)

for many pseudo-experiments at once (`lik="bbp"`, the "background-Poisson" likelihood of
the paper). The per-bin background parameters beta_i are profiled in closed form (the
non-negative root of a quadratic), so the Newton iteration runs over (mu, nu) only, with
analytic gradients and a Hessian that includes the beta dependence through a Schur
complement. All pseudo-experiments are updated simultaneously as NumPy arrays. Free fits
respect mu ≥ 0: Newton from mu = 1, 0.3, 2 with projection onto the bound, plus the
boundary solution at mu = 0, keeping the lowest NLL. Each fit uses 60 Newton iterations
with steps clipped to ±1 and a backtracking line search (up to 12 halvings) that rejects
steps leaving the domain 1 + nu delta_i > 0. `lik="bb"` (Gaussian constraint on a per-bin
factor with pseudo-count 1/2) and `lik="plugin"` (no simulation statistics) are also
available.

```python
import sys; sys.path.insert(0, "src")
import numpy as np
import phase3_fit as F3

# templates of one design: signal yields s, simulated background counts m with weight w_b,
# shape response d (= delta * m * w_b); pseudo-data n (T x B) and auxiliary measurements nu0 (T,)
tau = m + 0.5                     # interface: tau = m + 1/2 and b = tau * w_b
F = F3.Fitter(s, tau * w_b, d, tau, n, nu0, "bbp")
mu_hat, nu_hat, nll_hat = F.fit_multistart()      # free fit, mu >= 0
_, _, nll_0 = F.fit(fix_mu=0.0)                   # background-only fit
q0 = np.maximum(2 * (nll_0 - nll_hat), 0.0) * (mu_hat > 0)
```

Several channels with common (mu, nu) are fitted by concatenating their bins
(`round2_run.fit_block`); each bin carries its own w_b.

## Reproducibility

Every random number derives from an explicit seed: replica r uses `1000 r + 17`
(confirmatory), `1000 r + 11` (exploratory) and `1000 r + 13` (scaling); pseudo-experiments
use `[seed, 99, j]` and are common to all workflows of a replica (paired comparisons); the
oracle reference samples use the seeds 12345 and 54321. Results agree within statistical
uncertainties across scikit-learn versions, but are bit-identical only for the same
versions of NumPy and scikit-learn. `results/round2_N*_K*_q0.npz` hold the q0 value of every
pseudo-experiment, so that calibrations can be recomputed without refitting.

## Citation

If you use this code, please cite the paper and the archived software version
(see `CITATION.cff`; Zenodo DOI: `10.5281/zenodo.<id>`).

## Licence

MIT, see `LICENSE`.
