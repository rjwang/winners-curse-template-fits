# Code map: which script produces which result of the paper

Numbers of tables and figures refer to the CPC submission (v0.4). All commands are run from
the repository root; `scripts/` contains them in runnable form.

## Library modules (imported by the run scripts)

| File | Content |
|---|---|
| `src/pilot_A_v3.py` | Toy model: `sample_sig`, `sample_bkg`, `nuis_raw` (shape bump), constants `S_TOT = 50`, `B_TOT = 1000`; profiled Fisher information `ieff` (selection objective, Eq. 4); `build_oracle` (reference samples, seed 12345) |
| `src/phase3_fit.py` | `Fitter`: vectorised profile-likelihood fits (`bbp` = background-Poisson likelihood of Eq. 2, `bb` = Gaussian constraint with pseudo-count, `plugin`); `fit_multistart` (mu >= 0), `fit(fix_mu=...)`; `run_toys` |
| `src/phase4_kfold.py` | k-fold design family: `draw_hparams`, `Design` (fold models + quantile edges on out-of-fold signal scores), `build_kfold`, `counts`, templates |
| `src/phase4d_honest.py` | All exploratory workflows (R0 naive, R1, R2, R3 independent templates, R4 bootstrap correction, R5 fold-back, R6/R6h cross-selection) |

Dependency chain: `pilot_A_v3` -> `phase3_fit` -> `phase4_kfold` -> `phase4d_honest` -> `round2_run`.
`phase4b_methods.py`, `phase4c_crosssel.py` and `phase4d_honest.py` are successive,
self-contained extensions of each other; the confirmatory run uses only `phase4d_honest.py`.

## Results of the paper

| Result in the paper | Script (run) | Output in `results/` | Table / figure script |
|---|---|---|---|
| Validity of the workflows (paired shift, FPR, ΔI) — Table "Validity", Fig. 3 | `round2_run.py` (confirmatory) | `round2_N*_K*_raw.csv`, `round2_N*_K*_q0.npz` | `analyze_round2.py` → `round2_analysis_T1_matched.csv`, `_T2_absolute.csv`; `make_figures_v03.py` → `fig3_validity` |
| Estimation of mu (bias, coverage, RMSE) — Table "Estimation" | same | same | `analyze_round2.py` → `_T2_absolute.csv` |
| Calibration and power — Table "Calibration", Fig. 4 | same | same | `analyze_round2.py` → `_T6_calibration.csv`, `_T7_power_paired.csv`; `make_figures_v03.py` → `fig4_power` |
| Fraction of analyses with FPR > 10 % (Sec. 6) | same | same | `analyze_round2.py` → `_T8_conditional.csv` |
| Two-channel structure (Sec. 5) | same | same | `analyze_round2.py` → `_T3_structure.csv` |
| Channel correlations (Sec. 5) | same | same | `analyze_round2.py` → `_T4_channels.csv` |
| Exact signal template / shape response (Sec. 5) | same | same | `analyze_round2.py` → `_T5_ablation.csv` |
| Second oracle sample (Sec. 5) | same (`--no_oracle2` not set) | same | `analyze_round2.py` → `_T2_absolute.csv` (`true_inf2`) |
| FPR versus number of candidates — Fig. 2; exploratory R1 shift 0.54σ; bootstrap R4 (Sec. 5) | `phase4d_honest.py` (exploratory, `--liks bbp`, 50 replicas) | `phase4d_raw.csv`, `phase4d_summary.csv` | `make_figures_v02.py` → `fig2_fpr_vs_K` |
| Gaussian-constraint vs background-Poisson likelihood (Sec. 4) | `phase4c_crosssel.py` (`--liks bb bbp`, 30 replicas) | `phase4c_raw.csv`, `phase4c_summary.csv` | — |
| Training reuse R0; tail rates with `n_noise = 0` (Secs. 4, 7) | `phase4d_honest.py` (`--n_noise 0 10`, 30 replicas) | `phase4e_naive_bbp_*.csv` | — |
| Plug-in Fisher information of a fixed design within 3 % (Sec. 4) | `pilot_A_v2.py` (`P_fixed`) | `phase1b_conservative_*.csv` | — |
| Winner's-curse statistics σ, τ, Gaussian prediction — Table "Winner's curse", Fig. 5(a) | `theory_selection.py` (scaling, 30 replicas) | `theory_raw.csv`, `theory_summary.csv` | `make_figures_v02.py` → `fig5_theory` |
| First-order and exact-Asimov predictions of the shift — Fig. 5(b) | `theory_asimov.py` | `theory_asimov_raw.csv`, `theory_asimov_summary.csv` | `make_figures_v02.py` → `fig5_theory` |
| Data-flow schematic — Fig. 1 | — | — | `make_figures_v02.py` → `fig1_schematic` |
| Fitter validation (Newton vs Nelder–Mead; perfect templates) | `check_fitter.py` | printed | also `tests/test_fitter.py` |
| Fitter performance (Sec. "Software implementation and performance") | `benchmarks/bench_fitter.py` | `benchmarks/results/bench_fitter*.{csv,json}` | — |

Every run writes `<out>_config.json` (all arguments) and, where applicable, `<out>_log.txt`.

## Earlier stages kept for completeness (not used for the paper's results)

| File | Stage |
|---|---|
| `src/pilot_A.py`, `src/pilot_A_v2.py` | Stage 1: decision-tree designs and the optimism of the plug-in Fisher information (`phase1_*`, `phase1b_*`, `cmp_v2_*`, `cmp_v3_*` = consistency checks between v2 and v3) |
| `src/pilot_A_v3.py` (`main`) | Stage 2: MC-aware objective (`phase2a_mcaware_*`) |
| `src/phase3_fit.py` (`main`) | Stage 3: decision-tree designs with full fits (`phase3_main_*`) |
| `src/phase4_kfold.py` (`main`), `src/phase4b_methods.py` | Stage 4a/4b: first k-fold runs (`phase4a_*`, `phase4b_*`) |
| `src/make_figures.py` | Figures of the first draft (superseded by `make_figures_v02.py`, `make_figures_v03.py`) |

## Naming used in the code versus the paper

| Code | Paper |
|---|---|
| `bbp` | background-Poisson likelihood (Eq. 2) |
| `bb` | Gaussian constraint on a per-bin factor with pseudo-count 1/2 |
| `R6h` (phase 4d), `CS` (round 2) | cross-selection |
| `CSs` | strictly split cross-selection |
| `R3` | independent-template reference of the exploratory run |
| `<W>_ind` | matched control of workflow W |
| `R1_ind2ch` | R1 design split into two channels with independent templates (two-channel structure check) |
| `_xs`, `_xsd` | exact signal template; exact signal template and shape response |

Note: the docstring of `phase3_fit.py` calls `bbp` "full Barlow-Beeston"; as stated in the
paper, the signal template and shape response are held fixed, so it is a Barlow-Beeston
treatment of the background statistics only.
