#!/usr/bin/env python3
"""
Topic A - Phase 5a: mechanism / scaling of selection reuse.

Per replica: train K_max k-fold candidates (as R1 in phase 4), then for nested subsets of the
first K candidates (K in --K) measure
  I_hat_c   : MC-aware Fisher from the OOF templates (the selection statistic)
  I_c       : infinite-MC Fisher on the oracle (true design quality)
  eps_c     = I_hat_c - I_c
  wc_I      = eps_sel - mean_c eps_c                       (winner's curse in the statistic)
  tau       = sd_c I_c ,  sigma = sd_c (eps_c - mean eps)  (spread of truth / of noise)
  wc_gauss  = sigma^2 / sqrt(tau^2 + sigma^2) * e_K        (Gaussian winner's-curse prediction)
  dmu_c     = linearised mu-hat bias from template errors of candidate c (no fit parameters),
              nu profiled: delta theta = (J^T W J + C)^-1 J^T W (lam_true - lam_hat), theta=(mu,nu)
  wc_mu     = dmu_sel - mean_c dmu_c                       (prediction of the R1 - R3 paired shift)
Phase 5b additions (nonlinearity of the fit):
  wc_mu_tw     : same linearisation but with weights 1/lambda_hat from the TEMPLATES (the fit
                 weights bins by its own model, and selection picks designs whose templates are
                 low exactly where the signal is)
  wc_mu_asimov : exact full-Barlow-Beeston fit to Asimov data (data = true expected yields),
                 mu_hat - 1 per candidate; no toys, no linearisation
If --paired points to phase4d_raw.csv, the measured R1 - R3 paired shift of mu-hat (mu_true = 1)
is printed next to the prediction.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_A_v3 as P
import phase4_kfold as K4
import phase3_fit as F3


def expected_max_normal(K, n=200_000, seed=0):
    if K == 1:
        return 0.0
    z = np.random.default_rng(seed).standard_normal((n // K + 1000, K))
    return float(z.max(1).mean())


def dmu_linear(s_hat, b_hat, s, b, d, mu=1.0):
    """Linearised mu-hat bias (nu profiled, unit Gaussian constraint) when the fit uses templates
    (s_hat, b_hat) but data follow (s, b): delta theta = (J^T W J + C)^-1 J^T W (lam_true - lam_hat)."""
    lam = np.maximum(b + mu * s, 1e-12)
    J = np.stack([s, d], 1)
    Fm = J.T @ (J / lam[:, None]) + np.diag([0.0, 1.0])
    resid = (b - b_hat) + mu * (s - s_hat)
    return float(np.linalg.solve(Fm, J.T @ (resid / lam))[0])


def dmu_linear_tw(s_hat, b_hat, s, b, d, w_floor, mu=1.0):
    """As dmu_linear but weighting bins by the template prediction lambda_hat."""
    lam_hat = np.maximum(b_hat + mu * s_hat, w_floor)
    J = np.stack([s, d], 1)
    Fm = J.T @ (J / lam_hat[:, None]) + np.diag([0.0, 1.0])
    resid = (b - b_hat) + mu * (s - s_hat)
    return float(np.linalg.solve(Fm, J.T @ (resid / lam_hat))[0])


def dmu_asimov_bbp(ns, nb, g, smp, s, b):
    """Exact full-BB fit of Asimov data (n = true expected counts at mu=1, nu=0)."""
    tm_s, tm_b, tm_d = ns * smp.ws, (nb + 0.5) * smp.wb, g * smp.wb
    Fi = F3.Fitter(tm_s, tm_b, tm_d, nb + 0.5, (b + s)[None, :], np.zeros(1), "bbp")
    mu, _, _ = Fi.fit_multistart()
    return float(mu[0] - 1.0)


def one_rep(seed, n_mc, Ks, n_noise, alpha, oracle, Fn):
    rng = np.random.default_rng(seed)
    n = n_mc // 2
    Xs, Xb = P.sample_sig(rng, n, n_noise), P.sample_bkg(rng, n, n_noise)
    full = K4.Smp(Xs, Xb, P.nuis_raw(Xb) - oracle["gmean"], alpha)
    osm = oracle["smp"][alpha]
    orc = K4.Smp(osm.Xs, osm.Xb, P.nuis_raw(osm.Xb) - oracle["gmean"], alpha)
    hps = K4.draw_hparams(rng, max(Ks))
    Ihat, Itrue, dmu, dmu_tw, dmu_as = [], [], [], [], []
    for hp in hps:
        d = K4.build_kfold(hp, full, Fn)
        ns, nb, g = K4.counts(d, full)
        Ihat.append(float(P.est_from_counts(ns, nb, g, full)))
        nso, nbo, go = K4.counts(d, orc)
        s, b, dd = nso * orc.ws, np.maximum(nbo * orc.wb, 1e-3 * orc.wb), go * orc.wb
        Itrue.append(float(P.ieff(s, b, dd)))
        dmu.append(dmu_linear(ns * full.ws, nb * full.wb, s, b, dd))
        dmu_tw.append(dmu_linear_tw(ns * full.ws, nb * full.wb, s, b, dd, 0.5 * full.wb))
        dmu_as.append(dmu_asimov_bbp(ns, nb, g, full, s, b))
    Ihat, Itrue, dmu, dmu_tw, dmu_as = map(np.array, (Ihat, Itrue, dmu, dmu_tw, dmu_as))
    eps = Ihat - Itrue
    rows = []
    for K in Ks:
        sl = slice(0, K); k = int(np.argmax(Ihat[sl]))
        rows.append(dict(seed=seed, n_mc=n_mc, K=K, n_noise=n_noise, alpha=alpha,
                         wc_I=float(eps[k] - eps[sl].mean()),
                         tau=float(Itrue[sl].std()) if K > 1 else 0.0,
                         sigma=float((eps[sl] - eps[sl].mean()).std()) if K > 1 else 0.0,
                         regret=float(Itrue[sl].max() - Itrue[k]),
                         wc_mu=float(dmu[k] - dmu[sl].mean()),
                         wc_mu_tw=float(dmu_tw[k] - dmu_tw[sl].mean()),
                         wc_mu_asimov=float(dmu_as[k] - dmu_as[sl].mean()),
                         dmu_sel=float(dmu[k]), I_sel=float(Itrue[k])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_mc", type=int, nargs="+", default=[2000, 10000])
    p.add_argument("--K", type=int, nargs="+", default=[1, 20, 100])
    p.add_argument("--n_noise", type=int, nargs="+", default=[10])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.3])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--n_orc_s", type=int, default=100_000)
    p.add_argument("--n_orc_b", type=int, default=300_000)
    p.add_argument("--paired", default="", help="phase4d_raw.csv for measured R1-R3 shifts")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default="results/theory_asimov")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(vars(a), open(a.out + "_config.json", "w"), indent=2)

    t0 = time.time(); allrows = []
    for nn in a.n_noise:
        oracle = P.build_oracle(nn, a.alpha, a.n_orc_s, a.n_orc_b)
        for al in a.alpha:
            for N in a.n_mc:
                tasks = [delayed(one_rep)(1000 * r + 13, N, a.K, nn, al, oracle, a.folds)
                         for r in range(a.reps)]
                for rows in Parallel(n_jobs=a.jobs)(tasks):
                    allrows += rows
                print(f"done n_noise={nn} alpha={al} N_MC={N}  [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(a.out + "_raw.csv", index=False)
    keys = ["n_noise", "alpha", "n_mc", "K"]
    summ = df.groupby(keys).agg(wc_I=("wc_I", "mean"), wc_I_se=("wc_I", "sem"),
                                tau=("tau", "mean"), sigma=("sigma", "mean"),
                                regret=("regret", "mean"),
                                wc_mu=("wc_mu", "mean"), wc_mu_se=("wc_mu", "sem"),
                                wc_mu_tw=("wc_mu_tw", "mean"), wc_mu_tw_se=("wc_mu_tw", "sem"),
                                wc_mu_asimov=("wc_mu_asimov", "mean"),
                                wc_mu_asimov_se=("wc_mu_asimov", "sem")).reset_index()
    summ["e_K"] = [expected_max_normal(k) for k in summ.K]
    summ["wc_gauss"] = summ.sigma ** 2 / np.sqrt(summ.tau ** 2 + summ.sigma ** 2 + 1e-300) * summ.e_K
    if a.paired and os.path.exists(a.paired):
        r = pd.read_csv(a.paired)
        r = r[(r.lik == "bbp") & r.method.isin(["R1_kfold", "R3_fresh"])]
        w = r.pivot_table(index=["n_noise", "alpha", "n_mc", "K", "seed"], columns="method",
                          values="mu_bias_mu1")
        dd = (w["R1_kfold"] - w["R3_fresh"]).groupby(level=[0, 1, 2, 3]).agg(["mean", "sem"])
        dd.columns = ["meas_shift", "meas_shift_se"]
        summ = summ.merge(dd.reset_index(), on=keys, how="left")
    summ.to_csv(a.out + "_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print("\n=== Phase 5b: winner's curse; linearised / template-weighted / exact-Asimov mu-shift ===")
    print("wc_I vs wc_gauss: Gaussian winner's-curse model;  wc_mu vs meas_shift: predicted vs"
          " measured R1-R3 paired mu-hat shift (mu_true=1)")
    print(summ.round(4).to_string(index=False))
    import sklearn; print("versions: sklearn", sklearn.__version__, "numpy", np.__version__)
    print(f"Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
