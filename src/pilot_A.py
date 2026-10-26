#!/usr/bin/env python3
"""
Topic A - Phase 1 mechanism pilot (Week 1, Days 3-4 of the plan).

Question: when analysis designs (here: shallow decision-tree partitions = binnings)
are trained, selected and evaluated on the SAME finite MC, how optimistic is the
reported profiled Fisher information I_eff, and do standard fixes (split, 3-way
split, cross-fitting) remove the problem, and at what cost in true sensitivity?

Toy: 2 physical features (+ n_noise irrelevant uniform features)
  background: truncated exponential in x1, x2 on [0,1]
  signal    : narrow Gaussian at (0.70, 0.65)
  nuisance  : zero-mean background shape bump near the signal (partially mimics it)
Model per bin: lambda_i = mu*s_i + b_i + nu*D_i, Gaussian constraint on nu (Omega=1)
  I_eff = s^T W s - (s^T W D)^2 / (D^T W D + Omega),  W = diag(1/lambda) at mu0=1

Methods (same total MC budget per replica):
  M0 naive   : fit K trees on all MC, estimate on all MC, pick argmax, report it
  M1 split   : fit on A(50%), estimate/select on B(50%), report B estimate
  M2 3way    : fit on A(50%), select on B1(25%), report on B2(25%)   (honest report)
  M3 crossfit: 2-fold out-of-fold score per hyper-config, pick best, refit on all
  M4 lcb     : fit on A, correlated Poisson bootstrap on B, pick by q-quantile
"true" I_eff of any chosen design is computed on a large, fixed oracle sample.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed

S_TOT, B_TOT = 50.0, 1000.0
SIG_MU, SIG_SD = np.array([0.70, 0.65]), 0.08
BKG_RATE = 3.0
NUIS_C, NUIS_SD = np.array([0.62, 0.58]), 0.15


# ---------------- toy physics ----------------
def sample_bkg(rng, n, n_noise):
    u = rng.random((n, 2))
    x = -np.log(1.0 - u * (1.0 - np.exp(-BKG_RATE))) / BKG_RATE
    return np.hstack([x, rng.random((n, n_noise))]).astype(np.float32)


def sample_sig(rng, n, n_noise):
    out = []
    got = 0
    while got < n:
        x = rng.normal(SIG_MU, SIG_SD, size=(2 * n, 2))
        x = x[np.all((x > 0) & (x < 1), axis=1)]
        out.append(x); got += len(x)
    x = np.vstack(out)[:n]
    return np.hstack([x, rng.random((n, n_noise))]).astype(np.float32)


def nuis_raw(x):
    r2 = np.sum((x[:, :2] - NUIS_C) ** 2, axis=1)
    return np.exp(-r2 / (2 * NUIS_SD ** 2))


# ---------------- Fisher information ----------------
def ieff(s, b, d, mu0=1.0, omega=1.0):
    """Works on (..., nbins) arrays. Single nuisance, Gaussian constraint omega."""
    lam = b + mu0 * s
    w = 1.0 / lam
    sws = np.sum(s * s * w, -1)
    swd = np.sum(s * d * w, -1)
    dwd = np.sum(d * d * w, -1)
    return sws - swd ** 2 / (dwd + omega)


class Sample:
    """An MC sample with per-event weights normalised to expected yields."""
    def __init__(self, Xs, Xb, g_b, alpha):
        self.Xs, self.Xb = Xs, Xb
        self.ws = S_TOT / len(Xs)
        self.wb = B_TOT / len(Xb)
        self.gb = alpha * g_b          # dD/dnu per bkg event (relative)

    def subset(self, sl_s, sl_b):
        o = object.__new__(Sample)
        o.Xs, o.Xb = self.Xs[sl_s], self.Xb[sl_b]
        o.ws, o.wb = S_TOT / len(o.Xs), B_TOT / len(o.Xb)
        o.gb = self.gb[sl_b]
        return o


def leaf_map(tree):
    t = tree.tree_
    leaves = np.where(t.children_left == -1)[0]
    m = -np.ones(t.node_count, dtype=np.int64)
    m[leaves] = np.arange(len(leaves))
    return m, len(leaves)


def templates(tree, smp, b_floor_events=1.0):
    m, nb = leaf_map(tree)
    ls, lb = m[tree.apply(smp.Xs)], m[tree.apply(smp.Xb)]
    s = np.bincount(ls, minlength=nb) * smp.ws
    b = np.bincount(lb, minlength=nb) * smp.wb
    d = np.bincount(lb, weights=smp.gb, minlength=nb) * smp.wb
    b = np.maximum(b, b_floor_events * smp.wb)   # naive analyst convention
    return s, b, d


def ieff_on(tree, smp):
    return float(ieff(*templates(tree, smp)))


def ieff_bootstrap(tree, smp, n_boot, rng):
    """Poisson bootstrap; nominal and nuisance templates share event weights
    (correlated resampling, as required for templates built from the same events)."""
    m, nb = leaf_map(tree)
    ls, lb = m[tree.apply(smp.Xs)], m[tree.apply(smp.Xb)]
    Ps = rng.poisson(1.0, size=(len(ls), n_boot)).astype(np.float64)
    Pb = rng.poisson(1.0, size=(len(lb), n_boot)).astype(np.float64)
    Hs = sparse.csr_matrix((np.ones(len(ls)), (ls, np.arange(len(ls)))), shape=(nb, len(ls)))
    Hb = sparse.csr_matrix((np.ones(len(lb)), (lb, np.arange(len(lb)))), shape=(nb, len(lb)))
    s = (Hs @ Ps).T * smp.ws
    b = (Hb @ Pb).T * smp.wb
    d = (Hb @ (Pb * smp.gb[:, None])).T * smp.wb
    b = np.maximum(b, smp.wb)
    return ieff(s, b, d)


# ---------------- design family ----------------
def draw_hparams(rng, K):
    return [dict(max_leaf_nodes=int(rng.integers(6, 41)),
                 min_samples_leaf=int(rng.choice([5, 10, 20, 50])),
                 max_features=float(rng.choice([0.5, 0.8, 1.0])),
                 random_state=int(rng.integers(2 ** 31 - 1))) for _ in range(K)]


def fit_tree(hp, smp):
    X = np.vstack([smp.Xs, smp.Xb])
    y = np.r_[np.ones(len(smp.Xs)), np.zeros(len(smp.Xb))]
    sw = np.r_[np.full(len(smp.Xs), 0.5 / len(smp.Xs)), np.full(len(smp.Xb), 0.5 / len(smp.Xb))]
    return DecisionTreeClassifier(**hp).fit(X, y, sample_weight=sw)


# ---------------- one replica ----------------
def one_rep(seed, n_mc, K, n_noise, alpha, n_boot, q, oracle):
    rng = np.random.default_rng(seed)
    n = n_mc // 2                                  # events per class
    Xs, Xb = sample_sig(rng, n, n_noise), sample_bkg(rng, n, n_noise)
    full = Sample(Xs, Xb, nuis_raw(Xb) - oracle["gmean"], alpha)
    h = n // 2
    A = full.subset(slice(0, h), slice(0, h))
    B = full.subset(slice(h, n), slice(h, n))
    q4 = h // 2
    B1 = B.subset(slice(0, q4), slice(0, q4))
    B2 = B.subset(slice(q4, None), slice(q4, None))
    orc = oracle["smp"][alpha]
    hps = draw_hparams(rng, K)
    rows = []

    def rec(method, tree, reported):
        rows.append(dict(method=method, reported=reported, true=ieff_on(tree, orc)))

    # M0 naive reuse
    T_all = [fit_tree(hp, full) for hp in hps]
    est = np.array([ieff_on(t, full) for t in T_all])
    k = int(np.argmax(est)); rec("M0_naive", T_all[k], est[k])

    # candidates fitted on A (shared by M1, M2, M4 and fold-1 of M3)
    T_A = [fit_tree(hp, A) for hp in hps]
    true_A = np.array([ieff_on(t, orc) for t in T_A])
    estB = np.array([ieff_on(t, B) for t in T_A])
    k = int(np.argmax(estB)); rec("M1_split", T_A[k], estB[k])

    estB1 = np.array([ieff_on(t, B1) for t in T_A])
    k = int(np.argmax(estB1)); rec("M2_3way", T_A[k], ieff_on(T_A[k], B2))

    T_B = [fit_tree(hp, B) for hp in hps]
    estA = np.array([ieff_on(t, A) for t in T_B])
    cf = 0.5 * (estB + estA)
    k = int(np.argmax(cf)); rec("M3_crossfit", fit_tree(hps[k], full), cf[k])

    lcb = np.array([np.quantile(ieff_bootstrap(t, B, n_boot, rng), q) for t in T_A])
    k = int(np.argmax(lcb)); rec("M4_lcb", T_A[k], lcb[k])

    for r in rows:
        r.update(seed=seed, n_mc=n_mc, K=K, n_noise=n_noise, alpha=alpha,
                 oracle_best_A=float(true_A.max()))
    return rows


def build_oracle(n_noise, alphas, n_orc_s, n_orc_b, seed=12345):
    rng = np.random.default_rng(seed)
    Xs, Xb = sample_sig(rng, n_orc_s, n_noise), sample_bkg(rng, n_orc_b, n_noise)
    g = nuis_raw(Xb); gmean = float(g.mean())
    return dict(gmean=gmean, smp={a: Sample(Xs, Xb, g - gmean, a) for a in alphas})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_mc", type=int, nargs="+", default=[2000, 10000])
    p.add_argument("--K", type=int, nargs="+", default=[1, 10, 50])
    p.add_argument("--n_noise", type=int, nargs="+", default=[0, 10])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.3])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--n_boot", type=int, default=100)
    p.add_argument("--q", type=float, default=0.1)
    p.add_argument("--n_orc_s", type=int, default=100_000)
    p.add_argument("--n_orc_b", type=int, default=300_000)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default="results/pilot_A")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(vars(a), open(a.out + "_config.json", "w"), indent=2)

    t0 = time.time(); allrows = []
    for nn in a.n_noise:
        oracle = build_oracle(nn, a.alpha, a.n_orc_s, a.n_orc_b)
        for al in a.alpha:
            for N in a.n_mc:
                for K in a.K:
                    tasks = [delayed(one_rep)(1000 * r + 7, N, K, nn, al, a.n_boot, a.q, oracle)
                             for r in range(a.reps)]
                    for rows in Parallel(n_jobs=a.jobs)(tasks):
                        allrows += rows
                    print(f"done n_noise={nn} alpha={al} N_MC={N} K={K}  [{time.time()-t0:.0f}s]",
                          flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(a.out + "_raw.csv", index=False)

    df["optimism"] = df.reported / df.true - 1.0
    ref = df[df.method == "M0_naive"].groupby(["n_noise", "alpha", "n_mc", "K"]).oracle_best_A.mean()
    g = df.groupby(["n_noise", "alpha", "n_mc", "K", "method"])
    summ = g.agg(true_I=("true", "mean"), true_I_se=("true", "sem"),
                 optimism=("optimism", "mean"), optimism_se=("optimism", "sem")).reset_index()
    summ.to_csv(a.out + "_summary.csv", index=False)
    pd.set_option("display.width", 160); pd.set_option("display.max_rows", 500)
    print("\n=== Summary (true_I: higher = better design; optimism = reported/true - 1) ===")
    print(summ.round(3).to_string(index=False))
    print(f"\nTotal wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
