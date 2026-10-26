#!/usr/bin/env python3
"""
Performance benchmark of the vectorised profile-likelihood fitter (src/phase3_fit.py).

For templates built from the toy model of the paper (N_MC simulated events, B quantile
bins of a signal-like score) and background-only pseudo-experiments, the script measures

  (1) vectorised fitter: Fitter(...).fit_multistart() + Fitter(...).fit(fix_mu=0) for T
      pseudo-experiments at once, i.e. exactly the two fits per pseudo-experiment that
      round2_run.py performs to obtain q0;
  (2) reference A: the same profiled NLL minimised for each pseudo-experiment separately
      with scipy's bounded Nelder-Mead (the procedure of check_fitter.py), free fit
      from 5 starts and mu = 0 fit from 3 starts;
  (3) reference B: the unprofiled likelihood, Eq. (2) of the paper, minimised for each
      pseudo-experiment separately over all B + 2 parameters (mu, nu, beta_1..beta_B)
      with scipy's L-BFGS-B and analytic gradients, from the same starts as the
      vectorised fitter (mu = 1, 0.3, 2 and the boundary mu = 0);

and reports the time per pseudo-experiment (both fits) and the largest amount by which
the vectorised minimum exceeds the reference minima (negative = vectorised is lower).

Usage:
    python benchmarks/bench_fitter.py                   # default scan, a few minutes on 1 core
    python benchmarks/bench_fitter.py --quick           # ~1 min
    python benchmarks/bench_fitter.py --bins 16 --toys 300 --n_ref 100

Run single-threaded for comparable numbers (the script sets OMP/MKL/OPENBLAS threads to 1).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, csv, json, platform, sys, time
import warnings
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import pilot_A_v3 as P          # noqa: E402  (toy model: generators, S_TOT, B_TOT)
import phase3_fit as F3         # noqa: E402  (vectorised Fitter)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- templates
def score(X):
    """Signal-like score: minus the distance to the signal centre in the informative plane."""
    return -np.sqrt(np.sum((X[:, :2] - P.SIG_MU) ** 2, axis=1))


def make_templates(rng, n_mc, nbins, alpha=0.3, n_noise=10, n_ref=(100_000, 300_000)):
    """Templates (s, b, d, m) as in round2_run.channel() with frac = 1, and oracle yields."""
    # mean of the shape bump under the background density (as build_oracle does)
    gmean = float(np.mean(P.nuis_raw(P.sample_bkg(np.random.default_rng(12345), 300_000, 0))))
    n = n_mc // 2
    Xs, Xb = P.sample_sig(rng, n, n_noise), P.sample_bkg(rng, n, n_noise)
    edges = np.quantile(score(Xs), np.linspace(0, 1, nbins + 1)[1:-1])
    lab = lambda X: np.searchsorted(edges, score(X))
    ns = np.bincount(lab(Xs), minlength=nbins).astype(float)
    m = np.bincount(lab(Xb), minlength=nbins).astype(float)
    g = np.bincount(lab(Xb), weights=alpha * (P.nuis_raw(Xb) - gmean), minlength=nbins)
    ws, wb = P.S_TOT / n, P.B_TOT / n
    s, b, d = ns * ws, (m + 0.5) * wb, g * wb
    # oracle yields from a large reference sample
    R = np.random.default_rng(54321)
    Rs, Rb = P.sample_sig(R, n_ref[0], 0), P.sample_bkg(R, n_ref[1], 0)
    s_o = np.bincount(lab(np.hstack([Rs, np.zeros((len(Rs), n_noise))])), minlength=nbins) * P.S_TOT / n_ref[0]
    b_o = np.bincount(lab(np.hstack([Rb, np.zeros((len(Rb), n_noise))])), minlength=nbins) * P.B_TOT / n_ref[1]
    return (s, b, d, m), (s_o.astype(float), b_o.astype(float))


def pseudo_data(rng, truth, T, mu_true=0.0):
    s_o, b_o = truth
    n = rng.poisson(mu_true * s_o + b_o, size=(T, len(b_o))).astype(float)
    nu0 = rng.normal(0.0, 1.0, T)
    return n, nu0


# ----------------------------------------------------------------------------- fitters
def vectorised(tm, n, nu0):
    s, b, d, m = tm
    Fi = F3.Fitter(s, b, d, m + 0.5, n, nu0, "bbp")
    mu, _, fh = Fi.fit_multistart()
    _, _, f0 = Fi.fit(fix_mu=0.0)
    return mu, fh, f0


def ref_nelder_mead(tm, n, nu0):
    s, b, d, m = tm
    T = len(nu0)
    fh, f0 = np.empty(T), np.empty(T)
    for i in range(T):
        G = F3.Fitter(s, b, d, m + 0.5, n[i:i + 1], nu0[i:i + 1], "bbp")
        fn = lambda x: float(G.nll(np.array([x[0]]), np.array([x[1]]))[0])
        fh[i] = min(minimize(fn, x0, method="Nelder-Mead", bounds=[(0, None), (None, None)],
                             options=dict(xatol=1e-8, fatol=1e-10, maxiter=8000)).fun
                    for x0 in ([1, 0], [0, 0], [0.5, 1], [0.5, -1], [0.01, 0.5]))
        f1 = lambda x: float(G.nll(np.array([0.0]), np.array([x[0]]))[0])
        f0[i] = min(minimize(f1, [x0], method="Nelder-Mead",
                             options=dict(xatol=1e-8, fatol=1e-10)).fun for x0 in (0, 1, -1))
    return fh, f0


class FullNLL:
    """Unprofiled NLL of Eq. (2) in (mu, nu, beta_1..beta_B), for one pseudo-experiment."""
    def __init__(self, tm, n, nu0):
        s, b, d, m = tm
        self.s, self.m, self.n, self.nu0 = s, m, n, nu0
        self.wb = b / (m + 0.5)
        self.delta = np.where(m > 0, d / np.where(m > 0, m * self.wb, 1.0), 0.0)
        dp, dm = self.delta.max(), self.delta.min()      # domain q_i = 1 + nu*delta_i > 0
        self.nu_bounds = (-1.0 / dp * 0.999 if dp > 0 else None, -1.0 / dm * 0.999 if dm < 0 else None)

    def f_g(self, x, fix_mu=None):
        if fix_mu is None:
            mu, nu, beta = x[0], x[1], x[2:]
        else:
            mu, nu, beta = fix_mu, x[0], x[1:]
        q = 1.0 + nu * self.delta
        lam = np.maximum(mu * self.s + beta * q, 1e-300)
        bt = np.maximum(beta, 1e-300)
        f = np.sum(lam - self.n * np.log(lam) + beta / self.wb - self.m * np.log(bt)) \
            + 0.5 * (nu - self.nu0) ** 2
        r = 1.0 - self.n / lam
        g_mu = np.sum(r * self.s)
        g_nu = np.sum(r * beta * self.delta) + (nu - self.nu0)
        g_b = r * q + 1.0 / self.wb - self.m / bt
        g = np.concatenate([[g_nu], g_b]) if fix_mu is not None else np.concatenate([[g_mu, g_nu], g_b])
        return f, g

    def fit(self, mu0=None, fix_mu=None):
        b0 = np.maximum(self.m, 0.5) * self.wb
        bb = [(1e-10, None)] * len(self.s)
        if fix_mu is None:
            x0, bounds = np.concatenate([[mu0, 0.0], b0]), [(0.0, None), self.nu_bounds] + bb
            fun = lambda x: self.f_g(x)
        else:
            x0, bounds = np.concatenate([[0.0], b0]), [self.nu_bounds] + bb
            fun = lambda x: self.f_g(x, fix_mu)
        r = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9))
        return r.fun


def ref_lbfgsb(tm, n, nu0):
    T = len(nu0)
    fh, f0 = np.empty(T), np.empty(T)
    for i in range(T):
        L = FullNLL(tm, n[i], nu0[i])
        f0[i] = L.fit(fix_mu=0.0)
        fh[i] = min([f0[i]] + [L.fit(mu0=m0) for m0 in (1.0, 0.3, 2.0)])
    return fh, f0


def timed(fn, *args, repeat=1):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter(); out = fn(*args); best = min(best, time.perf_counter() - t0)
    return best, out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_mc", type=int, default=2000)
    p.add_argument("--bins", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--toys", type=int, nargs="+", default=[1, 10, 100, 300, 1000, 3000])
    p.add_argument("--n_ref", type=int, default=100, help="pseudo-experiments for the references")
    p.add_argument("--repeat", type=int, default=3, help="repetitions of the vectorised timing (best of)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "results", "bench_fitter"))
    a = p.parse_args()
    if a.quick:
        a.bins, a.toys, a.n_ref, a.repeat = [8, 16], [1, 100, 1000], 20, 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    env = dict(python=platform.python_version(), numpy=np.__version__,
               scipy=__import__("scipy").__version__, machine=platform.machine(),
               processor=platform.processor() or "", platform=platform.platform(),
               threads=os.environ.get("OMP_NUM_THREADS"))
    try:
        with open("/proc/cpuinfo") as fh_:
            env["cpu"] = next(l.split(":", 1)[1].strip() for l in fh_ if l.startswith("model name"))
    except Exception:
        pass
    print("environment:", json.dumps(env))

    rows = []
    rng = np.random.default_rng(a.seed)
    print(f"\n{'B':>3} {'method':<28} {'T':>5} {'time/toy [ms]':>14} {'toys/s':>10} {'speed-up':>9} {'max dNLL':>10}")
    for B in a.bins:
        tm, truth = make_templates(rng, a.n_mc, B)
        # references on n_ref pseudo-experiments
        n, nu0 = pseudo_data(rng, truth, a.n_ref)
        t_v, (mu_v, fh_v, f0_v) = timed(vectorised, tm, n, nu0, repeat=a.repeat)
        t_nm, (fh_nm, f0_nm) = timed(ref_nelder_mead, tm, n, nu0)
        t_lb, (fh_lb, f0_lb) = timed(ref_lbfgsb, tm, n, nu0)
        for name, t, fh, f0 in (("reference A: Nelder-Mead", t_nm, fh_nm, f0_nm),
                                ("reference B: L-BFGS-B, B+2", t_lb, fh_lb, f0_lb)):
            dmax = float(max(np.max(fh_v - fh), np.max(f0_v - f0)))
            rows.append(dict(bins=B, method=name, toys=a.n_ref, sec_per_toy=t / a.n_ref,
                             max_dnll_vectorised_minus_ref=dmax))
            print(f"{B:>3} {name:<28} {a.n_ref:>5} {1e3 * t / a.n_ref:>14.3f} {a.n_ref / t:>10.1f}"
                  f" {'':>9} {dmax:>10.2e}")
        # vectorised scan in T
        for T in a.toys:
            n, nu0 = pseudo_data(rng, truth, T)
            t, _ = timed(vectorised, tm, n, nu0, repeat=a.repeat)
            su_nm, su_lb = (t_nm / a.n_ref) / (t / T), (t_lb / a.n_ref) / (t / T)
            rows.append(dict(bins=B, method="vectorised Newton", toys=T, sec_per_toy=t / T,
                             speedup_vs_A=su_nm, speedup_vs_B=su_lb))
            print(f"{B:>3} {'vectorised Newton':<28} {T:>5} {1e3 * t / T:>14.3f} {T / t:>10.1f}"
                  f" {su_lb:>8.0f}x")
        print()

    with open(a.out + ".csv", "w", newline="") as fh_:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(fh_, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(a.out + "_env.json", "w") as fh_:
        json.dump(dict(env, args=vars(a)), fh_, indent=1)
    print("speed-up column: relative to reference B (L-BFGS-B over all B+2 parameters)")
    print("wrote", a.out + ".csv", "and", a.out + "_env.json")


if __name__ == "__main__":
    main()
