#!/usr/bin/env python3
"""
Topic A - Phase 3: does adaptive design on finite MC bias the FINAL FIT?

For each replica we obtain designs (binnings) from several methods, fill templates
from a given MC sample, then generate pseudo-data from ORACLE expectations and run a
real Poisson profile-likelihood fit.

Template sources (design -> templates):
  M0        : K candidates fit on full MC, selected on full MC, templates from full MC
  M0_fresh  : same design as M0, templates from an INDEPENDENT fresh MC of equal size
  M1        : designs fit on A, selected on B, templates from B
  M3        : 2-fold cross-fit selection, refit on full MC, templates from full MC
  P_fixed   : fixed reference design, templates from full MC

Likelihoods:
  plugin : lambda = mu*s + b + nu*d,           nu ~ N(nu0, 1)
  bb     : lambda = mu*s + gamma_i*(b + nu*d), gamma_i ~ N(1, 1/tau_i), tau_i = n_bkgMC_i + 1/2
  bbp    : full Barlow-Beeston (1993): lambda = mu*s + beta_i*(1 + nu*delta_i), beta_i free >= 0,
           delta_i = d_i/bhat_i, MC count m_i ~ Poisson(beta_i / w_b) as auxiliary measurement;
           no pseudo-count.
           (Barlow-Beeston-lite; gamma profiled analytically bin by bin)
Pseudo-data: n_i ~ Poisson(mu_true*s_i + b_i) with oracle s, b; nu0 ~ N(0,1) per toy.
Free fits impose the physical bound mu >= 0 (standard for discovery tests; q0 ~ 1/2 delta + 1/2 chi2_1).
Metrics: mu-hat bias, 68% profile-likelihood coverage (mu_true=1 and 0),
         false-positive rate Z0>2 under mu_true=0 (nominal 2.28%).
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_A_v3 as P


# ---------------- vectorised profile-likelihood fitter ----------------
class Fitter:
    """Profiled NLL for T toys at once. Arrays: s,b,d (nbins,), n (T,nbins), nu0 (T,)."""
    def __init__(self, s, b, d, tau, n, nu0, lik):
        self.s, self.b, self.d, self.tau = s, b, d, tau
        self.n, self.nu0, self.lik = n, nu0, lik
        self.mu_lb = True          # physical bound mu >= 0 in free fits
        if lik == "bbp":           # tau = m + 1/2 and b = (m + 1/2) * w_b  =>  recover m, w_b
            self.m = tau - 0.5
            self.wb = b / tau
            bplug = self.m * self.wb
            self.delta = np.where(self.m > 0, d / np.where(self.m > 0, bplug, 1.0), 0.0)

    def _gamma(self, a, c):
        """Profiled MC-stat factor for lambda = a + gamma*c, gamma ~ N(1, 1/tau).
        Solves tau*c*g^2 + [(c-tau)*c + tau*a]*g + (c-tau)*a - n*c = 0 (positive root)."""
        t = self.tau
        A = t * c
        B = (c - t) * c + t * a
        C = (c - t) * a - self.n * c
        g = (-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)
        return np.maximum(g, 1e-9)

    # ---------- full Poisson Barlow-Beeston ----------
    # lambda_i = mu*s_i + beta_i * q_i,  q_i = 1 + nu*delta_i  (shape nuisance acts on background),
    # delta_i = d_i / bhat_i (relative shape from MC; 0 where m_i = 0), aux m_i ~ Pois(beta_i / w_i).
    def _beta(self, a, q):
        """Profiled beta >= 0: root of (q+1/w) q b^2 + [(q+1/w) a - (n+m) q] b - m a = 0.
        With a = mu*s >= 0 the positive root is unique for m > 0; m = 0 -> max(0, linear root)."""
        k = q + 1.0 / self.wb; m = self.m
        A = k * q
        B = k * a - (self.n + m) * q
        C = -m * a
        beta = (-B + np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)
        beta_m0 = np.maximum(0.0, (self.n * q - k * a) / A)
        return np.maximum(np.where(m > 0, beta, beta_m0), 0.0)

    def _nll_bbp(self, mu, nu):
        a = mu[:, None] * self.s
        q = 1.0 + nu[:, None] * self.delta
        bad = np.any(q <= 1e-9, 1)
        q = np.maximum(q, 1e-9)
        beta = self._beta(a, q)
        lam = a + beta * q
        bad |= np.any((lam <= 1e-12) & (self.n > 0), 1)
        lam = np.maximum(lam, 1e-300)
        bl = np.where(self.m > 0, self.m * np.log(np.maximum(beta, 1e-300)), 0.0)
        val = np.sum(lam - self.n * np.log(lam) + beta / self.wb - bl, 1)
        return np.where(bad, np.inf, val + 0.5 * (nu - self.nu0) ** 2)

    def _derivs_bbp(self, mu, nu):
        s, dl = self.s, self.delta
        a = mu[:, None] * s
        q = np.maximum(1.0 + nu[:, None] * dl, 1e-9)
        beta = self._beta(a, q)
        lam = np.maximum(a + beta * q, 1e-12)
        r = 1.0 - self.n / lam; w = self.n / lam ** 2
        bd = beta * dl                                  # d lambda / d nu
        gm = np.sum(s * r, 1); gn = np.sum(bd * r, 1) + (nu - self.nu0)
        hmm = np.sum(s * s * w, 1); hnn = np.sum(bd * bd * w, 1) + 1.0; hmn = np.sum(s * bd * w, 1)
        interior = beta > 1e-12
        hbb = q * q * w + np.where(self.m > 0, self.m / np.maximum(beta, 1e-12) ** 2, 0.0)
        cm = np.where(interior, s * q * w, 0.0)
        cn = np.where(interior, bd * q * w + dl * r, 0.0)
        hbb = np.where(interior, hbb, 1.0)
        hmm = hmm - np.sum(cm * cm / hbb, 1)
        hnn = hnn - np.sum(cn * cn / hbb, 1)
        hmn = hmn - np.sum(cm * cn / hbb, 1)
        return gm, gn, hmm, hnn, hmn

    def nll(self, mu, nu):
        if self.lik == "bbp":
            return self._nll_bbp(mu, nu)
        a = mu[:, None] * self.s
        c = self.b + nu[:, None] * self.d                  # background incl. shape nuisance
        pen = 0.5 * (nu - self.nu0) ** 2
        if self.lik == "plugin":
            lam = a + c
            bad = np.any(lam <= 1e-12, 1)
            lam = np.maximum(lam, 1e-12)
            return np.where(bad, np.inf, np.sum(lam - self.n * np.log(lam), 1) + pen)
        bad = np.any(c <= 1e-12, 1)
        c = np.maximum(c, 1e-12)
        g = self._gamma(a, c)
        lam = a + g * c
        bad |= np.any(lam <= 1e-12, 1)
        lam = np.maximum(lam, 1e-12)
        return np.where(bad, np.inf,
                        np.sum(lam - self.n * np.log(lam) + 0.5 * self.tau * (g - 1) ** 2, 1) + pen)

    def derivs(self, mu, nu):
        """Analytic gradient/Hessian of the profiled NLL in (mu, nu).
        bb: lambda = mu*s + gamma*(b + nu*d); gamma profiled exactly via Schur complement."""
        if self.lik == "bbp":
            return self._derivs_bbp(mu, nu)
        s, d = self.s, self.d
        a = mu[:, None] * s
        c = np.maximum(self.b + nu[:, None] * d, 1e-12)
        if self.lik == "plugin":
            lam = np.maximum(a + c, 1e-12)
            r = 1.0 - self.n / lam; w = self.n / lam ** 2
            return (np.sum(s * r, 1), np.sum(d * r, 1) + (nu - self.nu0),
                    np.sum(s * s * w, 1), np.sum(d * d * w, 1) + 1.0, np.sum(s * d * w, 1))
        g = self._gamma(a, c)
        lam = np.maximum(a + g * c, 1e-12)
        r = 1.0 - self.n / lam; w = self.n / lam ** 2
        gd = g * d
        gm = np.sum(s * r, 1)
        gn = np.sum(gd * r, 1) + (nu - self.nu0)
        hmm = np.sum(s * s * w, 1); hnn = np.sum(gd * gd * w, 1) + 1.0; hmn = np.sum(s * gd * w, 1)
        hgg = c * c * w + self.tau
        hmg = s * c * w
        hng = gd * c * w + d * r
        hmm = hmm - np.sum(hmg * hmg / hgg, 1)
        hnn = hnn - np.sum(hng * hng / hgg, 1)
        hmn = hmn - np.sum(hmg * hng / hgg, 1)
        return gm, gn, hmm, hnn, hmn

    def fit(self, mu0=None, fix_mu=None, iters=60):
        T = len(self.nu0)
        nu = np.zeros(T)
        if fix_mu is not None:
            mu = np.full(T, float(fix_mu))
            for _ in range(iters):
                f0 = self.nll(mu, nu)
                _, gn, _, hnn, _ = self.derivs(mu, nu)
                step = -gn / np.maximum(hnn, 1e-8)
                step = np.nan_to_num(np.clip(step, -1, 1))
                nu = self._backtrack(mu, nu, np.zeros_like(step), step, f0)[1]
            return mu, nu, self.nll(mu, nu)
        mu = np.full(T, 1.0) if mu0 is None else mu0.copy()
        for _ in range(iters):
            f0 = self.nll(mu, nu)
            gm, gn, hmm, hnn, hmn = self.derivs(mu, nu)
            det = hmm * hnn - hmn ** 2
            ok = (det > 1e-12) & (hmm > 0)
            sd = np.where(ok, det, 1.0)
            dmu = np.where(ok, -(hnn * gm - hmn * gn) / sd, -gm / np.maximum(hmm, 1e-3))
            dnu = np.where(ok, -(-hmn * gm + hmm * gn) / sd, -gn / np.maximum(hnn, 1e-3))
            dmu = np.nan_to_num(np.clip(dmu, -1, 1)); dnu = np.nan_to_num(np.clip(dnu, -1, 1))
            mu, nu = self._backtrack(mu, nu, dmu, dnu, f0)
        return mu, nu, self.nll(mu, nu)

    def fit_multistart(self, starts=(1.0, 0.3, 2.0)):
        """Free fit with the physical bound mu >= 0.
        Projected Newton from several starts, plus the boundary solution mu = 0
        (conditional fit in nu); keep the lowest NLL."""
        best = list(self.fit(fix_mu=0.0))
        for m0 in starts:
            mu, nu, f = self.fit(mu0=np.full(len(self.nu0), m0))
            if best is None:
                best = [mu, nu, f]
            else:
                better = f < best[2]
                best[0] = np.where(better, mu, best[0]); best[1] = np.where(better, nu, best[1])
                best[2] = np.where(better, f, best[2])
        return tuple(best)

    def _backtrack(self, mu, nu, dmu, dnu, f0, n_half=12):
        """Accept the largest step (1, 1/2, 1/4, ...) that does not increase the NLL."""
        acc = np.zeros(len(mu), bool)
        mu_new, nu_new = mu.copy(), nu.copy()
        t = 1.0
        for _ in range(n_half):
            m, v = np.maximum(mu + t * dmu, 0.0) if self.mu_lb else mu + t * dmu, nu + t * dnu
            fn = self.nll(m, v)
            take = (~acc) & np.isfinite(fn) & (fn <= f0 + 1e-10)
            mu_new[take], nu_new[take] = m[take], v[take]
            acc |= take
            if acc.all():
                break
            t *= 0.5
        return mu_new, nu_new


def run_toys(design_tmpl, truth, n_toys, lik, rng, mu_true):
    s, b, d, nbk = design_tmpl
    s_true, b_true = truth
    tau = nbk + 0.5
    lam_true = mu_true * s_true + b_true
    n = rng.poisson(lam_true, size=(n_toys, len(lam_true))).astype(float)
    nu0 = rng.normal(0, 1, n_toys)
    F = Fitter(s, np.maximum(b, 1e-9), d, tau, n, nu0, lik)
    mu_hat, _, f_hat = F.fit_multistart()
    _, _, f_true = F.fit(fix_mu=mu_true)
    q_true = 2 * (f_true - f_hat)
    out = dict(mu_bias=float(np.mean(mu_hat - mu_true)),
               mu_sd=float(np.std(mu_hat)),
               cov68=float(np.mean(q_true < 1.0)))
    if mu_true == 0:
        z0 = np.sqrt(np.maximum(q_true, 0)) * (mu_hat > 0)
        out["fpr_z2"] = float(np.mean(z0 > 2.0))
    return out


def tmpl_from(tree, smp, lik, b_floor):
    ns, nb_, g = P.counts(tree, smp)
    s = ns * smp.ws; d = g * smp.wb
    b = (nb_ + 0.5) * smp.wb if lik == "bb" else np.maximum(nb_ * smp.wb, b_floor * smp.wb)
    return s, b, d, nb_


def one_rep(seed, n_mc, K, n_noise, alpha, oracle, n_toys, min_leaf, b_floor, sel_obj):
    rng = np.random.default_rng(seed)
    n = n_mc // 2
    mk = lambda Xs, Xb: P.Sample(Xs, Xb, P.nuis_raw(Xb) - oracle["gmean"], alpha, b_floor, sel_obj)
    full = mk(P.sample_sig(rng, n, n_noise), P.sample_bkg(rng, n, n_noise))
    fresh = mk(P.sample_sig(rng, n, n_noise), P.sample_bkg(rng, n, n_noise))
    h = n // 2
    A = full.subset(slice(0, h), slice(0, h)); B = full.subset(slice(h, n), slice(h, n))
    hps = P.draw_hparams(rng, K, min_leaf)

    T_all = [P.fit_tree(hp, full) for hp in hps]
    k0 = int(np.argmax([P.ieff_on(t, full) for t in T_all]))
    T_A = [P.fit_tree(hp, A) for hp in hps]
    estB = np.array([P.ieff_on(t, B) for t in T_A]); k1 = int(np.argmax(estB))
    T_B = [P.fit_tree(hp, B) for hp in hps]
    estA = np.array([P.ieff_on(t, A) for t in T_B])
    k3 = int(np.argmax(0.5 * (estA + estB)))
    T3 = P.fit_tree(hps[k3], full)

    plans = [("M0", T_all[k0], full), ("M0_fresh", T_all[k0], fresh), ("M1", T_A[k1], B),
             ("M3", T3, full), ("P_fixed", oracle["ref_tree"], full)]
    orc = oracle["smp"][alpha]
    rows = []
    for name, tree, tsmp in plans:
        s_o, b_o, _ = P.templates(tree, orc)
        diag = P.bin_diag(tree, tsmp)
        for lik in ("plugin", "bb"):
            tm = tmpl_from(tree, tsmp, lik, b_floor)
            r1 = run_toys(tm, (s_o, b_o), n_toys, lik, rng, 1.0)
            r0 = run_toys(tm, (s_o, b_o), n_toys, lik, rng, 0.0)
            rows.append(dict(method=name, lik=lik, seed=seed, n_mc=n_mc, K=K, n_noise=n_noise,
                             alpha=alpha, **{f"{k}_mu1": v for k, v in r1.items()},
                             **{f"{k}_mu0": v for k, v in r0.items()}, **diag))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_mc", type=int, nargs="+", default=[2000, 10000])
    p.add_argument("--K", type=int, nargs="+", default=[1, 50])
    p.add_argument("--n_noise", type=int, nargs="+", default=[0, 10])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.3])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--n_toys", type=int, default=300)
    p.add_argument("--min_leaf", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--b_floor", type=float, default=1.0)
    p.add_argument("--sel_objective", choices=["plugin", "mcaware"], default="plugin")
    p.add_argument("--n_orc_s", type=int, default=100_000)
    p.add_argument("--n_orc_b", type=int, default=300_000)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default="results/phase3")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(vars(a), open(a.out + "_config.json", "w"), indent=2)

    t0 = time.time(); allrows = []
    for nn in a.n_noise:
        oracle = P.build_oracle(nn, a.alpha, a.n_orc_s, a.n_orc_b)
        for al in a.alpha:
            for N in a.n_mc:
                for K in a.K:
                    tasks = [delayed(one_rep)(1000 * r + 7, N, K, nn, al, oracle, a.n_toys,
                                              a.min_leaf, a.b_floor, a.sel_objective)
                             for r in range(a.reps)]
                    for rows in Parallel(n_jobs=a.jobs)(tasks):
                        allrows += rows
                    print(f"done n_noise={nn} alpha={al} N_MC={N} K={K}  [{time.time()-t0:.0f}s]",
                          flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(a.out + "_raw.csv", index=False)
    keys = ["n_noise", "alpha", "n_mc", "K", "lik", "method"]
    summ = df.groupby(keys).agg(
        bias_mu1=("mu_bias_mu1", "mean"), bias_mu1_se=("mu_bias_mu1", "sem"),
        sd_mu1=("mu_sd_mu1", "mean"), cov68_mu1=("cov68_mu1", "mean"),
        bias_mu0=("mu_bias_mu0", "mean"), bias_mu0_se=("mu_bias_mu0", "sem"),
        cov68_mu0=("cov68_mu0", "mean"), fpr_z2=("fpr_z2_mu0", "mean"),
        med_bkg_bin=("med_bkg_mc_per_bin", "mean")).reset_index()
    summ.to_csv(a.out + "_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print("\n=== Phase 3 summary ===")
    print("Nominal with perfect templates: mu=1 -> bias 0, cov68 0.683 ; "
          "mu=0 (bound mu>=0) -> bias ~0.4*sd(mu_hat), cov68 ~0.84, fpr_z2 0.0228")
    print(summ.round(3).to_string(index=False))
    import sklearn; print("versions: sklearn", sklearn.__version__, "numpy", np.__version__)
    print(f"Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
