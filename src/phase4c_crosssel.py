#!/usr/bin/env python3
"""
Topic A - Phase 4a: does SELECTION reuse still cause spurious signals in a REALISTIC
k-fold workflow, where every MC event is scored by a model that never saw it?

Design family: HistGradientBoostingClassifier (random hyper-parameters) -> score ->
nbins bins with edges at quantiles of the out-of-fold SIGNAL score distribution.

k-fold convention (as in real analyses): event i belongs to fold i % F; events of fold f
(MC and data alike) are scored by the model trained on all other folds. The oracle truth,
the templates and the pseudo-data therefore use exactly the same mapping x -> bin.

Workflows (same MC budget per replica):
  R0_naive    : single model on all MC; select on same MC; templates from same MC
  R1_kfold    : F-fold out-of-fold scores; select on OOF templates; fit with SAME OOF templates
  R2_holdout  : 20% of MC reserved for selection only; k-fold + templates on the other 80%
  R3_fresh    : R1 design, templates from an independent fresh MC of equal size (gold standard)

Phase 4b additions (no extra MC):
  R4_bootsel  : R1 design; selection-aware Poisson bootstrap of the SELECTION step only
                (OOF scores fixed). Among bootstrap worlds that re-select k*, the mean shift of
                the k* signal/background counts relative to the original counts estimates the
                selection-induced template bias; it is subtracted from the k* templates.
                K=1 -> no correction (built-in control).
  R5_foldback : hyper-parameters selected on the 20% holdout (as R2), then the chosen config is
                re-run as k-fold on ALL MC and templates use all OOF counts.

Phase 4c addition:
  R6_crosssel : cross-selection. MC and data are split by event parity into halves h=0,1.
                Data half h is analysed with the design selected on MC half 1-h, with templates
                from MC half h. Joint fit of the two channels (shared mu, nu; own BB gammas).
                No event is used both to select a design and to build that design's templates;
                all MC is used for templates; selection uses 50% of MC.
                IMPORTANT: bins are always computed on the FULL sample first (k-fold assignment
                depends on the event position) and only then split by parity.
Selection objective: MC-aware Fisher (v3). Fit likelihood: Barlow-Beeston-lite (validated).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # avoid oversubscription with joblib
import argparse, json, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_A_v3 as P
import phase3_fit as F3


class Smp:
    """Minimal MC sample: features, per-event weights normalised to yields, nuisance derivative."""
    def __init__(self, Xs, Xb, gb, alpha, objective="mcaware"):
        self.Xs, self.Xb = Xs, Xb
        self.ws, self.wb = P.S_TOT / len(Xs), P.B_TOT / len(Xb)
        self.gb = alpha * gb
        self.objective, self.bfloor = objective, 1.0

    def part(self, sl):
        o = object.__new__(Smp)
        o.Xs, o.Xb, o.gb = self.Xs[sl], self.Xb[sl], self.gb[sl]
        o.ws, o.wb = P.S_TOT / len(o.Xs), P.B_TOT / len(o.Xb)
        o.objective, o.bfloor = self.objective, self.bfloor
        return o


def draw_hparams(rng, K):
    return [dict(learning_rate=float(rng.choice([0.05, 0.1, 0.2])),
                 max_iter=int(rng.choice([50, 100, 200])),
                 max_leaf_nodes=int(rng.choice([7, 15, 31])),
                 min_samples_leaf=int(rng.choice([5, 20, 50])),
                 nbins=int(rng.integers(4, 17)),
                 random_state=int(rng.integers(2 ** 31 - 1))) for _ in range(K)]


def train(hp, Xs, Xb):
    X = np.vstack([Xs, Xb]); y = np.r_[np.ones(len(Xs)), np.zeros(len(Xb))]
    sw = np.r_[np.full(len(Xs), 0.5 / len(Xs)), np.full(len(Xb), 0.5 / len(Xb))]
    kw = {k: v for k, v in hp.items() if k != "nbins"}
    return HistGradientBoostingClassifier(early_stopping=False, **kw).fit(X, y, sample_weight=sw)


class Design:
    """F fold models (F=1 means a single model applied to everything) + score bin edges."""
    def __init__(self, models, edges):
        self.models, self.edges = models, edges

    def score(self, X):
        Fn = len(self.models)
        if Fn == 1:
            return self.models[0].predict_proba(X)[:, 1]
        out = np.empty(len(X)); idx = np.arange(len(X)) % Fn
        for f, m in enumerate(self.models):
            sel = idx == f
            if sel.any():
                out[sel] = m.predict_proba(X[sel])[:, 1]
        return out

    def bins(self, X):
        return np.digitize(self.score(X), self.edges)

    @property
    def nb(self):
        return len(self.edges) + 1


def edges_from(sig_scores, nbins):
    q = np.quantile(sig_scores, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.unique(q)


def build_kfold(hp, smp, Fn):
    """Models m_f trained on folds != f; OOF scores; edges from OOF signal scores."""
    fs, fb = np.arange(len(smp.Xs)) % Fn, np.arange(len(smp.Xb)) % Fn
    models = [train(hp, smp.Xs[fs != f], smp.Xb[fb != f]) for f in range(Fn)]
    d = Design(models, None)
    d.edges = edges_from(d.score(smp.Xs), hp["nbins"])
    return d


def build_single(hp, smp):
    d = Design([train(hp, smp.Xs, smp.Xb)], None)
    d.edges = edges_from(d.score(smp.Xs), hp["nbins"])
    return d


def counts(design, smp):
    ls, lb = design.bins(smp.Xs), design.bins(smp.Xb)
    nb = design.nb
    return (np.bincount(ls, minlength=nb).astype(float),
            np.bincount(lb, minlength=nb).astype(float),
            np.bincount(lb, weights=smp.gb, minlength=nb))


def est(design, smp):
    return float(P.est_from_counts(*counts(design, smp), smp))


def bb_templates(design, smp):
    ns, nb, g = counts(design, smp)
    return ns * smp.ws, (nb + 0.5) * smp.wb, g * smp.wb, nb


def bootsel_correction(designs, smp, k_star, n_boot, rng):
    """Selection-aware bootstrap correction for the templates of the selected design k_star.
    Returns (delta_ns, delta_nb, n_hit):
      mean(count_boot | boot re-selects k_star) - mean(count_boot over all boots)
    (control variate: exactly zero when K=1, lower Monte Carlo noise otherwise)."""
    from scipy import sparse
    Ps = rng.poisson(1.0, size=(len(smp.Xs), n_boot)).astype(float)
    Pb = rng.poisson(1.0, size=(len(smp.Xb), n_boot)).astype(float)
    Ib = np.empty((len(designs), n_boot)); keep = {}
    for c, dsg in enumerate(designs):
        ls, lb = dsg.bins(smp.Xs), dsg.bins(smp.Xb); nb = dsg.nb
        Hs = sparse.csr_matrix((np.ones(len(ls)), (ls, np.arange(len(ls)))), shape=(nb, len(ls)))
        Hb = sparse.csr_matrix((np.ones(len(lb)), (lb, np.arange(len(lb)))), shape=(nb, len(lb)))
        ns_b = np.asarray((Hs @ Ps).T); nb_b = np.asarray((Hb @ Pb).T)
        g_b = np.asarray((Hb @ (Pb * smp.gb[:, None])).T)
        Ib[c] = P.est_from_counts(ns_b, nb_b, g_b, smp)
        if c == k_star:
            keep = dict(ns_b=ns_b, nb_b=nb_b, ns0=np.bincount(ls, minlength=nb).astype(float),
                        nb0=np.bincount(lb, minlength=nb).astype(float))
    hit = np.argmax(Ib, 0) == k_star
    if not hit.any():
        z = np.zeros_like(keep["ns0"]); return z, z, 0
    return (keep["ns_b"][hit].mean(0) - keep["ns_b"].mean(0),
            keep["nb_b"][hit].mean(0) - keep["nb_b"].mean(0), int(hit.sum()))


class _W:
    """Weights/objective holder for a parity half, normalised to yields `frac` of the total."""
    def __init__(self, n_s, n_b, frac, objective="mcaware"):
        self.ws, self.wb = frac * P.S_TOT / n_s, frac * P.B_TOT / n_b
        self.objective, self.bfloor = objective, 1.0


def cross_selection(designs, smp, orc):
    """Returns concatenated two-channel BB templates, truth (s,b), true_inf, (kA, kB)."""
    ms = np.arange(len(smp.Xs)) % 2; mb = np.arange(len(smp.Xb)) % 2
    cache = []
    for dsg in designs:                                   # score on FULL sample, then split
        cache.append((dsg.bins(smp.Xs), dsg.bins(smp.Xb), dsg.nb))

    def half_counts(c, h):
        ls, lb, nb = cache[c]
        return (np.bincount(ls[ms == h], minlength=nb).astype(float),
                np.bincount(lb[mb == h], minlength=nb).astype(float),
                np.bincount(lb[mb == h], weights=smp.gb[mb == h], minlength=nb))

    W = {h: _W((ms == h).sum(), (mb == h).sum(), 0.5) for h in (0, 1)}
    score = {h: np.array([float(P.est_from_counts(*half_counts(c, h), W[h]))
                          for c in range(len(designs))]) for h in (0, 1)}
    k_for = {0: int(np.argmax(score[1])), 1: int(np.argmax(score[0]))}   # channel h uses k_for[h]
    S, B, D, NB, ST, BT = [], [], [], [], [], []
    for h in (0, 1):
        c = k_for[h]; ns, nbk, g = half_counts(c, h)
        S.append(ns * W[h].ws); B.append((nbk + 0.5) * W[h].wb); D.append(g * W[h].wb); NB.append(nbk)
        nso, nbo, _ = counts(designs[c], orc)
        ST.append(0.5 * nso * orc.ws); BT.append(np.maximum(0.5 * nbo * orc.wb, 1e-3 * orc.wb))
    s_o, b_o = np.concatenate(ST), np.concatenate(BT)
    d_o = []
    for h in (0, 1):
        _, _, go = counts(designs[k_for[h]], orc); d_o.append(0.5 * go * orc.wb)
    true_inf = float(P.ieff(s_o, b_o, np.concatenate(d_o)))
    tm = (np.concatenate(S), np.concatenate(B), np.concatenate(D), np.concatenate(NB))
    return tm, (s_o, b_o), true_inf, (k_for[0], k_for[1])


def one_rep(seed, n_mc, K, n_noise, alpha, oracle, n_toys, Fn, hold_frac, n_boot=200,
            skip_naive=False, liks=("bb", "bbp")):
    rng = np.random.default_rng(seed)
    n = n_mc // 2
    mk = lambda: (lambda Xs, Xb: Smp(Xs, Xb, P.nuis_raw(Xb) - oracle["gmean"], alpha))(
        P.sample_sig(rng, n, n_noise), P.sample_bkg(rng, n, n_noise))
    full, fresh = mk(), mk()
    nt = int(round(n * (1 - hold_frac)))
    T, S = full.part(slice(0, nt)), full.part(slice(nt, n))
    hps = draw_hparams(rng, K)

    if not skip_naive:
        D0 = [build_single(hp, full) for hp in hps]
        k0 = int(np.argmax([est(d, full) for d in D0]))
    D1 = [build_kfold(hp, full, Fn) for hp in hps]
    k1 = int(np.argmax([est(d, full) for d in D1]))          # OOF counts: scoring rule is OOF
    D2 = [build_kfold(hp, T, Fn) for hp in hps]
    k2 = int(np.argmax([est(d, S) for d in D2]))

    osm = oracle["smp"][alpha]
    orc = Smp(osm.Xs, osm.Xb, P.nuis_raw(osm.Xb) - oracle["gmean"], alpha)
    dns, dnb, n_hit = bootsel_correction(D1, full, k1, n_boot, rng)
    plans = [("R1_kfold", D1[k1], full, None), ("R2_holdout", D2[k2], T, None),
             ("R3_fresh", D1[k1], fresh, None), ("R4_bootsel", D1[k1], full, (dns, dnb)),
             ("R5_foldback", D1[k2], full, None)]
    if not skip_naive:
        plans = [("R0_naive", D0[k0], full, None)] + plans
    rows = []
    for name, dsg, tsmp, corr in plans:
        ns_o, nb_o, g_o = counts(dsg, orc)
        s_o, b_o, d_o = ns_o * orc.ws, np.maximum(nb_o * orc.wb, 1e-3 * orc.wb), g_o * orc.wb
        true_inf = float(P.ieff(s_o, b_o, d_o))
        tm = bb_templates(dsg, tsmp)
        if corr is not None:                        # subtract selection-induced count shift
            ns, nbk = counts(dsg, tsmp)[:2]
            ns_c = np.maximum(ns - corr[0], 0.0); nb_c = np.maximum(nbk - corr[1], 0.0)
            tm = (ns_c * tsmp.ws, (nb_c + 0.5) * tsmp.wb, tm[2], nb_c)
        nbk = tm[3]
        for lik in liks:
          trng = np.random.default_rng([seed, 99])   # common random numbers across methods & liks
          r1 = F3.run_toys(tm, (s_o, b_o), n_toys, lik, trng, 1.0)
          r0 = F3.run_toys(tm, (s_o, b_o), n_toys, lik, trng, 0.0)
          rows.append(dict(method=name, lik=lik, seed=seed, n_mc=n_mc, K=K, n_noise=n_noise, alpha=alpha,
                         true_inf=true_inf, n_bins=dsg.nb, med_bkg_bin=float(np.median(nbk)),
                         boot_hits=n_hit if name == "R4_bootsel" else -1,
                         **{f"{k}_mu1": v for k, v in r1.items()},
                         **{f"{k}_mu0": v for k, v in r0.items()}))
    # R6 cross-selection (two channels, joint fit)
    tm6, truth6, tinf6, (kA, kB) = cross_selection(D1, full, orc)
    for lik in liks:
      trng = np.random.default_rng([seed, 99])
      r1 = F3.run_toys(tm6, truth6, n_toys, lik, trng, 1.0)
      r0 = F3.run_toys(tm6, truth6, n_toys, lik, trng, 0.0)
      rows.append(dict(method="R6_crosssel", lik=lik, seed=seed, n_mc=n_mc, K=K, n_noise=n_noise, alpha=alpha,
                     true_inf=tinf6, n_bins=len(tm6[0]), med_bkg_bin=float(np.median(tm6[3])),
                     boot_hits=int(kA == kB),
                     **{f"{k}_mu1": v for k, v in r1.items()},
                     **{f"{k}_mu0": v for k, v in r0.items()}))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_mc", type=int, nargs="+", default=[2000, 10000])
    p.add_argument("--K", type=int, nargs="+", default=[1, 20, 100])
    p.add_argument("--n_noise", type=int, nargs="+", default=[0, 10])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.3])
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--n_toys", type=int, default=300)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--hold_frac", type=float, default=0.2)
    p.add_argument("--n_orc_s", type=int, default=100_000)
    p.add_argument("--n_orc_b", type=int, default=300_000)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--skip_naive", action="store_true", help="skip R0 (saves ~1/7 of training)")
    p.add_argument("--liks", nargs="+", default=["bb", "bbp"], choices=["bb", "bbp", "plugin"])
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default="results/phase4c")
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(vars(a), open(a.out + "_config.json", "w"), indent=2)

    t0 = time.time(); allrows = []
    for nn in a.n_noise:
        oracle = P.build_oracle(nn, a.alpha, a.n_orc_s, a.n_orc_b)
        for al in a.alpha:
            for N in a.n_mc:
                for K in a.K:
                    tasks = [delayed(one_rep)(1000 * r + 11, N, K, nn, al, oracle, a.n_toys,
                                              a.folds, a.hold_frac, a.n_boot, a.skip_naive,
                                              tuple(a.liks))
                             for r in range(a.reps)]
                    for rows in Parallel(n_jobs=a.jobs)(tasks):
                        allrows += rows
                    print(f"done n_noise={nn} alpha={al} N_MC={N} K={K}  [{time.time()-t0:.0f}s]",
                          flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(a.out + "_raw.csv", index=False)
    keys = ["n_noise", "alpha", "n_mc", "K", "lik", "method"]
    summ = df.groupby(keys).agg(
        fpr_z2=("fpr_z2_mu0", "mean"), fpr_z2_se=("fpr_z2_mu0", "sem"),
        cov68_mu1=("cov68_mu1", "mean"), bias_mu1=("mu_bias_mu1", "mean"),
        bias_mu1_se=("mu_bias_mu1", "sem"), sd_mu1=("mu_sd_mu1", "mean"),
        bias_mu0=("mu_bias_mu0", "mean"), true_inf=("true_inf", "mean"),
        n_bins=("n_bins", "mean"), med_bkg_bin=("med_bkg_bin", "mean"),
        boot_hits=("boot_hits", "mean")).reset_index()
    summ.to_csv(a.out + "_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print("(R6: n_bins = total over both channels; boot_hits = fraction of replicas with kA == kB)")
    print("\n=== Phase 4c summary (bb = BB-lite Gaussian, bbp = full Poisson BB; nominal fpr_z2 0.0228, cov68_mu1 0.683) ===")
    print(summ.round(3).to_string(index=False))
    import sklearn; print("versions: sklearn", sklearn.__version__, "numpy", np.__version__)
    print(f"Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
