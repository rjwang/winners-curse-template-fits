#!/usr/bin/env python3
"""
Topic A - Round 2 (response to review): one training per replica, many workflows.

Workflows (all share the candidate set of the replica):
  R1        k-fold selection on all MC, templates = same OOF counts
  R2        20% hold-out for selection only; train/template on 80%
  R2h       50% hold-out (same selection-sample size as CS)
  R5        select on 20% hold-out, retrain chosen config on all MC, OOF templates from all
  CS        cross-selection with honest edges (= R6h of phase 4d): channel h uses the design
            selected (and binned) on MC half 1-h, templates from MC half h; classifiers trained
            (k-fold) on both halves
  CSs       STRICT outer split: for channel h, k-fold training, selection and binning use only
            MC half 1-h; half h is only scored and templated
Matched controls (review E1): for each W, W_ind freezes the models, edges and channel structure
  of W and fills its templates from an independent MC sample of the same budget.
  R1_ind    (= R3 of the paper)
  R1_ind2ch R1 design, independent templates, split into two channels like CS
            (isolates the effect of the two-channel structure)
Signal / shape ablation (review E4), on R1 and R1_ind:
  _xs       exact (oracle) signal template
  _xsd      exact signal template and exact shape response
Per workflow and hypothesis mu in (1, 0, 0.25, 0.5): q0 of every pseudo-experiment is stored
(npz) for calibration and power studies; mu=1 gives bias, RMSE, 68%/95% coverage.
Two-channel workflows also record the Asimov mu-shift of each channel separately (review E3).
A second oracle with an independent seed re-evaluates the oracle information (review E9).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_A_v3 as P
import phase3_fit as F3
import phase4d_honest as H

HYP_ALL = {"mu1": 1.0, "mu0": 0.0, "mu025": 0.25, "mu05": 0.5}
HYP = tuple(HYP_ALL.items())                 # may be restricted by --hyps (mu1, mu0 required)
Q_Z2 = 4.0                                   # q0 threshold for Z > 2 (asymptotic)


# --------------------------------------------------------------------------- helpers
_TRUTH_CACHE = {}


def truth_of(design, orc, frac):
    """Oracle expected yields of a design, scaled to `frac`; oracle counts cached per design."""
    key = (id(design), id(orc))
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key] = (design, H.counts(design, orc))   # keep design alive -> stable id
    nso, nbo, go = _TRUTH_CACHE[key][1]
    return (frac * nso * orc.ws, np.maximum(frac * nbo * orc.wb, 1e-3 * orc.wb * frac),
            frac * go * orc.wb)


def chan(design, smp, frac, orcs, mask_s=None, mask_b=None):
    """One channel. Bins are computed on the array passed (k-fold position!) and only then
    masked. Templates normalised to `frac` of the expected yields."""
    ls, lb, gb = design.bins(smp.Xs), design.bins(smp.Xb), smp.gb
    if mask_s is not None:
        ls, lb, gb = ls[mask_s], lb[mask_b], gb[mask_b]
    nb = design.nb
    ns = np.bincount(ls, minlength=nb).astype(float)
    nbk = np.bincount(lb, minlength=nb).astype(float)
    g = np.bincount(lb, weights=gb, minlength=nb)
    ws, wb = frac * P.S_TOT / len(ls), frac * P.B_TOT / len(lb)
    return dict(tm=[ns * ws, (nbk + 0.5) * wb, g * wb, nbk], design=design, frac=frac,
                truths=[truth_of(design, o, frac) for o in orcs])


def concat(chs, k):
    return np.concatenate([c["tm"][k] for c in chs])


def concat_truth(chs, j=0):
    return tuple(np.concatenate([c["truths"][j][k] for c in chs]) for k in range(3))


def asimov_dmu(c):
    s, b, d, nbk = c["tm"]
    s_o, b_o, _ = c["truths"][0]
    F = F3.Fitter(s, b, d, nbk + 0.5, (s_o + b_o)[None, :], np.zeros(1), "bbp")
    mu, _, _ = F.fit_multistart()
    return float(mu[0] - 1.0)


def fit_block(chs, seed, n_toys):
    s, b, d, nbk = (concat(chs, k) for k in range(4))
    s_o, b_o, _ = concat_truth(chs)
    out, q0s = {}, np.zeros((len(HYP), n_toys), np.float32)
    for j, (name, mu) in enumerate(HYP):
        trng = np.random.default_rng([seed, 99, j])       # common random numbers
        n = trng.poisson(mu * s_o + b_o, size=(n_toys, len(b_o))).astype(float)
        nu0 = trng.normal(0, 1, n_toys)
        F = F3.Fitter(s, b, d, nbk + 0.5, n, nu0, "bbp")
        muh, _, fh = F.fit_multistart()
        if mu == 1.0:
            _, _, ft = F.fit(fix_mu=1.0)
            q = 2 * (ft - fh)
            out.update(bias_mu1=float(np.mean(muh - 1)), sd_mu1=float(np.std(muh)),
                       rmse_mu1=float(np.sqrt(np.mean((muh - 1) ** 2))),
                       cov68=float(np.mean(q < 1.0)), cov95=float(np.mean(q < 3.8415)))
        _, _, f0 = F.fit(fix_mu=0.0)
        q0 = np.maximum(2 * (f0 - fh), 0.0) * (muh > 0)
        q0s[j] = q0
        if mu == 0.0:
            out["fpr"] = float(np.mean(q0 > Q_Z2))
            out["bias_mu0"] = float(np.mean(muh))
    return out, q0s


def ablate(chs, mode):
    """mode 'xs': exact signal template; 'xsd': exact signal and exact shape response."""
    new = []
    for c in chs:
        s, b, d, nbk = c["tm"]
        s_o, b_o, d_o = c["truths"][0]
        wb = b / (nbk + 0.5)
        d_new = (d_o / b_o) * nbk * wb if mode == "xsd" else d
        new.append(dict(c, tm=[s_o.copy(), b, d_new, nbk]))
    return new


# --------------------------------------------------------------------------- one replica
def one_rep(seed, n_mc, K, n_noise, alpha, oracles, n_toys, Fn, do_ablation):
    _TRUTH_CACHE.clear()
    rng = np.random.default_rng(seed)
    n = n_mc // 2
    gmean = oracles[0]["gmean"]
    mk = lambda: (lambda Xs, Xb: H.Smp(Xs, Xb, P.nuis_raw(Xb) - gmean, alpha))(
        P.sample_sig(rng, n, n_noise), P.sample_bkg(rng, n, n_noise))
    full, fresh = mk(), mk()
    hps = H.draw_hparams(rng, K)
    orcs = []
    for o in oracles:
        osm = o["smp"][alpha]
        orcs.append(H.Smp(osm.Xs, osm.Xb, P.nuis_raw(osm.Xb) - o["gmean"], alpha))
    ms = np.arange(n) % 2                      # parity halves (same for sig and bkg: n each)
    W = {}

    # ---- R1 and its matched controls
    D1 = [H.build_kfold(hp, full, Fn) for hp in hps]
    k1 = int(np.argmax([H.est(d, full) for d in D1]))
    W["R1"] = [chan(D1[k1], full, 1.0, orcs)]
    W["R1_ind"] = [chan(D1[k1], fresh, 1.0, orcs)]
    W["R1_ind2ch"] = [chan(D1[k1], fresh, 0.5, orcs, ms == h, ms == h) for h in (0, 1)]

    # ---- hold-out 20% (R2, R5) and 50% (R2h)
    for name, frac in (("R2", 0.2), ("R2h", 0.5)):
        nt = int(round(n * (1 - frac)))
        T, S = full.part(slice(0, nt)), full.part(slice(nt, n))
        D2 = [H.build_kfold(hp, T, Fn) for hp in hps]
        k2 = int(np.argmax([H.est(d, S) for d in D2]))
        W[name] = [chan(D2[k2], T, 1.0, orcs)]
        W[name + "_ind"] = [chan(D2[k2], fresh.part(slice(0, nt)), 1.0, orcs)]
        if name == "R2":
            W["R5"] = [chan(D1[k2], full, 1.0, orcs)]
            W["R5_ind"] = [chan(D1[k2], fresh, 1.0, orcs)]

    # ---- CS: cross-selection with honest edges (models trained on both halves)
    sc = [(d.score(full.Xs), d.score(full.Xb)) for d in D1]
    Wh = {g: H._W((ms == g).sum(), (ms == g).sum(), 0.5) for g in (0, 1)}
    cs_des = {}
    for h in (0, 1):
        g = 1 - h
        best, bestI = None, -np.inf
        for c, hp in enumerate(hps):
            e = H.edges_from(sc[c][0][ms == g], hp["nbins"]); nb = len(e) + 1
            ls, lb = np.digitize(sc[c][0][ms == g], e), np.digitize(sc[c][1][ms == g], e)
            I = float(P.est_from_counts(np.bincount(ls, minlength=nb).astype(float),
                                        np.bincount(lb, minlength=nb).astype(float),
                                        np.bincount(lb, weights=full.gb[ms == g], minlength=nb),
                                        Wh[g]))
            if I > bestI:
                best, bestI = H.Design(D1[c].models, e), I
        cs_des[h] = best
    W["CS"] = [chan(cs_des[h], full, 0.5, orcs, ms == h, ms == h) for h in (0, 1)]
    W["CS_ind"] = [chan(cs_des[h], fresh, 0.5, orcs, ms == h, ms == h) for h in (0, 1)]

    # ---- CSs: strict outer split (training, selection, binning only on half 1-h)
    halves = {g: full.part(ms == g) for g in (0, 1)}
    fhalves = {g: fresh.part(ms == g) for g in (0, 1)}
    css = {}
    for g in (0, 1):
        Dg = [H.build_kfold(hp, halves[g], Fn) for hp in hps]
        css[1 - g] = Dg[int(np.argmax([H.est(d, halves[g]) for d in Dg]))]
    W["CSs"] = [chan(css[h], halves[h], 0.5, orcs) for h in (0, 1)]
    W["CSs_ind"] = [chan(css[h], fhalves[h], 0.5, orcs) for h in (0, 1)]

    # ---- signal / shape ablation
    if do_ablation:
        for base in ("R1", "R1_ind"):
            W[base + "_xs"] = ablate(W[base], "xs")
            W[base + "_xsd"] = ablate(W[base], "xsd")

    rows, q0 = [], {}
    for name, chs in W.items():
        res, q0[name] = fit_block(chs, seed, n_toys)
        s1, b1, d1 = concat_truth(chs, 0)
        r = dict(method=name, seed=seed, n_mc=n_mc, K=K, n_noise=n_noise, alpha=alpha,
                 n_ch=len(chs), true_inf=float(P.ieff(s1, b1, d1)),
                 n_bins=int(sum(len(c["tm"][0]) for c in chs)),
                 med_bkg_bin=float(np.median(concat(chs, 3))), **res)
        if len(orcs) > 1:
            s2, b2, d2 = concat_truth(chs, 1)
            r["true_inf2"] = float(P.ieff(s2, b2, d2))
        if len(chs) == 2 and not name.endswith(("_xs", "_xsd")):
            r["dmu_ch0"], r["dmu_ch1"] = asimov_dmu(chs[0]), asimov_dmu(chs[1])
        rows.append(r)
    return rows, q0


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="2000:1:100,2000:100:200,10000:1:50,10000:100:100",
                   help="comma-separated N_MC:K:replicas")
    p.add_argument("--n_noise", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--n_toys", type=int, default=300)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed_offset", type=int, default=17)
    p.add_argument("--no_ablation", action="store_true")
    p.add_argument("--no_oracle2", action="store_true")
    p.add_argument("--n_orc_s", type=int, default=100_000)
    p.add_argument("--n_orc_b", type=int, default=300_000)
    p.add_argument("--chunk", type=int, default=25,
                   help="replicas per saved chunk; finished chunks are skipped on rerun")
    p.add_argument("--hyps", default="mu1,mu0,mu025,mu05",
                   help="pseudo-experiment hypotheses (mu1 and mu0 are required)")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--out", default="results/round2")
    a = p.parse_args()
    global HYP
    hs = a.hyps.split(",")
    assert "mu1" in hs and "mu0" in hs, "--hyps must contain mu1 and mu0"
    HYP = tuple((h, HYP_ALL[h]) for h in hs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(vars(a), open(a.out + "_config.json", "w"), indent=2)

    oracles = [P.build_oracle(a.n_noise, [a.alpha], a.n_orc_s, a.n_orc_b, seed=12345)]
    if not a.no_oracle2:
        oracles.append(P.build_oracle(a.n_noise, [a.alpha], a.n_orc_s, a.n_orc_b, seed=54321))
    t0 = time.time()
    for cfg in a.configs.split(","):
        N, K, reps = (int(x) for x in cfg.split(":"))
        base = f"{a.out}_N{N}_K{K}"
        if os.path.exists(base + "_raw.csv"):                  # unchunked result from before
            print(f"skip {base} (complete file exists)", flush=True)
            continue
        starts = list(range(0, reps, a.chunk))
        t_cfg = time.time(); done_here = 0
        for ci, r0 in enumerate(starts):
            r1 = min(r0 + a.chunk, reps)
            tag = f"{base}_r{r0:04d}-{r1:04d}"
            if os.path.exists(tag + "_raw.csv"):
                print(f"skip {tag} (exists)", flush=True)
                continue
            tasks = [delayed(one_rep)(1000 * r + a.seed_offset, N, K, a.n_noise, a.alpha,
                                      oracles, a.n_toys, a.folds, not a.no_ablation)
                     for r in range(r0, r1)]
            res = Parallel(n_jobs=a.jobs, verbose=5)(tasks)
            rows = [x for rr, _ in res for x in rr]
            methods = list(res[0][1].keys())
            q0 = np.stack([np.stack([qq[m] for m in methods]) for _, qq in res])
            pd.DataFrame(rows).to_csv(tag + "_raw.csv", index=False)
            np.savez_compressed(tag + "_q0.npz", q0=q0, methods=np.array(methods),
                                seeds=np.array([1000 * r + a.seed_offset for r in range(r0, r1)]),
                                hyp=np.array([h for h, _ in HYP]))
            done_here += r1 - r0
            el = time.time() - t_cfg
            left = sum(min(x + a.chunk, reps) - x for x in starts[ci + 1:])
            print(f"[N={N} K={K}] replicas {r0}-{r1} saved | {el/60:.1f} min for {done_here} "
                  f"replicas | ETA this config ~{el/done_here*left/60:.0f} min "
                  f"| total {(time.time()-t0)/60:.1f} min", flush=True)
    import sklearn; print("versions: sklearn", sklearn.__version__, "numpy", np.__version__)
    print(f"Total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
