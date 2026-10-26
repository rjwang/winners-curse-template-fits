#!/usr/bin/env python3
"""
Analysis of round2_run.py outputs (results/round2_N*_K*_raw.csv and *_q0.npz).

Tables (printed and written to <out>_*.csv):
  T1 matched   : for each workflow W, paired W - W_ind: mu-hat shift in units of sigma(W_ind)
                 with 95% CI and equivalence at +-0.1 sigma; FPR difference; mu=0 bias difference
  T2 absolute  : uncalibrated FPR, bias, RMSE, 68%/95% coverage, oracle information vs R1
                 (both oracles), all with standard errors across replicas
  T3 structure : decomposition of the CS offset: R1_ind2ch - R1_ind (two-channel structure),
                 CS_ind - R1_ind2ch (different/mutually selected designs), CS - CS_ind (reuse)
  T4 channels  : correlation of the per-channel Asimov mu shifts (Fisher-z 95% CI)
  T5 ablation  : exact signal (xs) and exact signal+shape (xsd) vs MC templates
  T6 calib     : thresholds calibrated on half of the replicas (even index), evaluated on the
                 other half: (a) full-pipeline calibration on W's own null; (b) calibration on the
                 matched independent-template null W_ind. FPR and power at mu = 0.25, 0.5.
  T6 is swap-averaged: thresholds from even replicas are applied to odd ones and vice versa.
  T7 power     : PAIRED differences of calibrated power between workflows (same pseudo-data
                 seeds within a replica), for both calibrations, against CS, R5 and R2h.
  T8 cond.     : spread of the per-analysis false-positive rate after calibration.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy import stats

ALPHA_Z2 = 0.02275
BASE = ["R1", "R2", "R2h", "R5", "CS", "CSs"]


def ci95(x):
    x = np.asarray(x, float); n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, se, m - t * se, m + t * se


def load(prefix):
    """Reads complete files (<prefix>_N*_K*_raw.csv) and chunk files
    (<prefix>_N*_K*_r####-####_raw.csv); chunks of one configuration are concatenated."""
    raws, parts = [], {}
    pat = re.compile(r"_N(\d+)_K(\d+)(?:_r\d+-\d+)?_raw\.csv$")
    for f in sorted(glob.glob(prefix + "_N*_K*_raw.csv")):
        mm = pat.search(f)
        if not mm:
            continue
        N, K = int(mm.group(1)), int(mm.group(2))
        raws.append(pd.read_csv(f))
        z = np.load(f.replace("_raw.csv", "_q0.npz"))
        parts.setdefault((N, K), []).append(z)
    qs = {}
    for key, zs in parts.items():
        meth, hyp = list(zs[0]["methods"]), list(zs[0]["hyp"])
        assert all(list(z["methods"]) == meth and list(z["hyp"]) == hyp for z in zs), key
        qs[key] = dict(q0=np.concatenate([z["q0"] for z in zs]), methods=meth, hyp=hyp,
                       seeds=np.concatenate([z["seeds"] for z in zs]))
    df = pd.concat(raws, ignore_index=True).drop_duplicates(subset=["n_mc", "K", "seed", "method"])
    return df, qs


def t1_matched(df):
    out = []
    for (N, K), g in df.groupby(["n_mc", "K"]):
        w = g.pivot_table(index="seed", columns="method",
                          values=["bias_mu1", "sd_mu1", "fpr", "bias_mu0"])
        for m in BASE:
            ref = m + "_ind"
            if (("bias_mu1", m) not in w) or (("bias_mu1", ref) not in w):
                continue
            sig = w[("sd_mu1", ref)].mean()
            d = (w[("bias_mu1", m)] - w[("bias_mu1", ref)]) / sig
            mm, se, lo, hi = ci95(d)
            f = ci95(100 * (w[("fpr", m)] - w[("fpr", ref)]))
            b0 = ci95(w[("bias_mu0", m)] - w[("bias_mu0", ref)])
            out.append(dict(N=N, K=K, W=m, reps=len(d), dmu_sig=mm, dmu_lo=lo, dmu_hi=hi,
                            equiv_0p1=bool(lo > -0.1 and hi < 0.1), halfwidth=(hi - lo) / 2,
                            reps_for_hw0p05=int(np.ceil(len(d) * ((hi - lo) / 2 / 0.05) ** 2)),
                            dfpr_pp=f[0], dfpr_se=f[1], dbias_mu0=b0[0], dbias_mu0_se=b0[1]))
    return pd.DataFrame(out)


def t2_absolute(df):
    out = []
    for (N, K), g in df.groupby(["n_mc", "K"]):
        w = g.pivot_table(index="seed", columns="method",
                          values=["true_inf"] + (["true_inf2"] if "true_inf2" in g else []))
        for m, gm in g.groupby("method"):
            r = dict(N=N, K=K, W=m, reps=len(gm))
            for v, scale in (("fpr", 100), ("bias_mu1", 1), ("cov68", 1), ("cov95", 1),
                             ("bias_mu0", 1)):
                a = ci95(scale * gm[v]); r[v], r[v + "_se"] = a[0], a[1]
            r["rmse_mu1"] = float(np.sqrt(np.mean(gm.rmse_mu1 ** 2)))
            for col in ("true_inf", "true_inf2"):
                if (col, m) in w and (col, "R1") in w:
                    a = ci95(100 * (w[(col, m)] / w[(col, "R1")] - 1))
                    r["dI_" + col + "_pct"], r["dI_" + col + "_se"] = a[0], a[1]
            out.append(r)
    return pd.DataFrame(out)


def t3_structure(df):
    pairs = [("R1_ind2ch", "R1_ind", "two-channel structure"),
             ("CS_ind", "R1_ind2ch", "different, mutually selected designs"),
             ("CS", "CS_ind", "template reuse in CS"),
             ("CSs_ind", "R1_ind2ch", "strict-split designs (indep. templates)"),
             ("CSs", "CSs_ind", "template reuse in CSs")]
    out = []
    for (N, K), g in df.groupby(["n_mc", "K"]):
        w = g.pivot_table(index="seed", columns="method", values="fpr")
        for a_, b_, lab in pairs:
            if a_ in w and b_ in w:
                m, se, _, _ = ci95(100 * (w[a_] - w[b_]))
                out.append(dict(N=N, K=K, comparison=f"{a_} - {b_}", meaning=lab,
                                dfpr_pp=m, se=se))
    return pd.DataFrame(out)


def t4_channels(df):
    out = []
    for (N, K, m), g in df.dropna(subset=["dmu_ch0"]).groupby(["n_mc", "K", "method"]):
        r = np.corrcoef(g.dmu_ch0, g.dmu_ch1)[0, 1]; n = len(g)
        z, s = np.arctanh(r), 1 / np.sqrt(max(n - 3, 1))
        out.append(dict(N=N, K=K, W=m, reps=n, corr=r, lo=np.tanh(z - 1.96 * s),
                        hi=np.tanh(z + 1.96 * s)))
    return pd.DataFrame(out)


def t5_ablation(df):
    out = []
    for (N, K), g in df.groupby(["n_mc", "K"]):
        w = g.pivot_table(index="seed", columns="method", values=["bias_mu1", "sd_mu1", "fpr"])
        if ("bias_mu1", "R1_xs") not in w:
            continue
        sig = w[("sd_mu1", "R1_ind")].mean()
        for tag in ("", "_xs", "_xsd"):
            d = ci95((w[("bias_mu1", "R1" + tag)] - w[("bias_mu1", "R1_ind" + tag)]) / sig)
            f = ci95(100 * (w[("fpr", "R1" + tag)] - w[("fpr", "R1_ind" + tag)]))
            out.append(dict(N=N, K=K, templates={"": "MC", "_xs": "exact signal",
                                                 "_xsd": "exact signal+shape"}[tag],
                            dmu_R1_vs_ind_sig=d[0], lo=d[2], hi=d[3], dfpr_pp=f[0], dfpr_se=f[1],
                            fpr_R1=100 * w[("fpr", "R1" + tag)].mean(),
                            fpr_R1_ind=100 * w[("fpr", "R1_ind" + tag)].mean()))
    return pd.DataFrame(out)


def _calibrated_rates(z, schemes=("own", "matched")):
    """Swap-averaged calibration: replicas are split by index parity; thresholds from one half
    are applied to the other and vice versa, so every replica is evaluated exactly once.
    'own'     : threshold from W's own background-only q0 distribution (full-pipeline
                calibration; requires re-running the whole workflow on independent simulation)
    'matched' : threshold from W_ind (same frozen design, independent templates), i.e. a
                calibration that accounts for template statistics but not for selection."""
    q0, meth, hyp = z["q0"], z["methods"], z["hyp"]
    idx = np.arange(q0.shape[0])
    splits = [(idx % 2 == 0, idx % 2 == 1), (idx % 2 == 1, idx % 2 == 0)]
    hs = [h for h in ("mu0", "mu025", "mu05") if h in hyp]
    out = {}
    for sch in schemes:
        for m in BASE:
            if m not in meth or (m + "_ind") not in meth:
                continue
            src = meth.index(m if sch == "own" else m + "_ind"); i = meth.index(m)
            rates = {h: np.zeros(len(idx)) for h in hs}; zthr = []
            for cal, ev in splits:
                c = np.quantile(q0[cal, src, hyp.index("mu0")].ravel(), 1 - ALPHA_Z2)
                zthr.append(float(np.sqrt(c)))
                for h in hs:
                    rates[h][ev] = (q0[ev, i, hyp.index(h)] > c).mean(1)
            out[(sch, m)] = dict(rates=rates, zthr=zthr)
    return out


def t6_calibration(qs):
    out = []
    for (N, K), z in sorted(qs.items()):
        cr = _calibrated_rates(z)
        for (sch, m), v in cr.items():
            r = dict(N=N, K=K, W=m, calibration=sch, Z_thr_a=v["zthr"][0], Z_thr_b=v["zthr"][1],
                     reps=len(v["rates"]["mu0"]))
            for h, lab in (("mu0", "fpr"), ("mu025", "power_mu025"), ("mu05", "power_mu05")):
                if h in v["rates"]:
                    a = ci95(100 * v["rates"][h]); r[lab], r[lab + "_se"] = a[0], a[1]
            out.append(r)
        # asymptotic threshold, for reference
        q0, meth, hyp = z["q0"], z["methods"], z["hyp"]
        for m in BASE:
            if m in meth:
                a = ci95(100 * (q0[:, meth.index(m), hyp.index("mu0")] > 4.0).mean(1))
                out.append(dict(N=N, K=K, W=m, calibration="asymptotic", Z_thr_a=2.0, Z_thr_b=2.0,
                                reps=q0.shape[0], fpr=a[0], fpr_se=a[1]))
    return pd.DataFrame(out)


def t7_power_paired(qs, refs=("CS", "R5", "R2h")):
    out = []
    for (N, K), z in sorted(qs.items()):
        cr = _calibrated_rates(z)
        for sch in ("own", "matched"):
            for h in ("mu025", "mu05"):
                for ref in refs:
                    if (sch, ref) not in cr or h not in cr[(sch, ref)]["rates"]:
                        continue
                    pr = cr[(sch, ref)]["rates"][h]
                    for m in BASE:
                        if m == ref or (sch, m) not in cr:
                            continue
                        pw = cr[(sch, m)]["rates"][h]
                        d = ci95(100 * (pw - pr))
                        out.append(dict(N=N, K=K, calibration=sch, mu=h, W=m, ref=ref,
                                        power_W=100 * pw.mean(), power_ref=100 * pr.mean(),
                                        dpower_pp=d[0], se=d[1], lo=d[2], hi=d[3]))
    return pd.DataFrame(out)


def t8_conditional(qs):
    """Spread of the per-analysis (conditional) false-positive rate after calibration."""
    out = []
    for (N, K), z in sorted(qs.items()):
        for (sch, m), v in _calibrated_rates(z).items():
            r = v["rates"]["mu0"]
            out.append(dict(N=N, K=K, W=m, calibration=sch, mean=100 * r.mean(),
                            median=100 * np.median(r), p90=100 * np.quantile(r, 0.9),
                            frac_gt5=float(np.mean(r > 0.05)), frac_gt10=float(np.mean(r > 0.10))))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results/round2")
    p.add_argument("--out", default="results/round2_analysis")
    a = p.parse_args()
    df, qs = load(a.prefix)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 40)
    tabs = dict(T1_matched=t1_matched(df), T2_absolute=t2_absolute(df),
                T3_structure=t3_structure(df), T4_channels=t4_channels(df),
                T5_ablation=t5_ablation(df), T6_calibration=t6_calibration(qs),
                T7_power_paired=t7_power_paired(qs), T8_conditional=t8_conditional(qs))
    for name, t in tabs.items():
        t.to_csv(f"{a.out}_{name}.csv", index=False)
        print(f"\n=== {name} ===")
        print(t.round(3).to_string(index=False) if len(t) else "(empty)")


if __name__ == "__main__":
    main()
