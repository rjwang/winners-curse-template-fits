#!/usr/bin/env python3
"""Validate phase3 Fitter (free fits with mu >= 0): (1) Newton == Nelder-Mead on realistic finite-MC templates,
(2) calibration (bias 0, cov68 ~0.683, fpr ~0.023) with perfect oracle templates."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pilot_A_v3 as P, phase3_fit as F
from scipy.optimize import minimize

o = P.build_oracle(10, [0.3], 100000, 300000); orc = o["smp"][0.3]; t = o["ref_tree"]
so, bo, do = P.templates(t, orc); _, nbo, _ = P.counts(t, orc)
print("[1] Newton vs Nelder-Mead (max NLL excess of Newton; should be <~1e-3)")
for seed in (7, 8):
    rng = np.random.default_rng(seed); n = 1000
    Xs, Xb = P.sample_sig(rng, n, 10), P.sample_bkg(rng, n, 10)
    full = P.Sample(Xs, Xb, P.nuis_raw(Xb) - o["gmean"], 0.3)
    for lik in ("bb", "bbp", "plugin"):
        s, b, d, nb = F.tmpl_from(t, full, "bb" if lik == "bbp" else lik, 1.0)
        for mut in (0.0, 1.0):
            T = 40; nd = rng.poisson(mut * so + bo, size=(T, len(bo))).astype(float)
            nu0 = rng.normal(0, 1, T)
            Fi = F.Fitter(s, b, d, nb + 0.5, nd, nu0, lik); mu, nu, f = Fi.fit_multistart()
            _, _, fc = Fi.fit(fix_mu=mut)
            wf = wc = 0.0
            for i in range(T):
                G = F.Fitter(s, b, d, nb + 0.5, nd[i:i+1], nu0[i:i+1], lik)
                fn = lambda x: float(G.nll(np.array([x[0]]), np.array([x[1]]))[0])
                best = min((minimize(fn, x0, method="Nelder-Mead", bounds=[(0, None), (None, None)],
                                     options=dict(xatol=1e-8, fatol=1e-10, maxiter=8000))
                            for x0 in ([1, 0], [0, 0], [0.5, 1], [0.5, -1], [0.01, 0.5])),
                           key=lambda r: r.fun)
                wf = max(wf, f[i] - best.fun)
                f1 = lambda x: float(G.nll(np.array([mut]), np.array([x[0]]))[0])
                bc = min((minimize(f1, [x0], method="Nelder-Mead",
                                   options=dict(xatol=1e-8, fatol=1e-10)) for x0 in (0, 1, -1)),
                         key=lambda r: r.fun)
                wc = max(wc, fc[i] - bc.fun)
            print(f"   seed={seed} {lik:6s} mu_true={mut}: free {wf:9.5f}   fixed-mu {wc:9.5f}")

print("[2] calibration with perfect templates (4000 toys)")
rng = np.random.default_rng(1)
for lik in ("plugin", "bb", "bbp"):
    for mut in (1.0, 0.0):
        r = F.run_toys((so, bo, do, nbo), (so, bo), 4000, lik, rng, mut)
        print("  ", lik, mut, {k: round(v, 4) for k, v in r.items()})
