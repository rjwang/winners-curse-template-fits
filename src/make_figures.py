#!/usr/bin/env python3
"""
Figures for the paper (IOP / MLST). All numbers are computed from the result CSVs.

  fig1_schematic.pdf   standard k-fold selection vs cross-selection (data flow)
  fig2_fpr_vs_K.pdf    background-only rate of Z>2 vs number of candidates K
  fig3_fpr_ecdf.pdf    per-replica distribution of the Z>2 rate (N_MC=2000, K=100)
  fig4_tradeoff.pdf    calibration (paired shift / sigma) vs cost (Fisher information loss)
  fig5_theory.pdf      (a) sigma, tau vs N_MC; (b) predicted vs measured paired shift

Accessibility (IOP guidance): Okabe-Ito colours AND distinct markers/line styles,
so no information is carried by colour alone. Vector PDF output (+ PNG previews).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "mathtext.fontset": "cm",
})
FULL, HALF = 6.3, 3.1          # inches (IOP single-column text width ~16 cm)
NOMINAL = 0.02275
# Okabe-Ito
C = dict(R1="#D55E00", R2="#56B4E9", R3="#000000", R4="#E69F00", R5="#0072B2",
         R6="#999999", R6h="#009E73")
STY = {  # method -> (label, colour, marker, linestyle)
    "R1_kfold":     ("R1  $k$-fold selection",    C["R1"],  "o", "-"),
    "R3_fresh":     ("R3  independent templates", C["R3"],  "s", "--"),
    "R2_holdout":   ("R2  hold-out",              C["R2"],  "v", ":"),
    "R4_bootsel":   ("R4  bootstrap correction",  C["R4"],  "D", "-."),
    "R5_foldback":  ("R5  fold-back",             C["R5"],  "^", ":"),
    "R6_crosssel":  ("R6  cross-sel. (shared edges)", C["R6"], "P", "-."),
    "R6h_crosssel": ("R6h cross-selection",       C["R6h"], "*", "-"),
}


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def paired(raw, var, m, ref="R3_fresh"):
    w = raw.pivot_table(index=["n_mc", "K", "seed"], columns="method", values=var)
    d = (w[m] - w[ref]).groupby(level=[0, 1])
    return d.mean(), d.sem()


# ----------------------------------------------------------------------------- fig 1
def fig_schematic(out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw=dict(wspace=0.08))

    def box(ax, xy, w, h, text, fc, fs=6.8):
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec="k", lw=0.8))
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
                linespacing=1.15)

    def arrow(ax, p, q, col="k", ls="-", lw=0.9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                     color=col, ls=ls, shrinkA=0, shrinkB=0))

    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    # (a) standard k-fold selection
    ax = axes[0]
    ax.set_title("(a) $k$-fold selection (standard practice)", loc="left")
    box(ax, (0.0, 2.3), 2.5, 1.6, "simulation\n(out-of-fold\nscores)", "#F2F2F2")
    box(ax, (3.3, 4.2), 3.1, 1.3, "select best of\n$K$ designs", "#FBE3D6")
    box(ax, (3.3, 0.6), 3.1, 1.3, "fill templates", "#FBE3D6")
    box(ax, (7.4, 2.3), 2.5, 1.6, "fit data\n(all events)", "#FFFFFF")
    arrow(ax, (2.5, 3.5), (3.3, 4.8)); arrow(ax, (2.5, 2.7), (3.3, 1.3))
    arrow(ax, (6.4, 4.8), (7.4, 3.5)); arrow(ax, (6.4, 1.3), (7.4, 2.7))
    arrow(ax, (3.9, 4.2), (3.9, 1.9), col=C["R1"], ls="--", lw=1.1)
    ax.text(4.1, 3.05, "same\nevents", fontsize=6.3, color=C["R1"], va="center", ha="left",
            linespacing=1.1)

    # (b) cross-selection
    ax = axes[1]
    ax.set_title("(b) cross-selection (this work)", loc="left")
    box(ax, (0.0, 4.1), 2.3, 1.4, "simul.\nhalf 1", "#DDEFE8")
    box(ax, (0.0, 0.7), 2.3, 1.4, "simul.\nhalf 0", "#E3EEF7")
    box(ax, (2.8, 4.1), 2.4, 1.4, "select $\\phi_A$\n(edges:\nhalf 1)", "#DDEFE8")
    box(ax, (2.8, 0.7), 2.4, 1.4, "select $\\phi_B$\n(edges:\nhalf 0)", "#E3EEF7")
    box(ax, (5.8, 4.1), 1.7, 1.4, "data\nhalf 0", "#FFFFFF")
    box(ax, (5.8, 0.7), 1.7, 1.4, "data\nhalf 1", "#FFFFFF")
    box(ax, (8.1, 2.3), 1.9, 1.6, "joint fit\n$(\\mu,\\nu)$", "#F2F2F2")
    arrow(ax, (2.3, 4.8), (2.8, 4.8)); arrow(ax, (2.3, 1.4), (2.8, 1.4))
    arrow(ax, (5.2, 4.8), (5.8, 4.8)); arrow(ax, (5.2, 1.4), (5.8, 1.4))
    # templates cross over: simulation half h fills the templates of data half h
    arrow(ax, (1.15, 2.1), (5.85, 4.15), col=C["R5"], ls="--", lw=1.1)
    arrow(ax, (1.15, 4.1), (5.85, 2.05), col=C["R6h"], ls="--", lw=1.1)
    ax.text(3.9, 3.1, "templates", fontsize=6.3, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.6))
    arrow(ax, (7.5, 4.8), (8.1, 3.7)); arrow(ax, (7.5, 1.4), (8.1, 2.5))
    save(fig, out, "fig1_schematic")


# ----------------------------------------------------------------------------- fig 2
def fig_fpr_vs_K(raw, out):
    methods = ["R1_kfold", "R4_bootsel", "R3_fresh", "R5_foldback", "R6h_crosssel"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    for ax, N in zip(axes, (2000, 10000)):
        for i, m in enumerate(methods):
            g = raw[(raw.n_mc == N) & (raw.method == m)].groupby("K").fpr_z2_mu0
            mu, se = g.mean(), g.sem()
            lab, col, mk, ls = STY[m]
            x = mu.index.values * (1 + 0.06 * (i - 2))          # small horizontal offsets
            ax.errorbar(x, 100 * mu.values, yerr=100 * se.values, color=col, marker=mk, ls=ls,
                        ms=5 if mk != "*" else 8, lw=1.1, capsize=2, label=lab)
        ax.axhline(100 * NOMINAL, color="0.5", lw=0.8, ls=(0, (1, 2)))
        ax.set_xscale("log"); ax.set_xticks([1, 20, 100]); ax.set_xticklabels(["1", "20", "100"])
        ax.set_xlabel("number of candidate designs $K$")
        ax.set_title(rf"$N_{{\rm MC}} = {N:,}$".replace(",", r"\,"), loc="left")
        ax.text(3.2, 100 * NOMINAL - 1.3, "nominal 2.28%", fontsize=6.5, color="0.4")
    axes[0].set_ylabel(r"background-only rate of $Z>2$  [%]")
    axes[0].set_ylim(0, 25)
    axes[1].legend(loc="upper left", frameon=False)
    save(fig, out, "fig2_fpr_vs_K")


# ----------------------------------------------------------------------------- fig 3
def fig_fpr_ecdf(raw, out, N=2000, K=100):
    methods = ["R1_kfold", "R3_fresh", "R5_foldback", "R6h_crosssel"]
    fig, ax = plt.subplots(figsize=(HALF + 0.3, 2.5))
    for m in methods:
        v = np.sort(100 * raw[(raw.n_mc == N) & (raw.K == K) & (raw.method == m)].fpr_z2_mu0.values)
        y = np.arange(1, len(v) + 1) / len(v)
        lab, col, mk, ls = STY[m]
        ax.step(v, 1 - y + 1 / len(v), where="post", color=col, ls=ls, lw=1.3, label=lab)
        ax.plot(v[::5], (1 - y + 1 / len(v))[::5], ls="none", marker=mk, color=col,
                ms=4 if mk != "*" else 7)
    ax.axvline(10, color="0.5", lw=0.8, ls=(0, (1, 2)))
    ax.set_xlabel(r"rate of $Z>2$ in a single analysis  [%]")
    ax.set_ylabel("fraction of analyses above")
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02)
    ax.set_title(rf"$N_{{\rm MC}}=2000$, $K={K}$, 50 replicas", loc="left")
    hd = [Line2D([], [], color=STY[m][1], ls=STY[m][3], marker=STY[m][2], lw=1.3,
                 ms=4 if STY[m][2] != "*" else 7, label=STY[m][0]) for m in methods]
    ax.legend(handles=hd, loc="upper right", frameon=False)
    save(fig, out, "fig3_fpr_ecdf")


# ----------------------------------------------------------------------------- fig 4
def fig_tradeoff(raw, out, K=100):
    methods = ["R1_kfold", "R4_bootsel", "R2_holdout", "R5_foldback", "R6_crosssel", "R6h_crosssel"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7), sharey=False,
                             gridspec_kw=dict(wspace=0.28))
    for ax, N in zip(axes, (2000, 10000)):
        sub = raw[(raw.n_mc == N) & (raw.K == K)]
        sig = sub[sub.method == "R3_fresh"].mu_sd_mu1.mean()
        w = sub.pivot_table(index="seed", columns="method", values=["mu_bias_mu1", "true_inf"])
        for m in methods:
            d = (w[("mu_bias_mu1", m)] - w[("mu_bias_mu1", "R3_fresh")]) / sig
            c = 100 * (w[("true_inf", m)] / w[("true_inf", "R1_kfold")] - 1)
            lab, col, mk, ls = STY[m]
            ax.errorbar(c.mean(), d.mean(), xerr=c.sem(), yerr=d.sem(), color=col, marker=mk,
                        ms=6 if mk != "*" else 10, capsize=2, lw=1, ls="none", label=lab)
        ax.axhline(0, color="0.5", lw=0.8, ls=(0, (1, 2)))
        ax.axvline(0, color="0.5", lw=0.8, ls=(0, (1, 2)))
        ax.set_xlabel(r"true Fisher information vs R1  [%]")
        ax.set_title(rf"$N_{{\rm MC}} = {N:,}$, $K={K}$".replace(",", r"\,", 1), loc="left")
        ax.annotate("ideal", xy=(0, 0), xytext=(-1.2 if N == 2000 else -0.8, -0.17 if N == 2000 else -0.08),
                    fontsize=7, color="0.35", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
    axes[0].set_ylabel(r"paired shift of $\hat\mu$ vs R3  [$\sigma_{\hat\mu}$]")
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.8)
    save(fig, out, "fig4_tradeoff")


# ----------------------------------------------------------------------------- fig 5
def fig_theory(th, tha, out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw=dict(wspace=0.36))
    ax = axes[0]
    t = th[th.K == 100].sort_values("n_mc")
    N = t.n_mc.values
    ax.plot(N, t.sigma, color=C["R1"], marker="o", ls="-", label=r"$\sigma$  estimation noise")
    ax.plot(N, t.tau, color=C["R5"], marker="s", ls="--", label=r"$\tau$  spread of true quality")
    NN = np.geomspace(1800, 11000, 50)
    ax.plot(NN, t.sigma.values[0] * np.sqrt(N[0] / NN), color=C["R1"], lw=0.8, ls=":",
            label=r"$\propto N_{\rm MC}^{-1/2}$")
    ax.set_xscale("log"); ax.set_xticks([2000, 5000, 10000]); ax.set_xticklabels(["2000", "5000", "10000"])
    ax.minorticks_off()
    ax.set_xlabel(r"simulated events $N_{\rm MC}$"); ax.set_ylabel("spread among candidates")
    ax.set_title("(a) noise vs genuine differences, $K=100$", loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 1.3)

    ax = axes[1]
    d = tha[tha.K > 1].copy()
    preds = [("wc_mu", "first-order expansion", "0.45", "v"),
             ("wc_mu_tw", "template-weighted expansion", C["R5"], "^"),
             ("wc_mu_asimov", "exact Asimov fit", C["R6h"], "*")]
    for col, lab, colr, mk in preds:
        ax.errorbar(d.meas_shift, d[col], xerr=d.meas_shift_se, yerr=d[col + "_se"],
                    color=colr, marker=mk, ms=5 if mk != "*" else 9, ls="none", capsize=2,
                    lw=0.9, label=lab)
    lim = [-0.01, 0.135]
    ax.plot(lim, lim, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"measured shift of $\hat\mu$, R1 $-$ R3")
    ax.set_ylabel(r"predicted shift")
    ax.set_title("(b) predicted vs measured shift", loc="left")
    ax.legend(frameon=False, loc="upper left")
    for _, r in d[d.n_mc == 2000].iterrows():
        ax.annotate(rf"$N_{{\rm MC}}$=2000, $K$={int(r.K)}", (r.meas_shift, r.wc_mu_asimov),
                    xytext=(-8, 10), textcoords="offset points", fontsize=5.8, color="0.3",
                    ha="right")
    ax.annotate(r"$N_{\rm MC}$=10000," + "\n" + r"$K$=20, 100", (0.012, 0.009), xytext=(22, 18),
                textcoords="offset points", fontsize=5.8, color="0.3",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))
    save(fig, out, "fig5_theory")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p4d", default="results/phase4d_raw.csv")
    p.add_argument("--theory", default="results/theory_summary.csv")
    p.add_argument("--theory_asimov", default="results/theory_asimov_summary.csv")
    p.add_argument("--out", default="paper")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = pd.read_csv(a.p4d)
    raw = raw[raw.lik == "bbp"]
    fig_schematic(a.out)
    fig_fpr_vs_K(raw, a.out)
    fig_fpr_ecdf(raw, a.out)
    fig_tradeoff(raw, a.out)
    fig_theory(pd.read_csv(a.theory), pd.read_csv(a.theory_asimov), a.out)


if __name__ == "__main__":
    main()
