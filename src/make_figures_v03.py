#!/usr/bin/env python3
"""
v0.3 figures from the round-2 analysis tables (analyze_round2.py outputs):
  fig3_validity.pdf : paired shift of mu-hat relative to each workflow's OWN matched
                      independent-template control (95% CI) vs oracle Fisher information
  fig4_power.pdf    : calibrated size vs power at mu = 0.5, conventional calibration (filled)
                      and full-pipeline calibration (open)
Okabe-Ito colours plus distinct markers (no information carried by colour alone).
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.3, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "mathtext.fontset": "cm"})
FULL = 6.3
STY = {  # workflow -> (label, colour, marker)
    "R1":  ("R1  $k$-fold selection", "#D55E00", "o"),
    "R2":  ("R2  20% hold-out", "#56B4E9", "v"),
    "R2h": ("R2h 50% hold-out", "#CC79A7", "<"),
    "R5":  ("R5  fold-back", "#0072B2", "^"),
    "CS":  ("CS  cross-selection", "#009E73", "*"),
    "CSs": ("CSs cross-sel., strict split", "#999999", "P"),
}
ORDER = ["R1", "R2", "R2h", "R5", "CS", "CSs"]
NOM = 2.275


def ms(mk):
    return 9 if mk == "*" else 5.5


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig); print("wrote", name)


def fig_validity(t1, t2, out, K=100):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw=dict(wspace=0.3))
    for ax, N in zip(axes, (2000, 10000)):
        a = t1[(t1.N == N) & (t1.K == K)].set_index("W")
        b = t2[(t2.N == N) & (t2.K == K)].set_index("W")
        ax.axhspan(-0.1, 0.1, color="0.9", zorder=0)
        ax.axhline(0, color="0.5", lw=0.7, ls=(0, (1, 2)))
        for w in ORDER:
            lab, col, mk = STY[w]
            y, lo, hi = a.loc[w, "dmu_sig"], a.loc[w, "dmu_lo"], a.loc[w, "dmu_hi"]
            x, xe = b.loc[w, "dI_true_inf_pct"], b.loc[w, "dI_true_inf_se"]
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], xerr=xe, color=col, marker=mk,
                        ms=ms(mk), ls="none", capsize=2, lw=1, label=lab)
        ax.set_xlabel(r"oracle information vs R1 [%]")
        ax.set_title(rf"$N_{{\rm MC}} = {N:,}$, $K = {K}$".replace(",", r"\,", 1), loc="left")
        ax.text(0.03, 0.06, r"$\pm 0.1\,\sigma_{\hat\mu}$", transform=ax.transAxes, fontsize=7,
                color="0.4")
    axes[0].set_ylabel(r"shift of $\hat\mu$ vs own control  [$\sigma_{\hat\mu}$]")
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.8)
    save(fig, out, "fig3_validity")


def fig_power(t6, out, K=100, mu="mu05"):
    col_p = "power_mu05" if mu == "mu05" else "power_mu025"
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8), gridspec_kw=dict(wspace=0.28))
    for ax, N in zip(axes, (2000, 10000)):
        d = t6[(t6.N == N) & (t6.K == K)]
        ax.axvline(NOM, color="0.5", lw=0.7, ls=(0, (1, 2)))
        for w in ORDER:
            lab, col, mk = STY[w]
            for cal, fill in (("matched", True), ("own", False)):
                r = d[(d.W == w) & (d.calibration == cal)].iloc[0]
                ax.errorbar(r.fpr, r[col_p], xerr=r.fpr_se, yerr=r[col_p + "_se"], color=col,
                            marker=mk, ms=ms(mk), ls="none", capsize=1.5, lw=0.8,
                            mfc=col if fill else "white", mec=col)
        ax.set_xlabel(r"calibrated $Z>2$ rate [%]")
        ax.set_title(rf"$N_{{\rm MC}} = {N:,}$, $K = {K}$, $\mu = 0.5$".replace(",", r"\,", 1),
                     loc="left", pad=11)
        ax.text(NOM, 1.01, "nominal", transform=ax.get_xaxis_transform(), fontsize=6.3,
                color="0.4", ha="center", va="bottom")
    axes[0].set_ylabel(r"power  [%]")
    hd = [Line2D([], [], color=STY[w][1], marker=STY[w][2], ls="none", ms=ms(STY[w][2]),
                 label=STY[w][0]) for w in ORDER]
    hd += [Line2D([], [], color="0.3", marker="o", ls="none", ms=5, label="conventional calibration"),
           Line2D([], [], color="0.3", marker="o", ls="none", ms=5, mfc="white",
                  label="full-pipeline calibration")]
    fig.legend(handles=hd, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4,
               frameon=False, fontsize=6.8, columnspacing=1.2, handletextpad=0.3)
    save(fig, out, "fig4_power")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="results/round2_analysis")
    p.add_argument("--out", default="paper_v0.3")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t1 = pd.read_csv(a.prefix + "_T1_matched.csv")
    t2 = pd.read_csv(a.prefix + "_T2_absolute.csv")
    t6 = pd.read_csv(a.prefix + "_T6_calibration.csv")
    fig_validity(t1, t2, a.out)
    fig_power(t6, a.out)


if __name__ == "__main__":
    main()
