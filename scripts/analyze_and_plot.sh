#!/usr/bin/env bash
# Regenerate the analysis tables of the confirmatory run and all figures from the stored
# outputs in results/ (no refitting). Figures are written to figures/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p figures
python src/analyze_round2.py --prefix results/round2 --out results/round2_analysis | tee results/round2_analysis.txt
python src/make_figures_v02.py --out figures      # fig1_schematic, fig2_fpr_vs_K, fig5_theory (+ two v0.2 figures)
python src/make_figures_v03.py --out figures      # fig3_validity, fig4_power
echo "tables: results/round2_analysis_T*.csv   figures: figures/"
