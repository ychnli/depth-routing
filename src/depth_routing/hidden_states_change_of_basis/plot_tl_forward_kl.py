"""
plot_tl_forward_kl.py
────────────────────────────────────────────────────────────────────────────────
Two side-by-side panels in a single figure:

  Left  — Forward KL to final prediction vs. layer
           tl_forward_kl[l, j]: KL between TunedLens at layer l and the
           final-layer output.  Measures how far each layer's prediction is
           from the model's eventual answer.  Drops to 0 at layer L-1
           by construction.

  Right — Adjacent-layer KL vs. layer transition
           tl_adjacent_kl[l, j]: KL(TunedLens layer l || TunedLens layer l+1).
           Measures how much the prediction changes between consecutive layers.
           High values identify which layers are doing the most work.

Both panels are stratified by TunedLens stabilization bin (Easy / Medium / Hard)
with ±1 SEM shading.

Requires tl_data_utils.py in the same directory.

Usage
-----
  python plot_tl_forward_kl.py \\
      --data-dir  /path/to/depth_routing_data \\
      --output-dir figures/ \\
      --early-thresh 2 \\
      --late-thresh  6
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from tuned_lens_data_utils import load_and_pool, compute_stabilization_bins

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",     required=True)
parser.add_argument("--output-dir",   default=".")
parser.add_argument("--early-thresh", type=int, default=2)
parser.add_argument("--late-thresh",  type=int, default=6)
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load and bin
# ─────────────────────────────────────────────────────────────────────────────

pool = load_and_pool(DATA_DIR)
L    = pool["L"]

_, masks, bin_labels, counts = compute_stabilization_bins(
    pool["tl_top_token_ids"],
    early_thresh = args.early_thresh,
    late_thresh  = args.late_thresh,
)

fkl    = pool["tl_forward_kl"]    # (N, L)
adj_kl = pool["tl_adjacent_kl"]   # (N, L-1)

fkl_layers = np.arange(L)           # 0 … L-1  (integer layer indices)
adj_layers = np.arange(L - 1)       # 0 … L-2  (index of the "from" layer)

def bin_stats(arr, masks):
    means, sems = [], []
    for mask in masks:
        sub = arr[mask]
        means.append(sub.mean(axis=0))
        sems.append(sub.std(axis=0) / np.sqrt(sub.shape[0]))
    return means, sems

fkl_means, fkl_sems = bin_stats(fkl,    masks)
adj_means, adj_sems = bin_stats(adj_kl, masks)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Style
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
})

COLORS    = ["#2166ac", "#f4a582", "#b2182b"]
FIG_W     = 8.4
FIG_H     = 2.8
title_fs  = 8.0
subt_fs   = 5.8
tick_fs   = 6.5
label_fs  = 7.0
legend_fs = 6.2
tick_step = 2 if L > 8 else 1


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Helper — draw one panel
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(ax, x, means, sems, ylabel, title, show_legend, xlabels=None):
    for mean, sem, lbl, cnt, col in zip(means, sems, bin_labels, counts, COLORS):
        ax.plot(x, mean, color=col, linewidth=1.4,
                label=f"{lbl}  ($n$={cnt:,})")
        ax.fill_between(x, mean - sem, mean + sem,
                        color=col, alpha=0.15, linewidth=0)

    # Bin boundary markers
    for thresh, ls in [(args.early_thresh, "--"), (args.late_thresh, ":")]:
        ax.axvline(thresh, color="#aaaaaa", linewidth=0.7, linestyle=ls, zorder=0)

    if xlabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=tick_fs - 0.5,
                           rotation=45, ha="right")
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    else:
        ax.set_xticks(range(0, L, tick_step))
        ax.set_xlim(-0.3, L - 0.7)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Layer", fontsize=label_fs, color="#2a2a44", labelpad=3)
    ax.set_ylabel(ylabel, fontsize=label_fs, color="#2a2a44", labelpad=4)
    ax.tick_params(labelsize=tick_fs, length=2.5, width=0.4,
                   pad=2, colors="#2a2a44")
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#c0c0d0")

    ax.set_title(title, fontsize=title_fs, fontweight="bold",
                 color="#12122a", pad=5)

    if show_legend:
        ax.legend(
            fontsize=legend_fs, frameon=True, framealpha=0.9,
            edgecolor="#c0c0d0", facecolor="white",
            borderpad=0.6, handlelength=1.6, labelspacing=0.35,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Figure
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax_fkl, ax_adj) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

draw_panel(
    ax_fkl,
    x           = fkl_layers,
    means       = fkl_means,
    sems        = fkl_sems,
    ylabel      = "Mean forward KL  (nats)",
    title       = "Forward KL to Final Prediction",
    show_legend = True,
)

draw_panel(
    ax_adj,
    x           = adj_layers,
    means       = adj_means,
    sems        = adj_sems,
    ylabel      = "Mean adjacent KL  (nats)",
    title       = r"Adjacent-Layer KL  ($\ell\ {\to}\ \ell{+}1$)",
    show_legend = False,
    xlabels     = [f"{l}→{l+1}" for l in range(L - 1)],
)

# Shared subtitle
fig.text(
    0.5, -0.04,
    f"Binned by TunedLens stabilization layer  "
    f"(easy ≤ {args.early_thresh}, medium ≤ {args.late_thresh}, "
    f"hard > {args.late_thresh})  ·  shaded band = ±1 SEM",
    ha="center", va="top",
    fontsize=subt_fs, color="#50507a", style="italic",
)

plt.tight_layout(rect=[0, 0.07, 1, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Save
# ─────────────────────────────────────────────────────────────────────────────

for ext in ("pdf", "png"):
    out = OUTPUT_DIR / f"tl_kl_curves.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")

plt.show()