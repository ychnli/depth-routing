"""
plot_tl_cos_sim_hidden.py
────────────────────────────────────────────────────────────────────────────────
Plots hidden-space cosine similarity heatmaps from TunedLens-projected states,
binned by how early TunedLens stabilizes its top-1 prediction.

Three panels (Easy / Medium / Hard), each showing the average (L × L)
cosine-similarity matrix over tl_proj_ln_states at each layer pair,
upper-triangular only.

Requires tl_data_utils.py in the same directory.

Usage
-----
  python plot_tl_cos_sim_hidden.py \\
      --data-dir  /path/to/depth_routing_data \\
      --output-dir figures/ \\
      --early-thresh 2 \\
      --late-thresh  6
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

from tuned_lens_data_utils import load_and_pool, compute_stabilization_bins

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Hidden-space cosine similarity heatmaps, binned by TunedLens stabilization."
)
parser.add_argument("--data-dir",      required=True)
parser.add_argument("--output-dir",    default=".")
parser.add_argument("--early-thresh",  type=int, default=2,
                    help="Stabilization layer ≤ this → Easy (default: 2)")
parser.add_argument("--late-thresh",   type=int, default=6,
                    help="Stabilization layer ≤ this → Medium, else Hard (default: 6)")
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load, bin, average
# ─────────────────────────────────────────────────────────────────────────────

pool = load_and_pool(DATA_DIR)
L    = pool["L"]

_, masks, bin_labels, counts = compute_stabilization_bins(
    pool["tl_top_token_ids"],
    early_thresh = args.early_thresh,
    late_thresh  = args.late_thresh,
)

avg_mats = [pool["tl_cossim_hidden"][m].mean(axis=0) for m in masks]  # 3 × (L, L)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Layout constants  (identical proportions to original script)
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
})

FIG_W      = 6.0
LMARGIN_IN = 0.48
RMARGIN_IN = 0.06
HGAP_IN    = 0.14
PANEL_IN   = (FIG_W - LMARGIN_IN - RMARGIN_IN - 2 * HGAP_IN) / 3

BOT_PAD    = 0.05
CBAR_LABEL = 0.18
CBAR_H     = 0.14
CBAR_GAP   = 0.14
XLABEL_H   = 0.30
PANEL_TITLE= 0.28
TITLE_GAP  = 0.08
SUPTITLE_H = 0.52

FIG_H = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP
         + XLABEL_H + PANEL_IN + PANEL_TITLE
         + TITLE_GAP + SUPTITLE_H)

cell_pt  = (PANEL_IN * 72) / L
ann_fs   = max(4.0, min(6.0, cell_pt * 0.50))
tick_fs  = max(5.0, min(7.5, cell_pt * 0.65))
title_fs = 7.5
subt_fs  = 5.8

tick_step = 2 if L > 8 else 1
tick_pos  = list(range(0, L, tick_step))

cmap = plt.get_cmap("RdBu_r")
vmin, vmax = -1.0, 1.0
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Annotation helper
# ─────────────────────────────────────────────────────────────────────────────

def format_cosine(v):
    if v >= 1.0:  return "1"
    if v <= -1.0: return "-1"
    if v < 0:     return f"{v:.2f}"[0] + f"{v:.2f}"[2:]
    return f".{v:.2f}"[2:]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Draw
# ─────────────────────────────────────────────────────────────────────────────

fig  = plt.figure(figsize=(FIG_W, FIG_H))
bot  = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP + XLABEL_H) / FIG_H
im   = None

for k, (mat, lbl, cnt) in enumerate(zip(avg_mats, bin_labels, counts)):
    left = (LMARGIN_IN + k * (PANEL_IN + HGAP_IN)) / FIG_W
    ax   = fig.add_axes([left, bot, PANEL_IN / FIG_W, PANEL_IN / FIG_H])

    display = mat.copy()
    display[np.tril(np.ones_like(mat), k=-1).astype(bool)] = np.nan

    im = ax.imshow(
        display,
        cmap=cmap, norm=norm,
        aspect="equal", interpolation="nearest", origin="upper",
    )

    for r in range(L):
        for c in range(r, L):
            v = display[r, c]
            if np.isnan(v):
                continue
            brightness = (v - vmin) / (vmax - vmin + 1e-9)
            tcol = "white" if brightness > 0.60 else "#12122a"
            ax.text(c, r, format_cosine(v),
                    ha="center", va="center",
                    fontsize=ann_fs, color=tcol)

    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_pos, fontsize=tick_fs)
    ax.set_yticklabels(tick_pos if k == 0 else [], fontsize=tick_fs)
    ax.tick_params(length=2, width=0.4, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#c0c0d0")

    ax.set_xlabel("Layer  $\\ell$", fontsize=tick_fs + 0.5,
                  labelpad=4, color="#2a2a44")
    if k == 0:
        ax.set_ylabel("Layer  $\\ell^*$", fontsize=tick_fs + 0.5,
                      labelpad=4, color="#2a2a44")

    ax.set_title(
        f"{lbl}\n" + r"$\it{n}$" + f" = {cnt:,}",
        fontsize=title_fs, fontweight="bold",
        pad=4, color="#12122a", linespacing=1.6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Colorbar
# ─────────────────────────────────────────────────────────────────────────────

cb_bot  = (BOT_PAD + CBAR_LABEL) / FIG_H
cb_h    = CBAR_H / FIG_H
cb_left = (LMARGIN_IN + 0.12) / FIG_W
cb_w    = (3 * PANEL_IN + 2 * HGAP_IN - 0.24) / FIG_W

cbar_ax = fig.add_axes([cb_left, cb_bot, cb_w, cb_h])
cbar    = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Cosine similarity", fontsize=subt_fs + 0.3,
               labelpad=3, color="#2a2a44")
cbar.ax.tick_params(labelsize=subt_fs - 0.2, length=2.5,
                    width=0.4, colors="#2a2a44")
cbar.outline.set_linewidth(0.35)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Title
# ─────────────────────────────────────────────────────────────────────────────

title_y = 1.0 - (SUPTITLE_H * 0.28) / FIG_H
sub_y   = 1.0 - (SUPTITLE_H * 0.28 + SUPTITLE_H * 0.42) / FIG_H

fig.text(0.5, title_y,
         "TunedLens Hidden-Space Layer-wise Cosine Similarity",
         ha="center", va="top",
         fontsize=title_fs + 1.0, fontweight="bold", color="#12122a")
fig.text(0.5, sub_y,
         f"Binned by TunedLens stabilization layer  "
         f"(easy ≤ {args.early_thresh}, medium ≤ {args.late_thresh}, hard > {args.late_thresh})",
         ha="center", va="top",
         fontsize=subt_fs + 0.5, color="#50507a", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Save
# ─────────────────────────────────────────────────────────────────────────────

for ext in ("pdf", "png"):
    out = OUTPUT_DIR / f"tl_cossim_hidden.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")

plt.show()