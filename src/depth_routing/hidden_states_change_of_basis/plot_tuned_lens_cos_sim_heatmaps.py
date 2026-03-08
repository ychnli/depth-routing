"""
2_plot_tuned_lens_cos_sim_heatmaps.py
─────────────────────────
Stage 2 of 2: plot layer-wise cosine similarity heatmaps from
TunedLens-projected hidden states, stratified by token loss tercile (based on 
probability assigned for ground truth next-token).

Reads the .npz file produced by collect_tuned_lens_projections.py and
produces the same upper-triangular heatmap figure as plot_layerwise_cosine.py,
but with representations projected into the final layer's basis first.

Usage
-----
  python plot_tuned_lens_cos_sim_heatmaps.py \\
      --data-file  tuned_lens_projections/tuned_lens_cossims.npz \\
      --output-dir tuned_lens_figures
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Plot TunedLens-projected layer-wise cosine similarity heatmaps."
)
parser.add_argument("--data-file",  default="tuned_lens_projections/tuned_lens_cossims.npz",
                    help="Path to .npz file from collect_tuned_lens_projections.py")
parser.add_argument("--output-dir", default=".",
                    help="Directory for output PDF/PNG (default: current directory)")
args = parser.parse_args()

DATA_FILE  = args.data_file
OUTPUT_DIR = args.output_dir
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# 1.  Load
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading {DATA_FILE} …")
data        = np.load(DATA_FILE, allow_pickle=True)
all_losses  = data["all_losses"].astype(np.float32)   # (N,)
all_cossims = data["all_cossims"].astype(np.float32)  # (N, L, L)
L           = int(data["n_layers_p1"])

print(f"  tokens:      {len(all_losses):,}")
print(f"  n_layers+1:  {L}")


# 2.  Bin by loss tercile and average
# ─────────────────────────────────────────────────────────────────────────────

t33, t67 = np.percentile(all_losses, [33.33, 66.67])
print(f"  loss tercile boundaries: {t33:.3f} / {t67:.3f}")

masks  = [
    all_losses <= t33,
    (all_losses > t33) & (all_losses <= t67),
    all_losses > t67,
]
labels = ["Low Loss", "Mid Loss", "High Loss"]
counts = [m.sum() for m in masks]

avg_mats = []
for mask, lbl, cnt in zip(masks, labels, counts):
    mat = all_cossims[mask].mean(axis=0)   # (L, L)
    avg_mats.append(mat)
    print(f"  {lbl}: {cnt:,} tokens")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── Figure and panel layout ─────────────────────────────
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
         + XLABEL_H + PANEL_IN + PANEL_TITLE + TITLE_GAP + SUPTITLE_H)

# ── Fonts and matplotlib style ─────────────────────────
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
})

# ── Annotation formatting ─────────────────────────────
def format_cosine(v):
    if v >= 1.0:
        return "1"
    elif v <= -1.0:
        return "-1"
    elif v < 0:
        return f"{v:.2f}"[0] + f"{v:.2f}"[2:]
    else:
        return f".{v:.2f}"[2:]

cell_pt  = (PANEL_IN * 72) / L
ann_fs   = max(4.0, min(6.0, cell_pt * 0.50))
tick_fs  = max(5.0, min(7.5, cell_pt * 0.65))
title_fs = 7.5
subt_fs  = 5.8

tick_step = 2 if L > 8 else 1
tick_pos  = list(range(0, L, tick_step))

# ── Colormap and normalization ────────────────────────
cmap = plt.get_cmap("RdBu_r")  # professional diverging colormap
vmin, vmax = -1.0, 1.0
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax if vmax != vmin else vmin + 1e-9)

# ── Figure & axes ─────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H))
bot_panel = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP + XLABEL_H) / FIG_H

axes = []
for k in range(3):
    left = (LMARGIN_IN + k * (PANEL_IN + HGAP_IN)) / FIG_W
    ax   = fig.add_axes([left, bot_panel, PANEL_IN / FIG_W, PANEL_IN / FIG_H])
    axes.append(ax)

# ── Draw heatmaps ─────────────────────────────────────
for ax, mat, lbl, cnt in zip(axes, avg_mats, labels, counts):
    display = mat.copy()
    display[np.tril(np.ones_like(mat), k=-1).astype(bool)] = np.nan  # mask lower triangle

    im = ax.imshow(
        display,
        cmap=cmap,
        norm=norm,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )

    # ── Annotations ───────────────────────────────
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

    # ── Ticks & title ─────────────────────────────
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_pos, fontsize=tick_fs)
    ax.set_yticklabels(tick_pos, fontsize=tick_fs)
    ax.tick_params(length=2, width=0.4, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#c0c0d0")

    ax.set_title(f"{lbl}\n" + r"$\it{n}$" + f" = {cnt:,}",
                 fontsize=title_fs, fontweight="bold",
                 pad=4, color="#12122a", linespacing=1.6)

axes[0].set_ylabel("Layer  $\\ell^*$",
                   fontsize=tick_fs + 0.5, labelpad=4, color="#2a2a44")
for ax in axes:
    ax.set_xlabel("Layer  $\\ell$",
                  fontsize=tick_fs + 0.5, labelpad=4, color="#2a2a44")
for ax in axes[1:]:
    ax.set_yticklabels([])
    ax.set_ylabel("")

# ── Horizontal colorbar ─────────────────────────────
cb_bot  = (BOT_PAD + CBAR_LABEL) / FIG_H
cb_h    = CBAR_H / FIG_H
cb_left = (LMARGIN_IN + 0.12) / FIG_W
cb_w    = (3 * PANEL_IN + 2 * HGAP_IN - 0.24) / FIG_W

cbar_ax = fig.add_axes([cb_left, cb_bot, cb_w, cb_h])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Cosine similarity", fontsize=subt_fs + 0.3, labelpad=3, color="#2a2a44")
cbar.ax.tick_params(labelsize=subt_fs - 0.2, length=2.5, width=0.4, colors="#2a2a44")
cbar.outline.set_linewidth(0.35)

# ── Figure title & subtitle ─────────────────────────
title_y = 1.0 - (SUPTITLE_H * 0.28) / FIG_H
sub_y   = 1.0 - (SUPTITLE_H * 0.28 + SUPTITLE_H * 0.42) / FIG_H

fig.text(0.5, title_y,
         "TunedLens-Projected Layer-wise Cosine Similarity",
         ha="center", va="top",
         fontsize=title_fs + 1.0, fontweight="bold", color="#12122a")
fig.text(0.5, sub_y,
         "Hidden states projected to final-layer basis via TunedLens; binned by loss tercile",
         ha="center", va="top",
         fontsize=subt_fs + 0.5, color="#50507a", style="italic")

plt.show()

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out = Path(OUTPUT_DIR) / f"tuned_lens_cosine_similarity.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")