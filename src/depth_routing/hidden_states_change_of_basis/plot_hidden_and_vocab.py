"""
plot_hidden_and_vocab.py
─────────────────────────────────────────────────────────────────────────────
Reads per-excerpt .npz files produced by token_evolution_data_collect.py
and plots a combined figure with two rows of heatmaps:

  Row 1 — cosine similarity in HIDDEN space  (tl_cossim_hidden)
           Rows/cols index TunedLens-projected, LayerNorm-normalised hidden
           states h_ln at each layer.

  Row 2 — cosine similarity in VOCAB space   (tl_cossim_vocab)
           Rows/cols index the logit vectors (h_ln @ U.T) at each layer.

Both rows are binned by the same loss tercile boundaries (computed from
token_losses pooled across all excerpts), so comparisons across rows are
on identical token populations.

Usage
-----
  python plot_hidden_and_vocab.py \\
      --data-dir  /path/to/depth_routing_data \\
      --output-dir figures/
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Plot TunedLens cosine-similarity heatmaps (hidden + vocab space)."
)
parser.add_argument("--data-dir",   required=True,
                    help="Root output directory containing excerpts/excerpt_NNNN.npz")
parser.add_argument("--output-dir", default=".",
                    help="Directory for saved PDF/PNG (default: current directory)")
args = parser.parse_args()

DATA_DIR   = Path(args.data_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load and pool across all excerpts
# ─────────────────────────────────────────────────────────────────────────────

excerpt_files = sorted((DATA_DIR / "excerpts").glob("excerpt_*.npz"))
if not excerpt_files:
    raise FileNotFoundError(f"No excerpt_*.npz files found in {DATA_DIR / 'excerpts'}")

print(f"Found {len(excerpt_files)} excerpt files in {DATA_DIR / 'excerpts'}")

all_losses  = []   # (N,)        float32
all_hidden  = []   # (N, L, L)   float32
all_vocab   = []   # (N, L, L)   float32

for path in excerpt_files:
    d = np.load(path, allow_pickle=True)
    all_losses.append(d["token_losses"].astype(np.float32))           # (seq_len-1,)
    all_hidden.append(d["tl_cossim_hidden"].astype(np.float32))       # (seq_len-1, L, L)
    all_vocab.append( d["tl_cossim_vocab"].astype(np.float32))        # (seq_len-1, L, L)
    d.close()

all_losses = np.concatenate(all_losses, axis=0)   # (N,)
all_hidden = np.concatenate(all_hidden, axis=0)   # (N, L, L)
all_vocab  = np.concatenate(all_vocab,  axis=0)   # (N, L, L)

N, L, _ = all_hidden.shape
print(f"  Total tokens: {N:,}   Layers: {L}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Bin by loss tercile — shared boundaries for both rows
# ─────────────────────────────────────────────────────────────────────────────

t33, t67 = np.percentile(all_losses, [33.33, 66.67])
print(f"  Loss tercile boundaries: {t33:.3f} / {t67:.3f}")

masks = [
    all_losses <= t33,
    (all_losses > t33) & (all_losses <= t67),
    all_losses > t67,
]
bin_labels = ["Low Loss", "Mid Loss", "High Loss"]
counts     = [int(m.sum()) for m in masks]

# Average (L, L) matrices per bin, for each space
avg_hidden = [all_hidden[m].mean(axis=0) for m in masks]   # list of 3 × (L, L)
avg_vocab  = [all_vocab[m].mean(axis=0)  for m in masks]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Figure layout  (two rows, three columns)
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

BOT_PAD     = 0.05
CBAR_LABEL  = 0.18
CBAR_H      = 0.14
CBAR_GAP    = 0.14
XLABEL_H    = 0.30
PANEL_TITLE = 0.28
TITLE_GAP   = 0.08
ROW_GAP     = 0.38   # gap between the two rows (space for row label)
SUPTITLE_H  = 0.52

# Total height: bottom padding + colorbar area + xlabel + 2 panel rows +
# panel titles + row gap + suptitle
FIG_H = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP
         + XLABEL_H + PANEL_IN + PANEL_TITLE   # bottom row (vocab)
         + ROW_GAP
         + XLABEL_H + PANEL_IN + PANEL_TITLE   # top row (hidden)
         + TITLE_GAP + SUPTITLE_H)

# ── Font sizes derived from panel size ──────────────────
cell_pt  = (PANEL_IN * 72) / L
ann_fs   = max(4.0, min(6.0, cell_pt * 0.50))
tick_fs  = max(5.0, min(7.5, cell_pt * 0.65))
title_fs = 7.5
subt_fs  = 5.8

tick_step = 2 if L > 8 else 1
tick_pos  = list(range(0, L, tick_step))

# ── Colormap ─────────────────────────────────────────────
cmap = plt.get_cmap("RdBu_r")
vmin, vmax = -1.0, 1.0
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Helper: annotation text
# ─────────────────────────────────────────────────────────────────────────────

def format_cosine(v):
    if v >= 1.0:   return "1"
    if v <= -1.0:  return "-1"
    if v < 0:      return f"{v:.2f}"[0] + f"{v:.2f}"[2:]
    return f".{v:.2f}"[2:]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Draw
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Bottom of each panel row in figure-fraction coordinates
# Row 0 = vocab (bottom), Row 1 = hidden (top)
vocab_bot  = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP + XLABEL_H) / FIG_H
hidden_bot = vocab_bot + (PANEL_IN + PANEL_TITLE + ROW_GAP + XLABEL_H) / FIG_H

row_configs = [
    # (avg_mats,   bot_fraction,  row_label)
    (avg_vocab,  vocab_bot,   "Vocab space"),
    (avg_hidden, hidden_bot,  "Hidden space"),
]

im = None   # will hold the last imshow for the colorbar

for avg_mats, bot_frac, row_label in row_configs:

    # Row label on the left
    label_y = bot_frac + (PANEL_IN / FIG_H) / 2
    fig.text(
        0.005, label_y, row_label,
        ha="left", va="center",
        fontsize=tick_fs + 0.5, fontweight="bold", color="#2a2a44",
        rotation=90,
    )

    for k, (mat, lbl, cnt) in enumerate(zip(avg_mats, bin_labels, counts)):
        left = (LMARGIN_IN + k * (PANEL_IN + HGAP_IN)) / FIG_W
        ax   = fig.add_axes([left, bot_frac, PANEL_IN / FIG_W, PANEL_IN / FIG_H])

        display = mat.copy()
        display[np.tril(np.ones_like(mat), k=-1).astype(bool)] = np.nan

        im = ax.imshow(
            display,
            cmap=cmap, norm=norm,
            aspect="equal", interpolation="nearest", origin="upper",
        )

        # Annotations
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

        # Column titles only on the top row
        if row_label == "Hidden space":
            ax.set_title(
                f"{lbl}\n" + r"$\it{n}$" + f" = {cnt:,}",
                fontsize=title_fs, fontweight="bold",
                pad=4, color="#12122a", linespacing=1.6,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Shared colorbar
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
# 7.  Figure title and subtitle
# ─────────────────────────────────────────────────────────────────────────────

title_y = 1.0 - (SUPTITLE_H * 0.28) / FIG_H
sub_y   = 1.0 - (SUPTITLE_H * 0.28 + SUPTITLE_H * 0.42) / FIG_H

fig.text(0.5, title_y,
         "TunedLens-Projected Layer-wise Cosine Similarity",
         ha="center", va="top",
         fontsize=title_fs + 1.0, fontweight="bold", color="#12122a")
fig.text(0.5, sub_y,
         "Projected to final-layer basis via TunedLens; binned by loss tercile",
         ha="center", va="top",
         fontsize=subt_fs + 0.5, color="#50507a", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Save
# ─────────────────────────────────────────────────────────────────────────────

for ext in ("pdf", "png"):
    out = OUTPUT_DIR / f"tuned_lens_cosine_similarity.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")

plt.show()