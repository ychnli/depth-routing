"""
Layer-wise cosine similarity heatmaps stratified by token loss tercile.

For each token position j with loss L[j]:
  - Use hidden_states[:, j, :] (shape: n_layers+1 x hidden_dim)
  - Compute upper-triangular cosine similarity matrix between layer l* and l > l*
  - Bin token by L[j] into tercile (low / mid / high loss)
  - Average cosine similarity matrices within each tercile
  - Plot three side-by-side upper-triangular heatmaps

The last token of each excerpt is excluded (no associated loss).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm

import sys
import argparse
sys.path.insert(0, ".")          # so we can import from the token_evolution_data_collect script
from token_evolution_data_collect import load_excerpts, list_excerpt_ids   # adjust import if needed

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Plot layer-wise cosine similarity heatmaps stratified by token loss tercile."
)
parser.add_argument(
    "--data-dir", default="token_evolution_data",
    help="Root directory produced by collect_token_evolution_data (default: token_evolution_data)",
)
parser.add_argument(
    "--output-dir", default=".",
    help="Directory in which to write the output PDF/PNG (default: current directory)",
)
args = parser.parse_args()

DATA_DIR   = args.data_dir
OUTPUT_DIR = args.output_dir
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Scanning for excerpt IDs …")
ids = list_excerpt_ids(DATA_DIR)
print(f"  found {len(ids)} excerpts")

print("Loading excerpts …")
excerpts = load_excerpts(DATA_DIR, ids)

# Filter to those that actually contain hidden states
excerpts = [e for e in excerpts if "hidden_states" in e]
print(f"  {len(excerpts)} excerpts contain hidden_states")

if len(excerpts) == 0:
    raise RuntimeError(
        "No excerpts with hidden_states found. "
        "Re-run collection with --save-hidden-states."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Accumulate per-token cosine-similarity matrices + losses
# ─────────────────────────────────────────────────────────────────────────────

# We need to know n_layers from the first excerpt
n_layers_p1 = excerpts[0]["hidden_states"].shape[0]   # n_layers + 1
n_layers    = n_layers_p1                              # label convenience
print(f"  n_layers+1 = {n_layers_p1}")

# Storage: for each tercile we accumulate sum of cosine-sim matrices + count
# Shape of one cosine-sim matrix: (n_layers_p1, n_layers_p1)
# (upper triangular; entry [l1, l2] = cosine sim between layer l1 and l2, l1 <= l2)

all_losses  = []   # flat list of scalars
all_cossims = []   # flat list of (n_layers_p1, n_layers_p1) arrays

def cosine_sim_matrix(H):
    """
    H : (n_layers+1, hidden_dim) — one token's hidden states across layers
    Returns upper-triangular cosine similarity matrix of shape (L, L),
    with diagonal = 1.0 by construction.
    """
    L = H.shape[0]
    # L2-normalise each row
    norms = np.linalg.norm(H, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Hn = H / norms                          # (L, hidden_dim)
    sim = Hn @ Hn.T                          # (L, L)
    # clip for numerical safety
    sim = np.clip(sim, -1.0, 1.0)
    return sim                               # full; we'll mask lower tri later

print("Computing cosine-similarity matrices …")
for e in tqdm(excerpts):
    hs     = e["hidden_states"].astype(np.float32)   # (L, seq_len, hidden_dim)
    losses = e["token_losses"].astype(np.float32)    # (seq_len-1,)

    seq_len = losses.shape[0]   # number of (loss, hidden-state) pairs

    for j in range(seq_len):
        H = hs[:, j, :]                  # (n_layers_p1, hidden_dim)
        sim = cosine_sim_matrix(H)       # (n_layers_p1, n_layers_p1)
        all_losses.append(losses[j])
        all_cossims.append(sim)

all_losses  = np.array(all_losses, dtype=np.float32)  # (N,)
all_cossims = np.stack(all_cossims, axis=0)            # (N, L, L)
print(f"  total tokens: {len(all_losses)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Bin by loss tercile and average
# ─────────────────────────────────────────────────────────────────────────────

t33, t67 = np.percentile(all_losses, [33.33, 66.67])
print(f"  loss tercile boundaries: {t33:.3f}, {t67:.3f}")

masks = [
    all_losses <= t33,
    (all_losses > t33) & (all_losses <= t67),
    all_losses > t67,
]
labels = ["Low loss", "Mid loss", "High loss"]
counts = [m.sum() for m in masks]

avg_mats = []
for mask, lbl, cnt in zip(masks, labels, counts):
    mat = all_cossims[mask].mean(axis=0)   # (L, L)
    avg_mats.append(mat)
    print(f"  {lbl}: {cnt} tokens, mean sim on diagonal = {np.diag(mat).mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plot
# ─────────────────────────────────────────────────────────────────────────────

L = n_layers_p1

# ── Colormap: white → periwinkle → deep indigo ───────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "blue_indigo",
    list(zip(
        [0.00, 0.30, 0.60, 0.80, 0.93, 1.00],
        [(1.00, 1.00, 1.00),
         (0.88, 0.90, 0.98),
         (0.67, 0.72, 0.95),
         (0.40, 0.45, 0.85),
         (0.22, 0.18, 0.65),
         (0.10, 0.06, 0.42)],
    )),
    N=512,
)

mask_ut   = np.triu(np.ones((L, L), dtype=bool))
lower_tri = np.tril(np.ones((L, L), dtype=bool), k=-1)
vals_all  = np.concatenate([m[mask_ut] for m in avg_mats])
vmin      = max(0.0, vals_all.min())
vmax      = 1.0

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
})

# ── Layout: carve the figure into exact inches so panels are truly square ────
#
# Horizontal zones (left → right):
#   LMARGIN_IN  : space for y-axis label + ytick labels
#   PANEL_IN × 3 + HGAP_IN × 2  : the three square heatmap panels + gaps
#   RMARGIN_IN  : right breathing room
#
# Vertical zones (bottom → top):
#   BOT_PAD     : bottom breathing room
#   CBAR_LABEL  : colorbar label text
#   CBAR_H      : colorbar bar itself
#   CBAR_GAP    : gap between colorbar and panels
#   XLABEL_H    : x-axis label + ticks
#   PANEL_IN    : heatmap panel (= PANEL_IN wide → square)
#   PANEL_TITLE : per-panel title + n= line
#   TITLE_GAP   : gap between panel titles and figure suptitle
#   SUPTITLE_H  : figure title + subtitle

FIG_W        = 6.0
LMARGIN_IN   = 0.48
RMARGIN_IN   = 0.06
HGAP_IN      = 0.14
PANEL_IN     = (FIG_W - LMARGIN_IN - RMARGIN_IN - 2 * HGAP_IN) / 3   # ≈ 1.71 in

BOT_PAD      = 0.05
CBAR_LABEL   = 0.18
CBAR_H       = 0.14
CBAR_GAP     = 0.14
XLABEL_H     = 0.30
PANEL_TITLE  = 0.28
TITLE_GAP    = 0.08
SUPTITLE_H   = 0.52

FIG_H = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP
         + XLABEL_H + PANEL_IN + PANEL_TITLE + TITLE_GAP + SUPTITLE_H)

# Font sizes derived from the true panel size in points
cell_pt  = (PANEL_IN * 72) / L
ann_fs   = max(4.0, min(6.0, cell_pt * 0.50))   # annotation inside each cell
tick_fs  = max(5.0, min(7.5, cell_pt * 0.65))   # axis tick labels
title_fs = 7.5
subt_fs  = 5.8

tick_step = 2 if L > 8 else 1
tick_pos  = list(range(0, L, tick_step))

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Place the three axes at exact positions (figure fractions)
bot_panel = (BOT_PAD + CBAR_LABEL + CBAR_H + CBAR_GAP + XLABEL_H) / FIG_H
axes = []
for k in range(3):
    left = (LMARGIN_IN + k * (PANEL_IN + HGAP_IN)) / FIG_W
    ax = fig.add_axes([left, bot_panel, PANEL_IN / FIG_W, PANEL_IN / FIG_H])
    axes.append(ax)

def format_cosine(v):
    if v >= 1.0:
        return "1"
    elif v <= -1.0:
        return "-1"
    elif v < 0:
        return f"{v:.2f}"[0] + f"{v:.2f}"[2:]
    else:
        return f".{v:.2f}"[2:]
    
# ── Draw heatmaps ─────────────────────────────────────────────────────────────
for ax, mat, lbl, cnt in zip(axes, avg_mats, labels, counts):
    display = mat.copy()
    display[lower_tri] = np.nan

    im = ax.imshow(
        display,
        cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )

    # Cell annotations — drop leading zero: ".93" not "0.93"
    for r in range(L):
        for c in range(r, L):
            v = display[r, c]
            if np.isnan(v):
                continue
            brightness = (v - vmin) / (vmax - vmin + 1e-9)
            tcol = "white" if brightness > 0.60 else "#12122a"
            # label = "1" if v >= 1.0 else f"{v:.2f}"[1:]
            label = format_cosine(v)
            ax.text(c, r, label,
                    ha="center", va="center",
                    fontsize=ann_fs, color=tcol)

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

# ── Colorbar ──────────────────────────────────────────────────────────────────
cb_bot  = (BOT_PAD + CBAR_LABEL) / FIG_H
cb_h    = CBAR_H / FIG_H
cb_left = (LMARGIN_IN + 0.12) / FIG_W
cb_w    = (3 * PANEL_IN + 2 * HGAP_IN - 0.24) / FIG_W

cbar_ax = fig.add_axes([cb_left, cb_bot, cb_w, cb_h])
cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cb.set_label("Mean cosine similarity",
             fontsize=subt_fs + 0.3, labelpad=3, color="#2a2a44")
cb.ax.tick_params(labelsize=subt_fs - 0.2, length=2.5, width=0.4, colors="#2a2a44")
cb.outline.set_linewidth(0.35)

# ── Figure title & subtitle ───────────────────────────────────────────────────
top_panel = bot_panel + PANEL_IN / FIG_H
title_y = 1.0 - (SUPTITLE_H * 0.28) / FIG_H
sub_y   = 1.0 - (SUPTITLE_H * 0.28 + SUPTITLE_H * 0.42) / FIG_H

fig.text(0.5, title_y,
         "Layer-wise Cosine Similarity of Token Hidden States over Model Depth",
         ha="center", va="top",
         fontsize=title_fs + 1.0, fontweight="bold", color="#12122a")
fig.text(0.5, sub_y,
         "Binned by Next-Token Prediction Loss Tercile",
         ha="center", va="top",
         fontsize=subt_fs + 0.5, color="#50507a", style="italic")

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    out = Path(OUTPUT_DIR) / f"layerwise_cosine_similarity.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
