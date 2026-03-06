"""
Token Stabilization Depth Analysis

For each token position j (paired with loss[j]):
  - Finds the earliest layer l* such that cosine_sim(rep[l*], rep[m]) >= THRESHOLD
    for ALL subsequent layers m > l*.  That is the "stabilization depth".
  - Bins tokens into loss terciles (low / mid / high).

Plots (side by side, ~6 in wide):
  Panel A: Stabilization-depth distribution per loss tier (grouped bar chart)
  Panel B: Upper-triangular cosine-similarity heatmap for early-stabilizing
           tokens only (stab depth <= EARLY_STABILIZE_LAYER)

Data pipeline: reads per-excerpt .npz files via load_excerpts() /
list_excerpt_ids() from collect.py — same pattern as plot_layerwise_cosine.py.
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, ".")
from token_evolution_data_collect import load_excerpts, list_excerpt_ids


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Plot token stabilization depth analysis."
)
parser.add_argument("--data-dir",   default="token_evolution_data",
                    help="Root directory of .npz excerpt files")
parser.add_argument("--output-dir", default=".",
                    help="Directory for output PDF/PNG")
parser.add_argument("--threshold",  type=float, default=0.995,
                    help="Cosine-similarity threshold for stabilization (default: 0.995)")
parser.add_argument("--early-layer", type=int, default=4,
                    help="Max stabilization layer to count as 'early' (default: 4)")
args = parser.parse_args()

DATA_DIR             = args.data_dir
OUTPUT_DIR           = args.output_dir
STABILITY_THRESHOLD  = args.threshold
EARLY_STABILIZE_LAYER = args.early_layer
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Scanning for excerpt IDs …")
ids = list_excerpt_ids(DATA_DIR)
print(f"  found {len(ids)} excerpts")

print("Loading excerpts …")
excerpts = load_excerpts(DATA_DIR, ids)
excerpts = [e for e in excerpts if "hidden_states" in e]
print(f"  {len(excerpts)} excerpts contain hidden_states")

if not excerpts:
    raise RuntimeError(
        "No excerpts with hidden_states found. "
        "Re-run collection with --save-hidden-states."
    )

n_layers_p1 = excerpts[0]["hidden_states"].shape[0]   # L = n_layers + 1
L           = n_layers_p1
print(f"  n_layers+1 = {L}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Compute stabilization depths + accumulate early-token heatmap
# ─────────────────────────────────────────────────────────────────────────────

MIN_NORM = 1e-3   # hidden states with norm below this are considered degenerate
                  # (can happen due to float16 quantization flushing small values to zero)

def stabilization_depth(H_token):
    """
    H_token : (L, hidden_dim) — one token's hidden states across all layers.
    Assumes all row norms >= MIN_NORM (caller is responsible for checking).

    Returns the earliest layer index l* such that cosine_sim(H[l*], H[m]) >=
    STABILITY_THRESHOLD for every m in (l*, L-1].
    Returns -1 if no such layer exists.
    """
    norms = np.linalg.norm(H_token, axis=-1)     # (L,)
    Hn    = H_token / norms[:, None]              # (L, D) unit vectors
    sim   = np.clip(Hn @ Hn.T, -1.0, 1.0)        # (L, L)

    for l in range(L - 1):
        if (sim[l, l + 1:] >= STABILITY_THRESHOLD).all():
            return l
    return -1


print(f"\nComputing stabilization depths (threshold = {STABILITY_THRESHOLD}) …")

all_losses   = []   # (N,)
all_depths   = []   # (N,)  stabilization layer index, -1 = never
n_degenerate = 0    # tokens skipped due to near-zero norms

# For the early-token heatmap we accumulate a running sum
early_sim_sum = np.zeros((L, L), dtype=np.float64)
early_sim_n   = 0

for e in tqdm(excerpts, desc="Excerpts"):
    hs     = e["hidden_states"].astype(np.float32)   # (L, seq_len, hidden_dim)
    losses = e["token_losses"].astype(np.float32)    # (seq_len-1,)

    seq_len = losses.shape[0]   # number of paired (loss, hidden-state) positions

    for j in range(seq_len):
        H_j  = hs[:, j, :]              # (L, hidden_dim)
        loss = float(losses[j])

        # Check for degenerate vectors before calling stabilization_depth
        norms_j = np.linalg.norm(H_j, axis=-1)
        if (norms_j < MIN_NORM).any():
            n_degenerate += 1
            all_losses.append(loss)
            all_depths.append(-1)
            continue

        depth = stabilization_depth(H_j)
        all_losses.append(loss)
        all_depths.append(depth)

        # Accumulate cosine-sim matrix for early-stabilizing tokens
        # (depth != -1 already guarantees all norms >= MIN_NORM)
        if 0 <= depth <= EARLY_STABILIZE_LAYER:
            Hn  = H_j / norms_j[:, None]
            sim = np.clip(Hn @ Hn.T, -1.0, 1.0)
            early_sim_sum += sim
            early_sim_n   += 1

print("  depth distribution:", {int(d): int((all_depths == d).sum()) for d in sorted(np.unique(all_depths))})

all_losses = np.array(all_losses, dtype=np.float32)
all_depths = np.array(all_depths, dtype=np.int32)
print(f"  total tokens:     {len(all_losses):,}")
print(f"  degenerate (skipped): {n_degenerate:,} ({100*n_degenerate/max(len(all_losses),1):.1f}%)")
never_pct = (all_depths == -1).mean() * 100
print(f"  never stabilized: {never_pct:.1f}%")

early_sim_mean = early_sim_sum / max(early_sim_n, 1)
np.fill_diagonal(early_sim_mean, 1.0)
print(f"  early-stabilizing tokens (≤ layer {EARLY_STABILIZE_LAYER}): {early_sim_n:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Bin tokens into loss terciles
# ─────────────────────────────────────────────────────────────────────────────

t33, t67    = np.percentile(all_losses, [33.33, 66.67])
tier_masks  = [
    all_losses <= t33,
    (all_losses > t33) & (all_losses <= t67),
    all_losses > t67,
]
tier_labels = ["Low loss", "Mid loss", "High loss"]
tier_counts = [m.sum() for m in tier_masks]

print(f"\n  loss tercile boundaries: {t33:.3f} / {t67:.3f}")
for lbl, cnt in zip(tier_labels, tier_counts):
    print(f"  {lbl}: {cnt:,} tokens")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plot
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.linewidth":     0.5,
    "xtick.major.width":  0.4,
    "ytick.major.width":  0.4,
    "xtick.minor.width":  0.3,
    "ytick.minor.width":  0.3,
})

# ── Colormap (matches layerwise_cosine figure) ────────────────────────────────
CMAP = LinearSegmentedColormap.from_list(
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

# Tier accent colors — same periwinkle/indigo palette, three distinct stops
TIER_COLORS = [
    "#6080CC",   # low  loss — mid periwinkle blue
    "#4040A0",   # mid  loss — indigo
    "#1a1060",   # high loss — deep indigo
]

# ── Layout arithmetic (inches) ────────────────────────────────────────────────
FIG_W = 6.0

# Horizontal zones
LMARGIN    = 0.52    # y-label + ytick labels (bar panel)
RMARGIN    = 0.06
MID_GAP    = 0.48    # gap between the two panels
# Panel widths: bar panel gets slightly more room than heatmap
HEAT_IN    = (FIG_W - LMARGIN - RMARGIN - MID_GAP) * 0.44
BAR_IN     = (FIG_W - LMARGIN - RMARGIN - MID_GAP) * 0.56

# Vertical zones
BOT_PAD    = 0.05
CBAR_LBL   = 0.16
CBAR_H     = 0.12
CBAR_GAP   = 0.12
XLABEL_H   = 0.35
PANEL_H    = HEAT_IN     # heatmap is square; bar panel matches height
PANEL_TITLE= 0.26
TITLE_GAP  = 0.08
SUPTITLE_H = 0.46

FIG_H = (BOT_PAD + CBAR_LBL + CBAR_H + CBAR_GAP
         + XLABEL_H + PANEL_H + PANEL_TITLE + TITLE_GAP + SUPTITLE_H)

title_fs  = 7.5
subt_fs   = 5.8
tick_fs   = 5.8
label_fs  = 6.2
ann_fs    = max(3.8, min(5.8, (HEAT_IN * 72) / L * 0.48))

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

bot_panel = (BOT_PAD + CBAR_LBL + CBAR_H + CBAR_GAP + XLABEL_H) / FIG_H
top_panel = bot_panel + PANEL_H / FIG_H

# Bar panel — left
bar_left = LMARGIN / FIG_W
ax_bar   = fig.add_axes([bar_left, bot_panel, BAR_IN / FIG_W, PANEL_H / FIG_H])

# Heatmap panel — right (square)
heat_left = (LMARGIN + BAR_IN + MID_GAP) / FIG_W
ax_heat   = fig.add_axes([heat_left, bot_panel, HEAT_IN / FIG_W, HEAT_IN / FIG_H])


# ── Panel A: Grouped bar chart ────────────────────────────────────────────────

# Bin labels: one per layer (0 … L-1) + "Never"
bin_labels = [str(i) for i in range(L)] + ["—"]
n_bins     = len(bin_labels)          # L + 1
x          = np.arange(n_bins)

n_tiers = len(tier_labels)
total_w = 0.72                        # fraction of bin width used by all bars
bar_w   = total_w / n_tiers
offsets = np.linspace(-total_w / 2 + bar_w / 2,
                       total_w / 2 - bar_w / 2, n_tiers)

for offset, mask, lbl, color in zip(offsets, tier_masks, tier_labels, TIER_COLORS):
    depths = all_depths[mask]
    # map -1 → sentinel bin L (the "Never" column)
    depths_plot = np.where(depths == -1, L, depths)

    pcts = np.array([
        100.0 * (depths_plot == b).sum() / max(len(depths_plot), 1)
        for b in range(n_bins)
    ])

    ax_bar.bar(x + offset, pcts,
               width=bar_w * 0.88,
               color=color, label=lbl,
               alpha=0.90, zorder=3, linewidth=0)

# Early-zone shading
ax_bar.axvspan(-0.5, EARLY_STABILIZE_LAYER + 0.5,
               color="#6080CC", alpha=0.07, zorder=0)
ax_bar.axvline(EARLY_STABILIZE_LAYER + 0.5,
               color="#6080CC", linewidth=1.0, linestyle="--",
               alpha=0.65, zorder=2)
# Early-zone annotation — place it just inside the right edge of the shaded zone
ax_bar.text(
    EARLY_STABILIZE_LAYER + 0.35,
    ax_bar.get_ylim()[1] * 0.98,
    f"early\n≤ {EARLY_STABILIZE_LAYER}",
    fontsize=tick_fs - 0.5, color="#4a5aaa",
    va="top", ha="right", linespacing=1.3,
)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(bin_labels, fontsize=tick_fs)
ax_bar.set_xlabel("Stabilization layer", fontsize=label_fs, labelpad=3, color="#2a2a44")
ax_bar.set_ylabel("Tokens (%)", fontsize=label_fs, labelpad=3, color="#2a2a44")
ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax_bar.set_xlim(-0.55, n_bins - 0.45)
ax_bar.tick_params(axis="both", labelsize=tick_fs, length=2, width=0.4, pad=1.5)
ax_bar.yaxis.set_tick_params(labelsize=tick_fs)
ax_bar.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.30, zorder=0)
for sp in ["top", "right"]:
    ax_bar.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax_bar.spines[sp].set_linewidth(0.4)
    ax_bar.spines[sp].set_color("#c0c0d0")

ax_bar.set_title("a)  Stabilization depth by loss tier",
                 fontsize=title_fs, fontweight="bold",
                 loc="left", pad=4, color="#12122a")

leg = ax_bar.legend(fontsize=tick_fs, frameon=True, framealpha=0.92,
                    edgecolor="#d0d0e0", loc="upper right",
                    handlelength=1.2, handleheight=0.9,
                    borderpad=0.5, labelspacing=0.35)
leg.get_frame().set_linewidth(0.4)

# Stability criterion annotation (bottom-left)
stabilized_pct = 100.0 - never_pct
stabilized_n   = (all_depths != -1).sum()
ax_bar.annotate(
    f"Criterion: cos sim ≥ {STABILITY_THRESHOLD:.3f}\nto all subsequent layers\n"
    f"Stabilized (at all): {stabilized_n:,} tokens ({stabilized_pct:.1f}%)",
    xy=(0.03, 0.04), xycoords="axes fraction",
    fontsize=tick_fs - 0.8, color="#444460", va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.35", fc="white",
              ec="#c8c8d8", alpha=0.92, lw=0.4),
)


# ── Panel B: Cosine-similarity heatmap (early-stabilizing tokens) ─────────────

lower_tri  = np.tril(np.ones((L, L), dtype=bool), k=-1)
display    = early_sim_mean.copy()
display[lower_tri] = np.nan

mask_ut   = np.triu(np.ones((L, L), dtype=bool))
all_upper = early_sim_mean[mask_ut]
vmin      = max(0.0, float(np.nanmin(all_upper)))
vmax      = 1.0

tick_step = 2 if L > 8 else 1
tick_pos  = list(range(0, L, tick_step))

im = ax_heat.imshow(
    display,
    cmap=CMAP, vmin=vmin, vmax=vmax,
    aspect="equal",
    interpolation="nearest",
    origin="upper",
)

for r in range(L):
    for c in range(r, L):
        v = display[r, c]
        if np.isnan(v):
            continue
        brightness = (v - vmin) / (vmax - vmin + 1e-9)
        tcol  = "white" if brightness > 0.60 else "#12122a"
        label = "1" if v >= 1.0 else f"{v:.2f}"[1:]
        ax_heat.text(c, r, label,
                     ha="center", va="center",
                     fontsize=ann_fs, color=tcol)

ax_heat.set_xticks(tick_pos)
ax_heat.set_yticks(tick_pos)
ax_heat.set_xticklabels(tick_pos, fontsize=tick_fs)
ax_heat.set_yticklabels(tick_pos, fontsize=tick_fs)
ax_heat.tick_params(length=2, width=0.4, pad=1.5)
for sp in ax_heat.spines.values():
    sp.set_linewidth(0.4)
    sp.set_color("#c0c0d0")

ax_heat.set_xlabel("Layer  $\\ell$",
                   fontsize=label_fs, labelpad=3, color="#2a2a44")
ax_heat.set_ylabel("Layer  $\\ell^*$",
                   fontsize=label_fs, labelpad=3, color="#2a2a44")
ax_heat.set_title(
    f"b)  Similarity — early-stabilizing tokens (≤ layer {EARLY_STABILIZE_LAYER})",
    fontsize=title_fs, fontweight="bold",
    loc="left", pad=4, color="#12122a",
)

# Colorbar
cb_bot  = (BOT_PAD + CBAR_LBL) / FIG_H
cb_h    = CBAR_H / FIG_H
cb_left = (LMARGIN + BAR_IN + MID_GAP + 0.05) / FIG_W
cb_w    = (HEAT_IN - 0.10) / FIG_W

cbar_ax = fig.add_axes([cb_left, cb_bot, cb_w, cb_h])
cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cb.set_label("Mean Cosine Similarity",
             fontsize=subt_fs + 0.2, labelpad=3, color="#2a2a44")
cb.ax.tick_params(labelsize=subt_fs - 0.3, length=2, width=0.4, colors="#2a2a44")
cb.outline.set_linewidth(0.35)


# ── Figure title ──────────────────────────────────────────────────────────────

title_y = 1.0 - (SUPTITLE_H * 0.28) / FIG_H
sub_y   = 1.0 - (SUPTITLE_H * 0.28 + SUPTITLE_H * 0.42) / FIG_H

fig.text(0.5, title_y,
         "Token Representation Stabilization Across Transformer Layers",
         ha="center", va="top",
         fontsize=title_fs + 1.0, fontweight="bold", color="#12122a")
fig.text(0.5, sub_y,
         f"Cosine Similarity Threshold = {STABILITY_THRESHOLD:.3f}",
         ha="center", va="top",
         fontsize=subt_fs + 0.4, color="#50507a", style="italic")


# ── Save ─────────────────────────────────────────────────────────────────────

for ext in ("pdf", "png"):
    out = Path(OUTPUT_DIR) / f"stabilization_depth.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
