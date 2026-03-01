
"""
Token Stabilization Depth Analysis

For each token, finds the earliest layer l* such that cosine similarity
between layer l* and ALL subsequent layers exceeds a threshold (i.e., the
representation has stopped changing meaningfully).

Plots:
  Panel A: Stabilization depth distribution stratified by loss tier (bar chart)
  Panel B: Cosine similarity heatmap for early-stabilizing tokens only
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_DIR              = "token_evolution_data"
OUTPUT_DIR            = "token_evolution_figures"
NUM_LAYERS            = 13
FINAL_LAYER           = 11
STABILITY_THRESHOLD   = 0.90
EARLY_STABILIZE_LAYER = 4
TIERS                 = ["easy", "medium", "hard"]
TIER_LABELS           = {
    "easy":   "Easy (Low Loss)",
    "medium": "Medium",
    "hard":   "Hard (High Loss)",
}
TIER_COLORS = {
    "easy":   "#4C9BE8",
    "medium": "#F5A623",
    "hard":   "#E05C5C",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Step 1: Load and tier tokens
# ------------------------------------------------------------------
def load_tokens():
    print("Loading token data...")
    token_df = pd.read_csv(os.path.join(DATA_DIR, "token_level.csv"))
    real = token_df[token_df["loss"] > 0].copy()
    real["loss_tercile"] = pd.qcut(real["loss"], q=3, labels=TIERS)
    print(f"Total real tokens: {len(real):,}")
    print(real["loss_tercile"].value_counts().to_string())
    return real


# ------------------------------------------------------------------
# Step 2: Per-token stabilization depth
# ------------------------------------------------------------------
def get_stabilization_depth(hidden, positions):
    """
    For each token at `positions`, find earliest layer l such that
    cos_sim(rep[l], rep[m]) >= STABILITY_THRESHOLD for ALL m in (l, FINAL_LAYER].

    Returns int array of shape (N,); -1 = never stabilized.
    """
    reps = hidden[:FINAL_LAYER + 1, positions, :]   # (L', N, D)

    abs_max = np.abs(reps).max(axis=-1, keepdims=True)
    reps_scaled = reps / np.maximum(abs_max, 1e-8)
    norms = np.linalg.norm(reps_scaled, axis=-1, keepdims=True)
    reps_norm = reps_scaled / np.maximum(norms, 1e-8)   # (L', N, D)

    L = reps_norm.shape[0]
    N = reps_norm.shape[1]

    cos = np.einsum("lnd,mnd->lmn", reps_norm, reps_norm)
    cos = np.clip(cos, -1.0, 1.0)

    stab_depths = np.full(N, -1, dtype=int)

    for l in range(L - 1):
        above = cos[l, l + 1:, :]
        stable = (above >= STABILITY_THRESHOLD).all(axis=0)
        unassigned = stab_depths == -1
        stab_depths[unassigned & stable] = l

    return stab_depths


# ------------------------------------------------------------------
# Step 3: Collect stabilization data + early-token heatmap
# ------------------------------------------------------------------
def collect_stability_data(real_tokens):
    print(f"\nComputing stabilization depths (threshold = {STABILITY_THRESHOLD})...")

    records = []
    early_sim_sum = np.zeros((NUM_LAYERS, NUM_LAYERS))
    early_sim_n   = 0

    excerpt_ids = real_tokens["excerpt_id"].unique()
    print(f"Processing {len(excerpt_ids)} excerpts...\n")

    for i, excerpt_id in enumerate(excerpt_ids):
        npy_path = os.path.join(DATA_DIR, f"hidden_states_excerpt_{excerpt_id}.npy")
        if not os.path.exists(npy_path):
            continue

        hidden = np.load(npy_path)
        excerpt_tokens = real_tokens[real_tokens["excerpt_id"] == excerpt_id]

        for tier in TIERS:
            tier_tokens = excerpt_tokens[excerpt_tokens["loss_tercile"] == tier]
            if tier_tokens.empty:
                continue

            positions = tier_tokens["token_idx"].values
            losses    = tier_tokens["loss"].values
            depths    = get_stabilization_depth(hidden, positions)

            for pos, depth, loss in zip(positions, depths, losses):
                records.append({
                    "excerpt_id": excerpt_id,
                    "token_idx":  pos,
                    "tier":       tier,
                    "loss":       loss,
                    "stab_depth": depth,
                })

        excerpt_recs = [r for r in records if r["excerpt_id"] == excerpt_id]
        early_pos = np.array([
            r["token_idx"] for r in excerpt_recs
            if 0 <= r["stab_depth"] <= EARLY_STABILIZE_LAYER
        ])

        if len(early_pos) > 0:
            reps = hidden[:, early_pos, :]
            abs_max = np.abs(reps).max(axis=-1, keepdims=True)
            reps_s  = reps / np.maximum(abs_max, 1e-8)
            norms   = np.linalg.norm(reps_s, axis=-1, keepdims=True)
            rn      = reps_s / np.maximum(norms, 1e-8)
            cos     = np.einsum("lnd,mnd->lmn", rn, rn)
            cos     = np.clip(cos, -1.0, 1.0)
            early_sim_sum += cos.sum(axis=-1)
            early_sim_n   += cos.shape[-1]

        if i % 50 == 0:
            print(f"  {i}/{len(excerpt_ids)} excerpts processed")

    df = pd.DataFrame(records)

    early_sim_mean = early_sim_sum / max(early_sim_n, 1)
    np.fill_diagonal(early_sim_mean, 1.0)

    print(f"\nStabilization depth summary:")
    print(df.groupby("tier")["stab_depth"].describe().round(2).to_string())
    never_pct = (df["stab_depth"] == -1).mean() * 100
    print(f"\nNever stabilized: {never_pct:.1f}% of tokens")

    # save early_sim_mean for replotting without recomputing
    np.save(os.path.join(OUTPUT_DIR, "early_sim_mean.npy"), early_sim_mean)

    return df, early_sim_mean


# ------------------------------------------------------------------
# Step 4: Plot
# ------------------------------------------------------------------
def make_figure(df, early_sim_mean):
    print("\nRendering figure...")

    LAYER_LABELS = ["Emb"] + [str(i) for i in range(1, NUM_LAYERS)]

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         18,
        "axes.titlesize":    20,
        "axes.labelsize":    18,
        "xtick.labelsize":   16,
        "ytick.labelsize":   16,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    # fig = plt.figure(figsize=(26, 9), facecolor="white")
    fig = plt.figure(figsize=(6, ), facecolor="white")

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.07, right=0.97,
        top=0.87, bottom=0.16,
        wspace=0.38,
        width_ratios=[1.25, 1],
    )
    ax_bar  = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # ----------------------------------------------------------------
    # Panel a — Bar chart
    # ----------------------------------------------------------------
    bin_labels = LAYER_LABELS[:FINAL_LAYER + 1] + ["Never"]
    n_bins     = len(bin_labels)
    x          = np.arange(n_bins)
    bar_w      = 0.22
    offsets    = [-bar_w, 0, bar_w]

    for offset, tier in zip(offsets, TIERS):
        tier_df = df[df["tier"] == tier].copy()
        depths  = tier_df["stab_depth"].replace(-1, FINAL_LAYER + 1)

        counts = [(depths == l).sum() for l in range(FINAL_LAYER + 1)]
        counts.append((depths == FINAL_LAYER + 1).sum())

        total = sum(counts)
        pcts  = [100 * c / max(total, 1) for c in counts]

        ax_bar.bar(
            x + offset, pcts,
            width=bar_w * 0.87,
            color=TIER_COLORS[tier],
            label=TIER_LABELS[tier],
            alpha=0.88,
            zorder=3,
            linewidth=0,
        )

    # early-zone shading
    ax_bar.axvspan(-0.5, EARLY_STABILIZE_LAYER + 0.5,
                   color=TIER_COLORS["easy"], alpha=0.07, zorder=0)
    ax_bar.axvline(EARLY_STABILIZE_LAYER + 0.5,
                   color=TIER_COLORS["easy"], linewidth=1.8,
                   linestyle="--", alpha=0.7, zorder=2)
    ax_bar.text(
        EARLY_STABILIZE_LAYER + 0.65,
        ax_bar.get_ylim()[1] * 0.99,
        f"early zone (≤ layer {EARLY_STABILIZE_LAYER})",
        fontsize=16, color=TIER_COLORS["easy"], va="top", alpha=0.9,
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bin_labels)
    ax_bar.set_xlabel("Stabilization Layer  $l^*$", labelpad=12)
    ax_bar.set_ylabel("Percentage of Tokens (%)", labelpad=12)
    ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_bar.set_xlim(-0.6, n_bins - 0.4)
    ax_bar.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)
    ax_bar.set_title(
        "a)  Stabilization Depth by Prediction Difficulty",
        fontweight="bold", loc="left", pad=14,
    )
    ax_bar.legend(fontsize=16, frameon=True, framealpha=0.9,
                  edgecolor="#DDDDDD", loc="upper right")
    ax_bar.annotate(
        f"Stability criterion:\ncosine sim ≥ {STABILITY_THRESHOLD:.2f} to all higher layers",
        xy=(0.015, 0.97), xycoords="axes fraction",
        fontsize=15, color="#444444", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#CCCCCC", alpha=0.95),
    )

    # ----------------------------------------------------------------
    # Panel b — Heatmap
    # ----------------------------------------------------------------
    mask       = np.tril(np.ones_like(early_sim_mean, dtype=bool), k=-1)
    sim_masked = np.ma.masked_where(mask, early_sim_mean)

    all_upper = early_sim_mean[np.triu_indices(NUM_LAYERS, k=0)]
    vmin = float(np.nanmin(all_upper))
    vmax = 1.0

    im = ax_heat.imshow(
        sim_masked,
        vmin=vmin, vmax=vmax,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
    )

    for row in range(NUM_LAYERS):
        for col in range(NUM_LAYERS):
            if col < row:
                continue
            val = early_sim_mean[row, col]
            text_color = "white" if val < (vmin + (vmax - vmin) * 0.6) else "black"
            ax_heat.text(
                col, row, f"{val:.2f}",
                ha="center", va="center",
                fontsize=13, color=text_color,
            )

    ax_heat.set_xticks(range(NUM_LAYERS))
    ax_heat.set_yticks(range(NUM_LAYERS))
    ax_heat.set_xticklabels(LAYER_LABELS, rotation=45, ha="right")
    ax_heat.set_yticklabels(LAYER_LABELS)
    ax_heat.set_xlabel("Layer", labelpad=12)
    ax_heat.set_ylabel("Layer", labelpad=12)
    ax_heat.set_title(
        f"b)  Similarity — Early-Stabilizing Tokens (≤ Layer {EARLY_STABILIZE_LAYER})",
        fontweight="bold", loc="left", pad=14,
    )

    ax_heat.set_xticks(np.arange(-0.5, NUM_LAYERS, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, NUM_LAYERS, 1), minor=True)
    ax_heat.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.15)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.spines[:].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.044, pad=0.03)
    cbar.set_label("Cosine Similarity", fontsize=22)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_visible(False)

    # ----------------------------------------------------------------
    # Suptitle
    # ----------------------------------------------------------------
    fig.suptitle(
        "Token Representation Stabilization Across Transformer Layers",
        fontsize=28, fontweight="bold", y=1.02,
    )

    out_path = os.path.join(OUTPUT_DIR, "figure_2_stabilization.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {out_path}")
    plt.show()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    # -- Run full pipeline --
    # real_tokens = load_tokens()
    # df, early_sim_mean = collect_stability_data(real_tokens)
    # df.to_csv(os.path.join(OUTPUT_DIR, "token_stabilization_depths.csv"), index=False)
    # print(f"Stabilization CSV saved → {OUTPUT_DIR}/token_stabilization_depths.csv")
    # make_figure(df, early_sim_mean)

    # -- Plot from saved CSV --
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "token_stabilization_depths.csv"))
    print(f"Stabilized:   {len(df[df['stab_depth'] != -1]):,}")
    print(f"Total tokens: {len(df):,}")

    early_sim_path = os.path.join(OUTPUT_DIR, "early_sim_mean.npy")
    if os.path.exists(early_sim_path):
        early_sim_mean = np.load(early_sim_path)
        make_figure(df, early_sim_mean)
    else:
        print("early_sim_mean.npy not found — rerun collect_stability_data() to generate it.")