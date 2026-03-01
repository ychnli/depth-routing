"""
Figure 1a: Layer-wise cosine similarity heatmaps stratified by token loss.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Config
# ---------------------------
DATA_DIR   = "token_evolution_data"
OUTPUT_DIR = "token_evolution_figures"
NUM_LAYERS = 13

LAYER_LABELS = ["Emb"] + [str(i) for i in range(1, NUM_LAYERS)]
TIERS = ["easy", "medium", "hard"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Main
# ---------------------------
def plot_representation_similarity_heatmap():
    print("Loading CSVs...")

    token_df = pd.read_csv(os.path.join(DATA_DIR, "token_level.csv"))

    # ---------------------------
    # Remove padding tokens (loss=0)
    # ---------------------------
    real_tokens = token_df[token_df["loss"] > 0].copy()

    # ---------------------------
    # Assign global terciles
    # ---------------------------
    real_tokens["loss_tercile"] = pd.qcut(
        real_tokens["loss"],
        q=3,
        labels=TIERS
    )
    

    print(f"\nToken counts per tier:\n{real_tokens['loss_tercile'].value_counts()}\n")

    # ---------------------------
    # Accumulators
    # ---------------------------
    sim_sum = {tier: np.zeros((NUM_LAYERS, NUM_LAYERS)) for tier in TIERS}
    sim_n   = {tier: 0 for tier in TIERS}

    excerpt_ids = real_tokens["excerpt_id"].unique()
    print(f"Processing {len(excerpt_ids)} excerpts...\n")

    # ---------------------------
    # Loop over excerpts
    # ---------------------------
    for i, excerpt_id in enumerate(excerpt_ids):

        npy_path = os.path.join(DATA_DIR, f"hidden_states_excerpt_{excerpt_id}.npy")
        if not os.path.exists(npy_path):
            continue

        hidden = np.load(npy_path)  # (L, seq_len, D)

        excerpt_tokens = real_tokens[real_tokens["excerpt_id"] == excerpt_id]

        for tier in TIERS:

            tier_tokens = excerpt_tokens[excerpt_tokens["loss_tercile"] == tier]
            if tier_tokens.empty:
                continue

            positions = tier_tokens["token_idx"].values
            if len(positions) == 0:
                continue

            # Extract representations
            reps = hidden[:, positions, :]  # (L, N, D)

            # Numerically stable normalization — prevents overflow in norm computation
            abs_max = np.abs(reps).max(axis=-1, keepdims=True)
            abs_max = np.maximum(abs_max, 1e-8)
            reps_scaled = reps / abs_max

            norms = np.linalg.norm(reps_scaled, axis=-1, keepdims=True)
            reps_norm = reps_scaled / np.maximum(norms, 1e-8)  # (L, N, D), unit vectors

            # Vectorized cosine similarity: (L, L, N)
            cos = np.einsum("lnd,mnd->lmn", reps_norm, reps_norm)
            cos = np.clip(cos, -1.0, 1.0)

            sim_sum[tier] += cos.sum(axis=-1)
            sim_n[tier]   += cos.shape[-1]

        if i % 50 == 0:
            print(f"Processed {i}/{len(excerpt_ids)} excerpts")

    # ---------------------------
    # Compute mean
    # ---------------------------
    sim_mean = {
        tier: sim_sum[tier] / max(sim_n[tier], 1)
        for tier in TIERS
    }

    # Enforce perfect diagonal
    for tier in TIERS:
        np.fill_diagonal(sim_mean[tier], 1.0)

    # ---------------------------
    # Mask lower triangle
    # ---------------------------
    sim_mean_masked = {}
    for tier in TIERS:
        mat = sim_mean[tier].copy()
        mask = np.tril(np.ones_like(mat, dtype=bool), k=-1)
        sim_mean_masked[tier] = np.ma.masked_where(mask, mat)

    # ---------------------------
    # Sanity check
    # ---------------------------
    print("\nSanity check (diagonal should be 1.0):")
    for tier in TIERS:
        diag = np.diag(sim_mean[tier])
        print(f"  {tier:6s}: mean diag = {diag.mean():.6f}  min = {diag.min():.6f}")

    print("\nOff-diagonal similarity ranges (upper triangle):")
    for tier in TIERS:
        upper = sim_mean[tier][np.triu_indices(NUM_LAYERS, k=1)]
        print(f"  {tier:6s}: min={upper.min():.4f}  max={upper.max():.4f}  mean={upper.mean():.4f}")

    # ---------------------------
    # Save matrices
    # ---------------------------
    for tier in TIERS:
        np.save(
            os.path.join(OUTPUT_DIR, f"sim_matrix_{tier}.npy"),
            sim_mean[tier]
        )

    # # ---------------------------
    # # Plot
    # # ---------------------------
    # print("\nPlotting Figure 1a...")

    # plt.rcParams.update({
    #     "font.size": 10,
    #     "axes.titlesize": 12,
    #     "axes.labelsize": 11
    # })

    # fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    # tier_titles = {
    #     "easy":   "Easy Tokens\n(Low Loss)",
    #     "medium": "Medium Tokens",
    #     "hard":   "Hard Tokens\n(High Loss)"
    # }

    # # vmin from upper-triangle values only (avoids masked-array .min() issues)
    # all_upper = np.concatenate([
    #     sim_mean[t][np.triu_indices(NUM_LAYERS, k=0)]
    #     for t in TIERS
    # ])
    # vmin = float(np.nanmin(all_upper))
    # vmax = 1.0

    # print(f"\nColormap range: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # for ax, tier in zip(axes, TIERS):

    #     mat = sim_mean_masked[tier]

    #     im = ax.imshow(
    #         mat,
    #         vmin=vmin,
    #         vmax=vmax,
    #         cmap="viridis",
    #         interpolation="nearest"
    #     )

    #     # ---------------------------
    #     # Annotate cells with similarity values
    #     # ---------------------------
    #     for row in range(NUM_LAYERS):
    #         for col in range(NUM_LAYERS):
    #             if col < row:  # skip masked lower triangle
    #                 continue
    #             val = sim_mean[tier][row, col]
    #             text_color = "white" if val < (vmin + (vmax - vmin) * 0.6) else "black"
    #             ax.text(
    #                 col, row, f"{val:.2f}",
    #                 ha="center", va="center",
    #                 fontsize=6, color=text_color,
    #                 fontweight="normal"
    #             )

    #     ax.set_xticks(range(NUM_LAYERS))
    #     ax.set_yticks(range(NUM_LAYERS))
    #     ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right")
    #     ax.set_yticklabels(LAYER_LABELS)

    #     ax.set_xlabel("Layer")
    #     ax.set_ylabel("Layer")
    #     ax.set_title(tier_titles[tier])

    #     # thin grid lines
    #     ax.set_xticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
    #     ax.set_yticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
    #     ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.3)
    #     ax.tick_params(which="minor", bottom=False, left=False)

    # # shared colorbar
    # cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    # cbar.set_label("Cosine Similarity")

    # fig.suptitle(
    #     "Layer-wise Token Representation Similarity\nStratified by Prediction Difficulty",
    #     fontsize=14,
    #     fontweight="bold"
    # )

    # out_path = os.path.join(OUTPUT_DIR, "figure_1a_cosine_heatmaps.png")
    # plt.savefig(out_path, dpi=300, bbox_inches="tight")
    # print(f"\nSaved to {out_path}")
    # plt.show()

    # # ---------------------------
    # # Plot (paper-quality)
    # # ---------------------------
    # print("\nPlotting Figure 1a...")

    # plt.rcParams.update({
    #     "font.size": 16,
    #     "axes.titlesize": 20,
    #     "axes.labelsize": 16,
    #     "xtick.labelsize": 14,
    #     "ytick.labelsize": 14,
    #     "font.family": "sans-serif"
    # })

    # fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

    # tier_titles = {
    #     "easy":   "Easy Tokens (Low Loss)",
    #     "medium": "Medium Tokens",
    #     "hard":   "Hard Tokens (High Loss)"
    # }

    # # vmin from upper triangle
    # all_upper = np.concatenate([
    #     sim_mean[t][np.triu_indices(NUM_LAYERS, k=0)]
    #     for t in TIERS
    # ])
    # vmin = float(np.nanmin(all_upper))
    # vmax = 1.0

    # print(f"\nColormap range: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # for ax, tier in zip(axes, TIERS):

    #     mat = sim_mean_masked[tier]

    #     im = ax.imshow(
    #         mat,
    #         vmin=vmin,
    #         vmax=vmax,
    #         cmap="Blues",  # cleaner, professional
    #         interpolation="nearest"
    #     )

    #     # ---------------------------
    #     # Annotate cells
    #     # ---------------------------
    #     for row in range(NUM_LAYERS):
    #         for col in range(NUM_LAYERS):
    #             if col < row:
    #                 continue
    #             val = sim_mean[tier][row, col]

    #             # adaptive text color
    #             threshold = vmin + (vmax - vmin) * 0.6
    #             text_color = "white" if val < threshold else "black"

    #             ax.text(
    #                 col, row, f"{val:.2f}",
    #                 ha="center", va="center",
    #                 fontsize=9,  # bigger for readability
    #                 color=text_color
    #             )

    #     # Axis labels
    #     ax.set_xticks(range(NUM_LAYERS))
    #     ax.set_yticks(range(NUM_LAYERS))
    #     ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right")
    #     ax.set_yticklabels(LAYER_LABELS)

    #     ax.set_xlabel("Layer", labelpad=10)
    #     ax.set_ylabel("Layer", labelpad=10)
    #     ax.set_title(tier_titles[tier], pad=12)

    #     # Clean grid (very subtle)
    #     ax.set_xticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
    #     ax.set_yticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
    #     ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.15)
    #     ax.tick_params(which="minor", bottom=False, left=False)

    # # Shared colorbar
    # cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    # cbar.set_label("Cosine Similarity", fontsize=14)
    # cbar.ax.tick_params(labelsize=12)

    # # Title
    # fig.suptitle(
    #     "Layer-wise Token Representation Similarity",
    #     fontsize=24,
    #     fontweight="bold",
    #     y=1.04  # <-- add this (adjust between 1.02–1.08)
    # )

    # out_path = os.path.join(OUTPUT_DIR, "NEW_figure_1a_cosine_heatmaps.png")
    # plt.savefig(out_path, dpi=400, bbox_inches="tight")
    # print(f"\nSaved to {out_path}")
    # plt.show()

    # ---------------------------
    # Plot (paper-quality)
    # ---------------------------
    print("\nPlotting Figure 1a...")

    plt.rcParams.update({
        "font.size":       26,
        "axes.titlesize":  20,
        "axes.labelsize":  20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "font.family":     "DejaVu Sans",
    })

    fig, axes = plt.subplots(1, 3, figsize=(28, 9), constrained_layout=True)

    tier_titles = {
        "easy":   "Easy Tokens (Low Loss)",
        "medium": "Medium Tokens",
        "hard":   "Hard Tokens (High Loss)"
    }

    # vmin from upper triangle
    all_upper = np.concatenate([
        sim_mean[t][np.triu_indices(NUM_LAYERS, k=0)]
        for t in TIERS
    ])
    vmin = float(np.nanmin(all_upper))
    vmax = 1.0

    print(f"\nColormap range: vmin={vmin:.4f}, vmax={vmax:.4f}")

    for ax, tier in zip(axes, TIERS):

        mat = sim_mean_masked[tier]

        im = ax.imshow(
            mat,
            vmin=vmin,
            vmax=vmax,
            cmap="Blues",
            interpolation="nearest"
        )

        # ---------------------------
        # Annotate cells
        # ---------------------------
        for row in range(NUM_LAYERS):
            for col in range(NUM_LAYERS):
                if col < row:
                    continue
                val = sim_mean[tier][row, col]
                threshold = vmin + (vmax - vmin) * 0.6
                text_color = "white" if val < threshold else "black"
                ax.text(
                    col, row, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=14,
                    color=text_color
                )

        ax.set_xticks(range(NUM_LAYERS))
        ax.set_yticks(range(NUM_LAYERS))
        ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LAYER_LABELS)

        ax.set_xlabel("Layer", labelpad=12)
        ax.set_ylabel("Layer", labelpad=12)
        ax.set_title(tier_titles[tier], pad=14, fontweight="bold")

        ax.set_xticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
        ax.set_yticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.15)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.suptitle(
        "Layer-wise Token Representation Similarity",
        fontsize=28,
        fontweight="bold",
        y=1.05
    )

    out_path = os.path.join(OUTPUT_DIR, "NEW_figure_1a_cosine_heatmaps.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.show()

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    plot_representation_similarity_heatmap()