"""
Figure 1b: Layer-wise cosine similarity heatmap for top 15% easiest tokens (lowest loss).
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Main
# ---------------------------
def plot_easy_tokens_heatmap():
    print("Loading CSVs...")

    token_df = pd.read_csv(os.path.join(DATA_DIR, "token_level.csv"))

    # ---------------------------
    # Remove padding tokens (loss=0)
    # ---------------------------
    real_tokens = token_df[token_df["loss"] > 0].copy()

    # ---------------------------
    # Keep only top 5% easiest tokens (bottom 15th percentile of loss)
    # ---------------------------
    loss_threshold = real_tokens["loss"].quantile(0.05)
    easy_tokens = real_tokens[real_tokens["loss"] <= loss_threshold].copy()

    print(f"\nLoss threshold (15th percentile): {loss_threshold:.4f}")
    print(f"Easy token count: {len(easy_tokens)} / {len(real_tokens)} total ({100 * len(easy_tokens) / len(real_tokens):.1f}%)\n")

    # ---------------------------
    # Accumulators
    # ---------------------------
    sim_sum = np.zeros((NUM_LAYERS, NUM_LAYERS))
    sim_n   = 0

    excerpt_ids = easy_tokens["excerpt_id"].unique()
    print(f"Processing {len(excerpt_ids)} excerpts...\n")

    # ---------------------------
    # Loop over excerpts
    # ---------------------------
    for i, excerpt_id in enumerate(excerpt_ids):

        npy_path = os.path.join(DATA_DIR, f"hidden_states_excerpt_{excerpt_id}.npy")
        if not os.path.exists(npy_path):
            continue

        hidden = np.load(npy_path)  # (L, seq_len, D)

        excerpt_easy = easy_tokens[easy_tokens["excerpt_id"] == excerpt_id]
        if excerpt_easy.empty:
            continue

        positions = excerpt_easy["token_idx"].values
        if len(positions) == 0:
            continue

        # Extract representations
        reps = hidden[:, positions, :]  # (L, N, D)

        # Numerically stable normalization
        abs_max = np.abs(reps).max(axis=-1, keepdims=True)
        abs_max = np.maximum(abs_max, 1e-8)
        reps_scaled = reps / abs_max

        norms = np.linalg.norm(reps_scaled, axis=-1, keepdims=True)
        reps_norm = reps_scaled / np.maximum(norms, 1e-8)  # (L, N, D)

        # Vectorized cosine similarity: (L, L, N)
        cos = np.einsum("lnd,mnd->lmn", reps_norm, reps_norm)
        cos = np.clip(cos, -1.0, 1.0)

        sim_sum += cos.sum(axis=-1)
        sim_n   += cos.shape[-1]

        if i % 50 == 0:
            print(f"Processed {i}/{len(excerpt_ids)} excerpts")

    # ---------------------------
    # Compute mean
    # ---------------------------
    sim_mean = sim_sum / max(sim_n, 1)
    np.fill_diagonal(sim_mean, 1.0)

    # ---------------------------
    # Sanity check
    # ---------------------------
    diag = np.diag(sim_mean)
    print(f"\nSanity check: mean diag = {diag.mean():.6f}")
    upper = sim_mean[np.triu_indices(NUM_LAYERS, k=1)]
    print(f"Off-diagonal: min={upper.min():.4f}  max={upper.max():.4f}  mean={upper.mean():.4f}")

    # ---------------------------
    # Save matrix
    # ---------------------------
    np.save(os.path.join(OUTPUT_DIR, "sim_matrix_top15_easy.npy"), sim_mean)

    # ---------------------------
    # Mask lower triangle
    # ---------------------------
    mask = np.tril(np.ones_like(sim_mean, dtype=bool), k=-1)
    sim_mean_masked = np.ma.masked_where(mask, sim_mean)

    # ---------------------------
    # Plot
    # ---------------------------
    print("\nPlotting...")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11
    })

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    vmin = float(np.nanmin(sim_mean[np.triu_indices(NUM_LAYERS, k=0)]))
    vmax = 1.0
    print(f"Colormap range: vmin={vmin:.4f}, vmax={vmax:.4f}")

    im = ax.imshow(
        sim_mean_masked,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        interpolation="nearest"
    )

    # Annotate cells
    for row in range(NUM_LAYERS):
        for col in range(NUM_LAYERS):
            if col < row:
                continue
            val = sim_mean[row, col]
            text_color = "white" if val < (vmin + (vmax - vmin) * 0.6) else "black"
            ax.text(
                col, row, f"{val:.2f}",
                ha="center", va="center",
                fontsize=6, color=text_color
            )

    ax.set_xticks(range(NUM_LAYERS))
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LAYER_LABELS)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Top 15% Easiest Tokens (Lowest Loss)")

    ax.set_xticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, NUM_LAYERS, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity")

    fig.suptitle(
        "Layer-wise Token Representation Similarity\nTop 15% Easiest Tokens",
        fontsize=14,
        fontweight="bold"
    )

    out_path = os.path.join(OUTPUT_DIR, "figure_1b_easy15_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.show()


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    plot_easy_tokens_heatmap()