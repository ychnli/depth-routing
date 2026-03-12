"""
Layer-to-layer affine mapping with 5-fold cross-validation.

For tokens categorized as 'by_0' (stabilize at layer 0) and 'never' (never stabilize),
fits per-layer affine maps predicting hidden-state updates Δh_l = h_{l+1} - h_l from h_l.
Uses all available data with 5-fold CV, outputs per-fold metrics to CSV, and produces
an averaged R² / MSE plot.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
DATA_ROOT = Path(
    "/Users/anjali/Library/CloudStorage/GoogleDrive-anjalisr@stanford.edu"
    "/My Drive/depth_routing_data/token_evolution_data"
)
EXCERPT_DIR = DATA_ROOT / "excerpts"
N_LAYERS = 13  # embedding + 12 transformer blocks
N_FOLDS = 5
RNG_SEED = 0
ARTIFACTS_DIR = Path(__file__).parent / "affine_map_artifacts"
CATEGORIES = ["by_0", "never"]

REQUIRED_KEYS = {"hidden_states", "tl_top_token_ids", "token_losses", "input_ids"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ── Data structures ─────────────────────────────────────────────────────────────
@dataclass
class ExcerptBatch:
    path: Path
    hidden_states: np.ndarray   # (N_LAYERS, seq_len, d_model)
    tl_top_token_ids: np.ndarray  # (N_LAYERS, seq_len)
    token_losses: np.ndarray    # (seq_len - 1,)
    input_ids: np.ndarray       # (seq_len,)


@dataclass
class UpdateDataset:
    category: str
    states: np.ndarray          # (n_tokens, N_LAYERS, d_model)
    token_ids: np.ndarray       # (n_tokens,)


@dataclass
class LayerFoldMetrics:
    layer: int
    fold: int
    n_train: int
    n_test: int
    update_mse: float
    update_r2: float
    next_state_mse: float
    coef_fro_norm: float
    intercept_l2_norm: float


# ── Loading & preprocessing ─────────────────────────────────────────────────────
def load_excerpt(path: Path) -> ExcerptBatch:
    data = np.load(path, allow_pickle=True)
    missing = REQUIRED_KEYS - set(data.files)
    if missing:
        raise KeyError(f"{path.name}: missing keys {sorted(missing)}")

    hs = np.asarray(data["hidden_states"], dtype=np.float32)
    top = np.asarray(data["tl_top_token_ids"])
    losses = np.asarray(data["token_losses"], dtype=np.float32)
    ids = np.asarray(data["input_ids"])

    seq_len = hs.shape[1]
    assert top.shape == (N_LAYERS, seq_len), f"{path.name}: tl shape {top.shape}"
    assert ids.shape == (seq_len,)
    assert losses.shape == (seq_len - 1,)
    assert np.all(np.isfinite(hs)) and np.all(np.isfinite(losses))

    return ExcerptBatch(path=path, hidden_states=hs, tl_top_token_ids=top,
                        token_losses=losses, input_ids=ids)


def compute_stabilization_layer(top_ids: np.ndarray) -> np.ndarray:
    """Per-token earliest layer where top-1 matches final layer thereafter."""
    matches_final = top_ids == top_ids[-1:]
    cumulative = np.cumprod(matches_final[::-1], axis=0)[::-1]
    return np.argmax(cumulative, axis=0)


def get_category_mask(stabilization_layer: np.ndarray, category: str) -> np.ndarray:
    if category == "by_0":
        return stabilization_layer == 0
    elif category == "never":
        return stabilization_layer == (N_LAYERS - 1)
    else:
        raise ValueError(f"Unknown category: {category}")


# ── Collect all tokens for a category ───────────────────────────────────────────
def collect_all_tokens(category: str) -> UpdateDataset:
    """Load every excerpt file and collect all tokens matching the category."""
    excerpt_files = sorted(EXCERPT_DIR.glob("excerpt_*.npz"))
    if not excerpt_files:
        raise FileNotFoundError(f"No excerpt_*.npz files in {EXCERPT_DIR}")

    all_states, all_token_ids = [], []
    print(f"[collect] category={category!r} | scanning {len(excerpt_files)} files")

    for i, fpath in enumerate(excerpt_files, 1):
        if i % 100 == 0:
            print(f"  processed {i}/{len(excerpt_files)} files, {len(all_states):,} tokens so far")

        batch = load_excerpt(fpath)
        # Drop last position to align with next-token target
        top_work = batch.tl_top_token_ids[:, :-1]
        st_layer = compute_stabilization_layer(top_work)
        mask = get_category_mask(st_layer, category)
        if not np.any(mask):
            continue

        # states: (seq_len-1, N_LAYERS, d_model)
        states = np.transpose(batch.hidden_states[:, :-1, :], (1, 0, 2))
        token_ids = batch.input_ids[1:]

        idx = np.flatnonzero(mask)
        all_states.append(states[idx])
        all_token_ids.append(token_ids[idx])

    states_arr = np.concatenate(all_states).astype(np.float32)
    token_ids_arr = np.concatenate(all_token_ids)
    print(f"[collect] {category}: {len(states_arr):,} tokens, shape={states_arr.shape}")

    return UpdateDataset(category=category, states=states_arr, token_ids=token_ids_arr)


# ── Cross-validated fitting ─────────────────────────────────────────────────────
def make_fold_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) for each fold."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        test_idx = perm[start : start + size]
        train_idx = np.concatenate([perm[:start], perm[start + size :]])
        folds.append((train_idx, test_idx))
        start += size
    return folds


def fit_cv(dataset: UpdateDataset) -> list[LayerFoldMetrics]:
    """5-fold CV: fit affine maps per layer, return per-fold metrics."""
    folds = make_fold_indices(len(dataset.states), N_FOLDS, RNG_SEED)
    all_metrics: list[LayerFoldMetrics] = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_states = dataset.states[train_idx]
        test_states = dataset.states[test_idx]
        print(f"[fit] {dataset.category} fold {fold_i}: "
              f"train={len(train_idx):,}  test={len(test_idx):,}")

        for layer in range(N_LAYERS - 1):
            X_train = train_states[:, layer, :]
            Y_train = train_states[:, layer + 1, :] - X_train

            X_test = test_states[:, layer, :]
            Y_test = test_states[:, layer + 1, :] - X_test

            model = LinearRegression().fit(X_train, Y_train)
            delta_pred = model.predict(X_test)
            next_pred = X_test + delta_pred
            next_true = X_test + Y_test

            all_metrics.append(LayerFoldMetrics(
                layer=layer,
                fold=fold_i,
                n_train=len(X_train),
                n_test=len(X_test),
                update_mse=mean_squared_error(Y_test, delta_pred),
                update_r2=r2_score(Y_test, delta_pred),
                next_state_mse=mean_squared_error(next_true, next_pred),
                coef_fro_norm=float(np.linalg.norm(model.coef_)),
                intercept_l2_norm=float(np.linalg.norm(model.intercept_)),
            ))

    return all_metrics


# ── Output ──────────────────────────────────────────────────────────────────────
METRIC_FIELDS = [
    "layer", "fold", "n_train", "n_test",
    "update_mse", "update_r2", "next_state_mse",
    "coef_fro_norm", "intercept_l2_norm",
]


def write_metrics_csv(metrics: list[LayerFoldMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for m in metrics:
            writer.writerow({k: getattr(m, k) for k in METRIC_FIELDS})
    print(f"[write] {path}")


def aggregate_metrics(metrics: list[LayerFoldMetrics]) -> dict[str, np.ndarray]:
    """Compute per-layer mean and std across folds."""
    layers = np.arange(N_LAYERS - 1)
    r2_all = np.zeros((N_FOLDS, len(layers)))
    mse_all = np.zeros((N_FOLDS, len(layers)))

    for m in metrics:
        r2_all[m.fold, m.layer] = m.update_r2
        mse_all[m.fold, m.layer] = m.update_mse

    return {
        "layers": layers,
        "r2_mean": r2_all.mean(axis=0),
        "r2_std": r2_all.std(axis=0),
        "mse_mean": mse_all.mean(axis=0),
        "mse_std": mse_all.std(axis=0),
    }


# ── Plotting ────────────────────────────────────────────────────────────────────
# Palette: teal for easy, coral for hard
COLOR_BY0 = "#2a9d8f"
COLOR_NEVER = "#e76f51"


def plot_comparison(by0_agg: dict, never_agg: dict, save_path: Path) -> None:
    layers = by0_agg["layers"]
    xtick_labels = [f"{l}→{l+1}" for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    # Top: R²
    ax = axes[0]
    ax.plot(layers, by0_agg["r2_mean"], "o-", color=COLOR_BY0, lw=2, label="by_0 (easy)")
    ax.fill_between(layers,
                    by0_agg["r2_mean"] - by0_agg["r2_std"],
                    by0_agg["r2_mean"] + by0_agg["r2_std"],
                    color=COLOR_BY0, alpha=0.15)
    ax.plot(layers, never_agg["r2_mean"], "s-", color=COLOR_NEVER, lw=2, label="never (hard)")
    ax.fill_between(layers,
                    never_agg["r2_mean"] - never_agg["r2_std"],
                    never_agg["r2_mean"] + never_agg["r2_std"],
                    color=COLOR_NEVER, alpha=0.15)
    ax.axhline(0, ls="--", lw=0.8, color="grey", alpha=0.6)
    ax.set_ylabel("Update $R^2$")
    ax.set_title("Affine update prediction by layer (5-fold CV)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # Bottom: MSE
    ax = axes[1]
    ax.plot(layers, by0_agg["mse_mean"], "o-", color=COLOR_BY0, lw=2, label="by_0 (easy)")
    ax.fill_between(layers,
                    by0_agg["mse_mean"] - by0_agg["mse_std"],
                    by0_agg["mse_mean"] + by0_agg["mse_std"],
                    color=COLOR_BY0, alpha=0.15)
    ax.plot(layers, never_agg["mse_mean"], "s-", color=COLOR_NEVER, lw=2, label="never (hard)")
    ax.fill_between(layers,
                    never_agg["mse_mean"] - never_agg["mse_std"],
                    never_agg["mse_mean"] + never_agg["mse_std"],
                    color=COLOR_NEVER, alpha=0.15)
    ax.set_xlabel("Layer transition $l \\to l+1$")
    ax.set_ylabel("Update MSE")
    ax.grid(alpha=0.2)

    axes[1].set_xticks(layers)
    axes[1].set_xticklabels(xtick_labels, rotation=45, ha="right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[plot] {save_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    results = {}
    for cat in CATEGORIES:
        dataset = collect_all_tokens(cat)
        metrics = fit_cv(dataset)
        csv_path = ARTIFACTS_DIR / cat / "cv_metrics.csv"
        write_metrics_csv(metrics, csv_path)
        results[cat] = aggregate_metrics(metrics)

    plot_comparison(
        results["by_0"],
        results["never"],
        ARTIFACTS_DIR / "by0_vs_never_cv.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
