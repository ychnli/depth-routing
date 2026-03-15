"""
Layer-to-layer affine mapping with bootstrap uncertainty estimation.

For 'by_0' (stabilize at layer 0) and 'never' (never stabilize) tokens, fits
per-layer affine maps predicting Δh_l = h_{l+1} - h_l from h_l, using B=200
bootstrap replicates with out-of-bag evaluation to produce 95% confidence bands.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(
    "/Users/anjali/Library/CloudStorage/GoogleDrive-anjalisr@stanford.edu"
    "/My Drive/depth_routing_data/token_evolution_data"
)
EXCERPT_DIR = DATA_ROOT / "excerpts"
N_LAYERS = 13          # embedding + 12 transformer blocks
N_TRANSITIONS = N_LAYERS - 1
N_BOOT = 200           # bootstrap replicates
RNG_SEED = 42
ARTIFACTS_DIR = Path(__file__).parent / "affine_map_artifacts"
CATEGORIES = ["by_0", "never"]

REQUIRED_KEYS = {"hidden_states", "tl_top_token_ids", "token_losses", "input_ids"}


# ── Data structures ─────────────────────────────────────────────────────────────
@dataclass
class UpdateDataset:
    category: str
    states: np.ndarray     # (n_tokens, N_LAYERS, d_model)
    token_ids: np.ndarray  # (n_tokens,)


@dataclass
class BootstrapLayerMetrics:
    layer: int
    replicate: int
    n_train: int
    n_oob: int
    update_mse: float
    update_r2: float


# ── Data loading ────────────────────────────────────────────────────────────────
def load_excerpt(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    missing = REQUIRED_KEYS - set(data.files)
    if missing:
        raise KeyError(f"{path.name}: missing {sorted(missing)}")

    hs = np.asarray(data["hidden_states"], dtype=np.float32)
    top = np.asarray(data["tl_top_token_ids"])
    ids = np.asarray(data["input_ids"])

    seq_len = hs.shape[1]
    assert hs.shape[0] == N_LAYERS
    assert top.shape == (N_LAYERS, seq_len)
    assert ids.shape == (seq_len,)
    assert np.all(np.isfinite(hs))

    return {"hidden_states": hs, "tl_top_token_ids": top, "input_ids": ids}


def compute_stabilization_layer(top_ids: np.ndarray) -> np.ndarray:
    """Per-token earliest layer where top-1 matches final layer thereafter."""
    matches_final = top_ids == top_ids[-1:]
    return np.argmax(np.cumprod(matches_final[::-1], axis=0)[::-1], axis=0)


def get_category_mask(stab_layer: np.ndarray, category: str) -> np.ndarray:
    if category == "by_0":
        return stab_layer == 0
    elif category == "never":
        return stab_layer == (N_LAYERS - 1)
    raise ValueError(f"Unknown category: {category}")


def collect_all_tokens(category: str) -> UpdateDataset:
    """Scan all excerpt files and collect every token matching the category."""
    excerpt_files = sorted(EXCERPT_DIR.glob("excerpt_*.npz"))
    if not excerpt_files:
        raise FileNotFoundError(f"No excerpt_*.npz in {EXCERPT_DIR}")

    chunks_states, chunks_ids = [], []
    n_tokens = 0
    print(f"[collect] category={category!r} | {len(excerpt_files)} files")

    for i, fpath in enumerate(excerpt_files, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(excerpt_files)} files, {n_tokens:,} tokens")

        ex = load_excerpt(fpath)
        # Drop last position to align hidden states with next-token targets
        top_work = ex["tl_top_token_ids"][:, :-1]
        mask = get_category_mask(compute_stabilization_layer(top_work), category)
        if not mask.any():
            continue

        # (seq_len-1, N_LAYERS, d_model)
        states = np.transpose(ex["hidden_states"][:, :-1, :], (1, 0, 2))
        idx = np.flatnonzero(mask)
        chunks_states.append(states[idx])
        chunks_ids.append(ex["input_ids"][1:][idx])
        n_tokens += len(idx)

    states_arr = np.concatenate(chunks_states).astype(np.float32)
    ids_arr = np.concatenate(chunks_ids)
    assert states_arr.shape == (n_tokens, N_LAYERS, states_arr.shape[2])
    print(f"[collect] {category}: {n_tokens:,} tokens, d_model={states_arr.shape[2]}")
    return UpdateDataset(category=category, states=states_arr, token_ids=ids_arr)


# ── Class balancing ────────────────────────────────────────────────────────────
def subsample_dataset(dataset: UpdateDataset, n: int, seed: int) -> UpdateDataset:
    """Randomly sample exactly n tokens from dataset (without replacement)."""
    if n > len(dataset.states):
        raise ValueError(f"Requested {n} samples but dataset only has {len(dataset.states)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset.states), size=n, replace=False)
    return UpdateDataset(
        category=dataset.category,
        states=dataset.states[idx],
        token_ids=dataset.token_ids[idx],
    )


# ── Bootstrap fitting ───────────────────────────────────────────────────────────
def fit_bootstrap(dataset: UpdateDataset) -> list[BootstrapLayerMetrics]:
    """Run N_BOOT bootstrap replicates, evaluate on out-of-bag tokens."""
    n = len(dataset.states)
    rng = np.random.default_rng(RNG_SEED)
    all_metrics: list[BootstrapLayerMetrics] = []

    print(f"[boot] {dataset.category}: {n:,} tokens, {N_BOOT} replicates")

    for b in range(N_BOOT):
        if (b + 1) % 10 == 0:
            print(f"  replicate {b + 1}/{N_BOOT}")

        # Resample with replacement; OOB = tokens never drawn
        boot_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = np.flatnonzero(oob_mask)

        if len(oob_idx) < 50:
            print(f"  WARNING: replicate {b} has only {len(oob_idx)} OOB tokens, skipping")
            continue

        train_states = dataset.states[boot_idx]   # includes duplicates
        test_states = dataset.states[oob_idx]

        for layer in range(N_TRANSITIONS):
            X_tr = train_states[:, layer, :]
            Y_tr = train_states[:, layer + 1, :] - X_tr
            X_te = test_states[:, layer, :]
            Y_te = test_states[:, layer + 1, :] - X_te

            model = LinearRegression().fit(X_tr, Y_tr)
            delta_pred = model.predict(X_te)

            all_metrics.append(BootstrapLayerMetrics(
                layer=layer,
                replicate=b,
                n_train=len(boot_idx),
                n_oob=len(oob_idx),
                update_mse=mean_squared_error(Y_te, delta_pred),
                update_r2=r2_score(Y_te, delta_pred),
            ))

    print(f"[boot] {dataset.category}: {len(all_metrics)} total metric rows")
    return all_metrics


# ── Output ──────────────────────────────────────────────────────────────────────
CSV_FIELDS = ["layer", "replicate", "n_train", "n_oob", "update_mse", "update_r2"]


def write_csv(metrics: list[BootstrapLayerMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for m in metrics:
            w.writerow({k: getattr(m, k) for k in CSV_FIELDS})
    print(f"[write] {path}")


def aggregate(metrics: list[BootstrapLayerMetrics]) -> dict[str, np.ndarray]:
    """Per-layer median and 95% CI from bootstrap replicates."""
    # Gather per-(replicate, layer) into arrays
    replicates = sorted(set(m.replicate for m in metrics))
    n_rep = len(replicates)
    rep_to_row = {r: i for i, r in enumerate(replicates)}

    r2_arr = np.full((n_rep, N_TRANSITIONS), np.nan)
    mse_arr = np.full((n_rep, N_TRANSITIONS), np.nan)

    for m in metrics:
        row = rep_to_row[m.replicate]
        r2_arr[row, m.layer] = m.update_r2
        mse_arr[row, m.layer] = m.update_mse

    # Sanity: every cell should be filled
    assert not np.any(np.isnan(r2_arr)), "Missing replicate×layer entries"

    layers = np.arange(N_TRANSITIONS)
    return {
        "layers": layers,
        "r2_median": np.median(r2_arr, axis=0),
        "r2_lo": np.percentile(r2_arr, 2.5, axis=0),
        "r2_hi": np.percentile(r2_arr, 97.5, axis=0),
        "mse_median": np.median(mse_arr, axis=0),
        "mse_lo": np.percentile(mse_arr, 2.5, axis=0),
        "mse_hi": np.percentile(mse_arr, 97.5, axis=0),
    }


# ── Plotting ────────────────────────────────────────────────────────────────────
COLOR_BY0 = "tab:blue"  
COLOR_NEVER = "tab:red"


def plot_bootstrap(by0: dict, never: dict, save_path: Path) -> None:
    layers = by0["layers"]
    xticks = [f"{l}→{l+1}" for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    # ── R² ──
    ax = axes[0]
    ax.plot(layers, by0["r2_median"], "o-", color=COLOR_BY0, lw=1.8,
            ms=5, label="easy")
    ax.fill_between(layers, by0["r2_lo"], by0["r2_hi"],
                    color=COLOR_BY0, alpha=0.18)
    ax.plot(layers, never["r2_median"], "s-", color=COLOR_NEVER, lw=1.8,
            ms=5, label="hard")
    ax.fill_between(layers, never["r2_lo"], never["r2_hi"],
                    color=COLOR_NEVER, alpha=0.18)
    ax.axhline(0, ls="--", lw=0.7, color="grey", alpha=0.5)
    ax.set_ylabel("Update $R^2$")
    ax.set_title(f"Affine update prediction by layer ({N_BOOT} bootstrap replicates)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    # ── MSE ──
    ax = axes[1]
    ax.plot(layers, by0["mse_median"], "o-", color=COLOR_BY0, lw=1.8,
            ms=5, label="easy")
    ax.fill_between(layers, by0["mse_lo"], by0["mse_hi"],
                    color=COLOR_BY0, alpha=0.18)
    ax.plot(layers, never["mse_median"], "s-", color=COLOR_NEVER, lw=1.8,
            ms=5, label="hard")
    ax.fill_between(layers, never["mse_lo"], never["mse_hi"],
                    color=COLOR_NEVER, alpha=0.18)
    ax.set_ylabel("Update MSE")
    ax.set_xlabel("Layer transition $l \\to l+1$")
    ax.grid(alpha=0.2)

    axes[1].set_xticks(layers)
    axes[1].set_xticklabels(xticks, rotation=45, ha="right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[plot] {save_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Collect full datasets first so we know the minority class size
    datasets = {cat: collect_all_tokens(cat) for cat in CATEGORIES}

    # Balance: subsample majority class to match minority class
    n_min = min(len(ds.states) for ds in datasets.values())
    minority = min(datasets, key=lambda c: len(datasets[c].states))
    print(f"[balance] minority class is '{minority}' with {n_min:,} tokens — "
          f"subsampling all categories to {n_min:,}")
    datasets = {cat: subsample_dataset(ds, n_min, RNG_SEED) for cat, ds in datasets.items()}

    results = {}
    for cat in CATEGORIES:
        dataset = datasets[cat]
        metrics = fit_bootstrap(dataset)
        write_csv(metrics, ARTIFACTS_DIR / cat / "bootstrap_metrics.csv")
        results[cat] = aggregate(metrics)

    plot_bootstrap(
        results["by_0"],
        results["never"],
        ARTIFACTS_DIR / "by0_vs_never_bootstrap.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()
