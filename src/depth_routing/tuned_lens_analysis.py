"""TunedLens depth analysis for Pythia models on WikiText.

Computes per-token TunedLens prediction trajectories (cross-entropy, entropy,
forward KL) across all layers, saves them as .npz, and provides aggregate and
per-sample visualizations.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens
from tuned_lens.plotting import PredictionTrajectory

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "EleutherAI/pythia-160m-deduped"

def load_wikitext_subset(
    split: str = "test",
    num_samples: Optional[int] = None,
    min_length: int = 32,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
) -> list[str]:
    """Load a subset of WikiText, filtering out short/empty paragraphs."""
    ds = load_dataset(dataset_name, dataset_config, split=split)
    texts: list[str] = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_length:
            texts.append(text)
            if num_samples is not None and len(texts) >= num_samples:
                break
    logger.info("Loaded %d samples from %s/%s (split=%s)", len(texts), dataset_name, dataset_config, split)
    return texts


def _process_one_sample(
    tuned_lens: TunedLens,
    model: AutoModelForCausalLM,
    tokenizer,
    text: str,
    idx: int,
    max_seq_len: int,
    include_log_probs: bool,
    device: torch.device,
) -> dict | None:
    """
    Tokenise text and compute a TunedLens trajectory. 
    
    Returns None if too short.
    """
    input_ids_enc = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_seq_len,
    )
    input_ids_list = (
        input_ids_enc.squeeze(0).tolist()
        if isinstance(input_ids_enc, torch.Tensor)
        else input_ids_enc[0]
    )
    if len(input_ids_list) < 2:
        logger.warning("Skipping sample %d: too short after tokenisation.", idx)
        return None

    targets = input_ids_list[1:] + [tokenizer.eos_token_id or 0]

    with torch.no_grad():
        traj = PredictionTrajectory.from_lens_and_model(
            tuned_lens, model,
            tokenizer=tokenizer,
            input_ids=input_ids_list,
            targets=targets,
        )

    # Compute top_token_ids cheaply before the large log_probs array is freed
    entry: dict = {
        "index": idx,
        "input_ids": input_ids_list,
        "token_strings": [tokenizer.decode([tid]) for tid in input_ids_list],
        "cross_entropy": traj.cross_entropy().stats,
        "entropy": traj.entropy().stats,
        "forward_kl": traj.forward_kl().stats,
        "top_token_ids": np.argmax(traj.log_probs, axis=-1).astype(np.int32),
    }
    if include_log_probs:
        entry["log_probs"] = traj.log_probs
    return entry


def compute_tuned_lens_trajectories(
    model_name: str = DEFAULT_MODEL,
    texts: list[str] | None = None,
    split: str = "test",
    num_samples: int = 20,
    max_seq_len: int = 512,
    device: str | None = None,
    include_log_probs: bool = False,
) -> dict:
    """
    Compute per-token TunedLens trajectories on WikiText samples.

    Returns a dict with:
      - model_name, layer_labels
      - trajectories: list of per-sample dicts, each containing
        cross_entropy, entropy, forward_kl as (n_layers+1, seq_len) arrays,
        plus input_ids and token_strings. If include_log_probs is True,
        each dict also contains log_probs as (n_layers+1, seq_len, vocab_size).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=device)
    tuned_lens = tuned_lens.to(device)

    if texts is None:
        texts = load_wikitext_subset(split=split, num_samples=num_samples)

    trajectories: list[dict] = []

    for idx, text in enumerate(tqdm(texts, desc="TunedLens trajectories")):
        entry = _process_one_sample(
            tuned_lens, model, tokenizer, text, idx, max_seq_len, include_log_probs, device,
        )
        if entry is not None:
            trajectories.append(entry)

    if not trajectories:
        raise RuntimeError("No valid samples were processed.")

    n_layers_plus_one = trajectories[0]["cross_entropy"].shape[0]
    layer_labels = [f"layer_{i}" for i in range(n_layers_plus_one - 1)] + ["output"]

    return {
        "model_name": model_name,
        "layer_labels": layer_labels,
        "trajectories": trajectories,
    }


def stream_and_save(
    model_name: str = DEFAULT_MODEL,
    split: str = "test",
    num_samples: int = 1000,
    max_seq_len: int = 512,
    device: str | None = None,
    include_log_probs: bool = False,
    output_dir: str | Path = "results",
    save_every: int = 50,
    resume_from_dir: str | Path | None = None,
) -> list[Path]:
    """
    Process WikiText samples one-by-one, saving to disk on save_every to a numbered
    batch file.  At most save_every trajectory dicts live in memory at once.

    If resume_from_dir is given, existing ``trajectories_batch_*.npz`` files in
    that directory are inspected to determine how many samples (and batches) were
    already processed.  Those samples are skipped and new batch files continue
    the numbering sequence.  When resuming into a different output_dir, the
    existing files are not copied — only the count is used.

    Returns the list of paths of newly written batch files.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    logger.info("Using device: %s", device_obj)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model %s …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device_obj)
    model.eval()
    tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=device_obj).to(device_obj)

    texts = load_wikitext_subset(split=split, num_samples=num_samples)

    # Resume logic: count existing samples in batch files to determine how many to
    # skip and what batch number to start from.
    skip_samples = 0
    batch_num = 0
    resume_dir = Path(resume_from_dir) if resume_from_dir else output_dir
    existing_batches = sorted(resume_dir.glob("trajectories_batch_*.npz"))
    if existing_batches:
        # Count total samples already saved across all existing batch files
        for bf in existing_batches:
            data = np.load(bf, allow_pickle=True)
            skip_samples += len(
                [k for k in data.files if k.startswith("ce_")]
            )
            data.close()
        batch_num = len(existing_batches)
        logger.info(
            "Resuming: found %d existing batch(es) with %d samples in %s — "
            "skipping those samples, continuing from batch %04d",
            len(existing_batches), skip_samples, resume_dir, batch_num,
        )
        texts = texts[skip_samples:]
        if not texts:
            logger.info("All samples already processed — nothing to do.")
            return []

    batch: list[dict] = []
    saved_paths: list[Path] = []
    layer_labels: list[str] | None = None

    for idx, text in enumerate(tqdm(texts, desc="TunedLens (streaming)")):
        global_idx = idx + skip_samples
        entry = _process_one_sample(
            tuned_lens, model, tokenizer, text, global_idx, max_seq_len, include_log_probs, device_obj,
        )
        if entry is None:
            continue

        if layer_labels is None:
            n = entry["cross_entropy"].shape[0]
            layer_labels = [f"layer_{i}" for i in range(n - 1)] + ["output"]

        batch.append(entry)

        if len(batch) >= save_every:
            path = output_dir / f"trajectories_batch_{batch_num:04d}.npz"
            save_trajectories(
                {"model_name": model_name, "layer_labels": layer_labels, "trajectories": batch},
                path,
            )
            logger.info("Batch %04d saved (%d samples, cumulative %d)", batch_num, len(batch), global_idx + 1)
            saved_paths.append(path)
            batch.clear()
            gc.collect()
            batch_num += 1

    if batch:
        path = output_dir / f"trajectories_batch_{batch_num:04d}.npz"
        save_trajectories(
            {"model_name": model_name, "layer_labels": layer_labels or [], "trajectories": batch},
            path,
        )
        logger.info("Final batch %04d saved (%d samples)", batch_num, len(batch))
        saved_paths.append(path)

    logger.info("Streaming complete — %d batch file(s) written to %s", len(saved_paths), output_dir)
    return saved_paths


def aggregate_trajectories(trajectories: list[dict]) -> dict:
    """Compute per-layer mean/std of cross-entropy, perplexity, entropy, forward KL.

    Each trajectory's per-token values are first averaged over the sequence
    dimension, then statistics are taken across samples.

    Returns a dict with mean_* and std_* arrays of shape (n_layers+1,).
    """
    ce_per_layer = np.array([t["cross_entropy"].mean(axis=-1) for t in trajectories])
    ent_per_layer = np.array([t["entropy"].mean(axis=-1) for t in trajectories])
    fkl_per_layer = np.array([t["forward_kl"].mean(axis=-1) for t in trajectories])
    ppl_per_layer = np.exp(ce_per_layer)

    return {
        "mean_cross_entropy": ce_per_layer.mean(axis=0),
        "std_cross_entropy": ce_per_layer.std(axis=0),
        "mean_perplexity": ppl_per_layer.mean(axis=0),
        "std_perplexity": ppl_per_layer.std(axis=0),
        "mean_entropy": ent_per_layer.mean(axis=0),
        "std_entropy": ent_per_layer.std(axis=0),
        "mean_forward_kl": fkl_per_layer.mean(axis=0),
        "std_forward_kl": fkl_per_layer.std(axis=0),
    }


def save_trajectories(results: dict, output_path: str | Path) -> Path:
    """
    Save trajectories to a compressed .npz archive.

    Per sample i: ce_i, ent_i, fkl_i (n_layers+1, seq_len), ids_i, tokens_i.
    Plus metadata: model_name, layer_labels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "model_name": np.array(results["model_name"]),
        "layer_labels": np.array(results["layer_labels"]),
    }
    for traj in results["trajectories"]:
        i = traj["index"]
        arrays[f"ce_{i}"] = np.asarray(traj["cross_entropy"], dtype=np.float16)
        arrays[f"ent_{i}"] = np.asarray(traj["entropy"], dtype=np.float16)
        arrays[f"fkl_{i}"] = np.asarray(traj["forward_kl"], dtype=np.float16)
        arrays[f"ids_{i}"] = np.asarray(traj["input_ids"], dtype=np.int64)
        arrays[f"tokens_{i}"] = np.array(traj["token_strings"], dtype=object)
        arrays[f"top_{i}"] = np.asarray(traj["top_token_ids"], dtype=np.int32)
        if "log_probs" in traj:
            arrays[f"logp_{i}"] = np.asarray(traj["log_probs"], dtype=np.float16)

    np.savez_compressed(output_path, **arrays)
    logger.info("Trajectories saved to %s", output_path)
    return output_path


def load_trajectories(path: str | Path) -> tuple[list[dict], dict]:
    """Load trajectories from a .npz archive.

    Returns (trajectories, meta) where meta has model_name and layer_labels.
    """
    data = np.load(path, allow_pickle=True)
    meta = {
        "model_name": str(data["model_name"]),
        "layer_labels": data["layer_labels"].tolist(),
    }

    sample_indices = sorted(
        {int(k.split("_", 1)[1]) for k in data.files if k.startswith("ce_")}
    )
    trajectories = [
        {
            "index": i,
            "input_ids": data[f"ids_{i}"].tolist(),
            "token_strings": data[f"tokens_{i}"].tolist(),
            "cross_entropy": data[f"ce_{i}"],
            "entropy": data[f"ent_{i}"],
            "forward_kl": data[f"fkl_{i}"],
            "top_token_ids": data[f"top_{i}"],
            **({"log_probs": data[f"logp_{i}"]} if f"logp_{i}" in data.files else {}),
        }
        for i in sample_indices
    ]
    return trajectories, meta


def _resolve_trajectories(source: dict | str | Path) -> tuple[list[dict], list[str], str]:
    """
    Extract (trajectories, layer_labels, model_name) from a results dict, a single
    .npz path, or a directory containing batch files named trajectories_batch_*.npz.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            batch_files = sorted(path.glob("trajectories_batch_*.npz"))
            if not batch_files:
                raise ValueError(f"No batch files found in {path}")
            all_trajectories: list[dict] = []
            layer_labels: list[str] | None = None
            model_name: str | None = None
            for f in batch_files:
                trajs, meta = load_trajectories(f)
                all_trajectories.extend(trajs)
                if layer_labels is None:
                    layer_labels = meta["layer_labels"]
                    model_name = meta["model_name"]
            logger.info("Loaded %d trajectories from %d batch(es) in %s",
                        len(all_trajectories), len(batch_files), path)
            return all_trajectories, layer_labels or [], model_name or ""
        # fall through to single-file handling below
        if path.suffix != ".npz":
            raise ValueError("File-based access requires a .npz file from save_trajectories().")
        trajectories, meta = load_trajectories(path)
        return trajectories, meta["layer_labels"], meta["model_name"]
    return source["trajectories"], source["layer_labels"], source["model_name"]


def _save_and_show(fig: plt.Figure, save_path, show: bool, pdf = False):
    """Optionally save and/or display a figure."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf:
            fig.savefig(save_path, bbox_inches="tight")
        else:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    if show:
        plt.show()


def _format_layer_axis(ax, layer_labels):
    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=60, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3)


def visualize_trajectory_quartiles(
    results: dict | str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Per-token cross-entropy and forward KL across layers, with quartile shading.

    For each layer, values are pooled across all tokens and all samples.
    The median is shown as a solid line; shaded bands show the IQR (25th–75th
    percentile, darker) and the full range (0th–100th, lighter).
    """
    trajectories, layer_labels, model_name = _resolve_trajectories(results)
    x = np.arange(len(layer_labels))

    # Concatenate all token values per layer: (n_layers+1, total_tokens)
    all_ce = np.concatenate([t["cross_entropy"].astype(np.float32) for t in trajectories], axis=1)
    all_fkl = np.concatenate([t["forward_kl"].astype(np.float32) for t in trajectories], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle(f"Per-Token Trajectory Quartiles – {model_name}  ({len(trajectories)} samples)", fontsize=13)

    for ax, data, title, color in [
        (axes[0], all_ce, "Cross-Entropy (nats)", "steelblue"),
        (axes[1], all_fkl, "Forward KL (nats)", "darkorange"),
    ]:
        p0, p25, p50, p75, p100 = np.percentile(data, [0, 25, 50, 75, 100], axis=1)
        ax.fill_between(x, p0, p100, alpha=0.12, color=color, label="min–max")
        ax.fill_between(x, p25, p75, alpha=0.35, color=color, label="IQR")
        ax.plot(x, p50, color=color, linewidth=2, label="median")
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        _format_layer_axis(ax, layer_labels)

    fig.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def visualize_earliest_match_histogram(
    results: dict | str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Histogram of the earliest layer whose top predicted token matches the final layer.

    For every token position in every sample, finds the minimum layer index l
    such that argmax(log_probs[l]) == argmax(log_probs[-1]).  Because the final
    layer always agrees with itself, every position is guaranteed a match.
    """
    trajectories, layer_labels, model_name = _resolve_trajectories(results)

    earliest_layers: list[np.ndarray] = []
    for t in trajectories:
        top = np.asarray(t["top_token_ids"])  # (n_layers+1, seq_len)
        final_top = top[-1:, :]               # (1, seq_len) — broadcasts
        # argmax on a boolean array returns the index of the first True
        earliest = np.argmax(top == final_top, axis=0)  # (seq_len,)
        earliest_layers.append(earliest)

    all_earliest = np.concatenate(earliest_layers)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        all_earliest,
        bins=np.arange(len(layer_labels) + 1) - 0.5,
        edgecolor="black", linewidth=0.4, color="steelblue",
    )
    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Earliest layer matching final-layer top token")
    ax.set_ylabel("Number of token positions")
    ax.set_title(
        f"Earliest Layer / Final-Layer Top-Token Agreement – {model_name}\n"
        f"({len(trajectories)} samples, {len(all_earliest):,} token positions)",
        fontsize=12,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_and_show(fig, save_path, show)
    return fig


def visualize_combined(
    results: dict | str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
    max_display_tokens: int = 16,
    early_layer_threshold: int = 6,
) -> plt.Figure:
    """Combined publication-grade figure with four subplots.

    (a) Token grid for a sequence with early stabilization, coloured by KL.
    (b) Cross-entropy trajectories (mean ± 2σ + early-stabilising subset).
    (c) Forward-KL trajectories (mean ± 2σ + early-stabilising subset).
    (d) Persistent earliest-match histogram.
    """
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec

    trajectories, layer_labels, model_name = _resolve_trajectories(results)
    n_layers = len(layer_labels)
    x = np.arange(n_layers)

    # Load tokenizer to decode top-token IDs for the grid
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Persistent earliest match for every token in every sample ----------
    # For token position p, persistent_earliest[p] is the earliest layer l
    # such that top_token_ids[l] == top_token_ids[l+1] == ... == top_token_ids[-1].
    all_persistent: list[np.ndarray] = []
    for t in trajectories:
        top = np.asarray(t["top_token_ids"])          # (n_layers, seq_len)
        matches = top == top[-1:]                      # broadcast vs final row
        cum = np.cumprod(matches[::-1], axis=0)[::-1]  # AND from final layer
        all_persistent.append(np.argmax(cum, axis=0))  # first True per column

    # --- Subplot (a): pick a sequence with early stabilisation --------------
    best_traj_idx = None
    for ti, t in enumerate(trajectories):
        seq_len = min(max_display_tokens, len(t["input_ids"]))
        if np.any(all_persistent[ti][:seq_len] <= early_layer_threshold):
            best_traj_idx = ti
            break
    if best_traj_idx is None:
        best_score = n_layers
        for ti, t in enumerate(trajectories):
            s = min(max_display_tokens, len(t["input_ids"]))
            m = all_persistent[ti][:s].min()
            if m < best_score:
                best_score, best_traj_idx = m, ti

    traj_a = trajectories[best_traj_idx]
    seq_a = min(max_display_tokens, len(traj_a["input_ids"]))
    top_ids = np.asarray(traj_a["top_token_ids"])[:, :seq_a]
    fkl_grid = np.asarray(traj_a["forward_kl"], dtype=np.float32)[:, :seq_a]

    # Decode predicted tokens at every (layer, position)
    token_grid = np.empty_like(top_ids, dtype=object)
    for l in range(n_layers):
        for s in range(seq_a):
            tok = tokenizer.decode([int(top_ids[l, s])]).replace("\n", "↵")
            token_grid[l, s] = tok[:8] if len(tok) <= 8 else tok[:7] + "…"

    # get per-token trajectories
    all_ce_trajs: list[np.ndarray] = []
    all_fkl_trajs: list[np.ndarray] = []
    for t in trajectories:
        ce = np.asarray(t["cross_entropy"], dtype=np.float32)
        fkl = np.asarray(t["forward_kl"], dtype=np.float32)
        for s in range(ce.shape[1]):
            all_ce_trajs.append(ce[:, s])
            all_fkl_trajs.append(fkl[:, s])
    all_ce = np.array(all_ce_trajs)    # (total_tokens, n_layers)
    all_fkl = np.array(all_fkl_trajs)

    ce_mean, ce_std = all_ce.mean(axis=0), all_ce.std(axis=0)
    fkl_mean, fkl_std = all_fkl.mean(axis=0), all_fkl.std(axis=0)

    # Bottom decile (10%) by layer-1 CE / KL
    l1 = min(1, n_layers - 1)
    n_decile = max(1, len(all_ce) // 10)
    ce_early_idx = np.argsort(all_ce[:, l1])[:n_decile]
    fkl_early_idx = np.argsort(all_fkl[:, l1])[:n_decile]
    ce_early = all_ce[ce_early_idx]
    fkl_early = all_fkl[fkl_early_idx]
    ce_e_mean, ce_e_std = ce_early.mean(axis=0), ce_early.std(axis=0)
    fkl_e_mean, fkl_e_std = fkl_early.mean(axis=0), fkl_early.std(axis=0)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.5, wspace=0.4)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # Bold subplot labels in top-left corner
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        ax.text(-0.04, 1.05, f"{label})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="bottom", ha="right")

    vmax = float(np.percentile(fkl_grid, 95))
    im = ax_a.imshow(
        fkl_grid, aspect="auto", origin="lower",
        cmap="Blues", norm=Normalize(vmin=0, vmax=vmax),
    )
    thresh_75 = float(np.percentile(fkl_grid, 75))
    for l in range(n_layers):
        for s in range(seq_a):
            ax_a.text(
                s, l, token_grid[l, s], ha="center", va="center", fontsize=4.5,
                color="white" if fkl_grid[l, s] > thresh_75 else "black",
            )
    ax_a.set_xticks(np.arange(seq_a))
    ax_a.set_xticklabels(
        [traj_a["token_strings"][i].replace("\n", "↵")[:10] for i in range(seq_a)],
        rotation=45, ha="right", fontsize=6,
    )
    ax_a.set_yticks(np.arange(n_layers))
    ax_a.set_yticklabels(range(n_layers), fontsize=6)
    ax_a.set_xlabel("Input token", fontsize=8)
    ax_a.set_ylabel("Layer", fontsize=8)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_a)
    cax = divider.append_axes("right", size="2%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label("Forward KL (nats)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax_b.fill_between(x, np.maximum(ce_mean - 2 * ce_std, 0), ce_mean + 2 * ce_std,
                       alpha=0.2, color="C0", label=r"all $\pm 2\sigma$")
    ax_b.plot(x, ce_mean, linewidth=1.5, color="C0", label="all mean")
    ax_b.fill_between(x, np.maximum(ce_e_mean - 2 * ce_e_std, 0), ce_e_mean + 2 * ce_e_std,
                       alpha=0.25, color="tab:purple", label=r"P10 $\pm 2\sigma$")
    ax_b.plot(x, ce_e_mean, linewidth=1.5, color="tab:purple", label="P10 mean")
    ax_b.set_xlabel("Layer", fontsize=8)
    ax_b.set_ylabel("Cross-entropy (nats)", fontsize=8)
    ax_b.set_ylim([0, 60])
    ax_b.set_title("Cross-entropy trajectories", fontsize=9)
    ax_b.legend(fontsize=6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(range(n_layers), fontsize=6)
    ax_b.tick_params(axis="y", labelsize=6) 
    ax_b.grid(True, alpha=0.3)

    ax_c.fill_between(x, np.maximum(fkl_mean - 2 * fkl_std, 0), fkl_mean + 2 * fkl_std,
                       alpha=0.2, color="C0", label=r"all $\pm 2\sigma$")
    ax_c.plot(x, fkl_mean, linewidth=1.5, color="C0", label="all mean")
    ax_c.fill_between(x, np.maximum(fkl_e_mean - 2 * fkl_e_std, 0), fkl_e_mean + 2 * fkl_e_std,
                       alpha=0.25, color="tab:purple", label=r"P10 $\pm 2\sigma$")
    ax_c.plot(x, fkl_e_mean, linewidth=1.5, color="tab:purple", label="P10 mean")
    ax_c.set_xlabel("Layer", fontsize=8)
    ax_c.set_ylabel("Forward KL (nats)", fontsize=8)
    ax_c.set_title("Forward KL trajectories", fontsize=9)
    ax_c.legend(fontsize=6)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(range(n_layers), fontsize=6)
    ax_c.tick_params(axis="y", labelsize=6)
    ax_c.grid(True, alpha=0.3)

    # --- (d) Persistent earliest-match histogram ----------------------------
    all_persist = np.concatenate(all_persistent)
    ax_d.hist(all_persist, bins=np.arange(n_layers + 1) - 0.5)
    ax_d.set_xticks(np.arange(n_layers))
    ax_d.set_xticklabels(range(n_layers), fontsize=6)
    ax_d.set_xlabel("Layer", fontsize=8)
    ax_d.set_ylabel("Count", fontsize=8)
    ax_d.set_title("Earliest persistent match", fontsize=9)
    ax_d.tick_params(axis="y", labelsize=6)
    ax_d.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_and_show(fig, save_path, show, pdf=True)
    return fig


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="TunedLens depth analysis on WikiText")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name")
    parser.add_argument("--split", default="test", help="WikiText split")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of WikiText samples")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", default=None, help="Torch device (auto-detected if omitted)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    parser.add_argument("--include-log-probs", action="store_true", help="Save full vocab distribution (large)")
    parser.add_argument(
        "--save-every", type=int, default=50, metavar="N",
        help="Stream mode: flush a batch .npz every N samples to cap RAM usage. 0 = load everything then save once.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip computation; load existing batch files from --output-dir and regenerate plots.",
    )
    parser.add_argument(
        "--resume-from", default=None, metavar="DIR",
        help="Resume streaming: skip samples already saved as batch files in DIR and continue numbering.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    show = not args.no_show

    if args.plot_only:
        logger.info("--plot-only: loading trajectories from %s", output_dir)
    elif args.save_every > 0:
        stream_and_save(
            model_name=args.model,
            split=args.split,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            device=args.device,
            include_log_probs=args.include_log_probs,
            output_dir=output_dir,
            save_every=args.save_every,
            resume_from_dir=args.resume_from,
        )
    else:
        results = compute_tuned_lens_trajectories(
            model_name=args.model,
            split=args.split,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            device=args.device,
            include_log_probs=args.include_log_probs,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = save_trajectories(results, output_dir / "trajectories.npz")
        print(f"Trajectories saved to {npz_path}")

    save_path = output_dir / "tuned_lens_combined.pdf"
    visualize_combined(output_dir, save_path=save_path, show=show)
    print(f"Figure saved to {save_path}")


if __name__ == "__main__":
    main()
