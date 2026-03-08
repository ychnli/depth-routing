"""
tuned_lens_data_utils.py
────────────────────────────────────────────────────────────────────────────────
Shared data-loading and binning utilities for TunedLens plotting scripts.

Binning strategy — "TunedLens stabilization"
─────────────────────────────────────────────
For each token position j we look at tl_top_token_ids[:, j] — the argmax
prediction at every layer — and find the earliest layer from which the
prediction is constant all the way to the final layer.

  stabilization_layer[j] = min { k : top[k, j] == top[k+1, j] == ... == top[L-1, j] }
  If no such k exists (prediction never stabilizes), set to L (sentinel).

Bins (default thresholds: early=2, late=6, for a 12-layer model):
  Easy   — stabilization_layer <= early_thresh   (locked in by layer 2)
  Medium — early_thresh < stabilization_layer <= late_thresh
  Hard   — stabilization_layer > late_thresh     (still changing after layer 6)

These thresholds are parametric so both scripts can accept --early-thresh /
--late-thresh CLI arguments and stay in sync.
"""

from pathlib import Path
import numpy as np


def load_and_pool(data_dir: Path, verbose: bool = True) -> dict:
    """
    Load all excerpt_*.npz files and pool arrays across excerpts.

    Returns
    -------
    dict with keys:
        token_losses      (N,)           float32
        tl_top_token_ids  (N, L)         int32
        tl_cossim_hidden  (N, L, L)      float32
        tl_forward_kl     (N, L)         float32
        tl_adjacent_kl    (N, L-1)       float32
        n_excerpts        int
        L                 int
    """
    excerpt_files = sorted((data_dir / "excerpts").glob("excerpt_*.npz"))
    if not excerpt_files:
        raise FileNotFoundError(
            f"No excerpt_*.npz files found in {data_dir / 'excerpts'}"
        )
    if verbose:
        print(f"Found {len(excerpt_files)} excerpt files in {data_dir / 'excerpts'}")

    all_losses   = []
    all_top_ids  = []
    all_hidden   = []
    all_fkl      = []
    all_adj_kl   = []

    for path in excerpt_files:
        d = np.load(path, allow_pickle=True)
        # token_losses:     (seq_len-1,)
        # tl_top_token_ids: (L,   seq_len-1) → transpose to (seq_len-1, L)
        # tl_cossim_hidden: (seq_len-1, L, L)
        # tl_forward_kl:    (L,   seq_len-1) → transpose to (seq_len-1, L)
        # tl_adjacent_kl:   (L-1, seq_len-1) → transpose to (seq_len-1, L-1)
        all_losses.append( d["token_losses"].astype(np.float32))
        all_top_ids.append(d["tl_top_token_ids"].T.astype(np.int32))
        all_hidden.append( d["tl_cossim_hidden"].astype(np.float32))
        all_fkl.append(    d["tl_forward_kl"].T.astype(np.float32))
        all_adj_kl.append( d["tl_adjacent_kl"].T.astype(np.float32))
        d.close()

    token_losses     = np.concatenate(all_losses,   axis=0)   # (N,)
    tl_top_token_ids = np.concatenate(all_top_ids,  axis=0)   # (N, L)
    tl_cossim_hidden = np.concatenate(all_hidden,   axis=0)   # (N, L, L)
    tl_forward_kl    = np.concatenate(all_fkl,      axis=0)   # (N, L)
    tl_adjacent_kl   = np.concatenate(all_adj_kl,   axis=0)   # (N, L-1)

    N, L = tl_top_token_ids.shape
    if verbose:
        print(f"  Total tokens: {N:,}   Layers: {L}")

    return {
        "token_losses":      token_losses,
        "tl_top_token_ids":  tl_top_token_ids,
        "tl_cossim_hidden":  tl_cossim_hidden,
        "tl_forward_kl":     tl_forward_kl,
        "tl_adjacent_kl":    tl_adjacent_kl,
        "n_excerpts":        len(excerpt_files),
        "L":                 L,
    }


def compute_stabilization_bins(
    tl_top_token_ids: np.ndarray,
    early_thresh: int = 2,
    late_thresh: int = 6,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """
    Compute per-token stabilization layer and bin into Easy / Medium / Hard.

    stabilization_layer[j] = the earliest layer k such that the top-1 token
    is constant for all layers k, k+1, ..., L-1.  If the prediction never
    stabilizes, returns L (a sentinel meaning "never").

    Parameters
    ----------
    tl_top_token_ids : (N, L) int32
    early_thresh     : tokens stabilizing at or before this layer → Easy
    late_thresh      : tokens stabilizing after early_thresh but at or before
                       this layer → Medium; beyond → Hard

    Returns
    -------
    stab_layer  : (N,) int   — stabilization layer per token
    masks       : list of 3 boolean arrays (N,) — Easy, Medium, Hard
    bin_labels  : list of 3 strings
    counts      : list of 3 ints
    """
    N, L = tl_top_token_ids.shape

    # For each token j, find the earliest layer k from which top-1 is constant.
    # Work backwards: start at L-1 (always "stable" trivially), extend left as
    # long as top[k-1] == top[k].
    stab_layer = np.full(N, L, dtype=np.int32)   # sentinel: never stabilizes

    # The final layer is always the reference; check from L-2 down to 0.
    # stable_so_far[j] = True if top[k, j] == top[L-1, j] for all layers seen so far
    final_top       = tl_top_token_ids[:, L - 1]   # (N,)
    stable_so_far   = np.ones(N, dtype=bool)         # all stable at layer L-1

    for k in range(L - 1, -1, -1):
        stable_so_far &= (tl_top_token_ids[:, k] == final_top)
        stab_layer[stable_so_far] = k

    masks = [
        stab_layer <= early_thresh,
        (stab_layer > early_thresh) & (stab_layer <= late_thresh),
        stab_layer > late_thresh,
    ]
    bin_labels = ["Easy", "Medium", "Hard"]
    counts     = [int(m.sum()) for m in masks]

    if verbose:
        print(f"  Stabilization thresholds: early≤{early_thresh}, "
              f"medium≤{late_thresh}, hard>{late_thresh}")
        for lbl, cnt, m in zip(bin_labels, counts, masks):
            pct = 100.0 * cnt / N
            med = int(np.median(stab_layer[m])) if cnt > 0 else -1
            print(f"    {lbl:8s}: {cnt:7,} tokens ({pct:.1f}%)  "
                  f"median stab layer = {med}")

    return stab_layer, masks, bin_labels, counts