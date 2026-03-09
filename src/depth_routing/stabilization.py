"""
Token stabilization criteria for TunedLens trajectories.

Provides functions to identify tokens that "stabilize early" according to
three complementary definitions:

1. **Cross-entropy**: final-layer CE is below a threshold.
2. **Forward KL**: final-layer KL divergence (relative to the *model* output)
   drops below a threshold by a given layer.
3. **Persistent top-1 match**: the argmax prediction at layer *l* agrees with
   the final-layer argmax for every subsequent layer *l, l+1, …, L*.

Each function operates on the per-excerpt trajectory dicts produced by
:func:`depth_routing.token_evolution_data_collect.load_tuned_lens_from_excerpts`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TokenMask:
    """Boolean mask over (excerpt, position) pairs.

    Attributes
    ----------
    excerpt_ids : ndarray, shape (N,)
        Excerpt index for each token (indexes into the trajectory list).
    positions : ndarray, shape (N,)
        Within-sequence position for each token.
    mask : ndarray, shape (N,)
        True if the token satisfies the stabilization criterion.
    """

    excerpt_ids: np.ndarray
    positions: np.ndarray
    mask: np.ndarray

    @property
    def indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (excerpt_ids, positions) for tokens that pass the filter."""
        return self.excerpt_ids[self.mask], self.positions[self.mask]

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    @property
    def total(self) -> int:
        return len(self.mask)

    @property
    def fraction(self) -> float:
        return self.count / self.total if self.total else 0.0

    def __repr__(self) -> str:
        return f"TokenMask({self.count}/{self.total} = {self.fraction:.1%})"


def _build_flat_arrays(
    trajectories: list[dict],
    drop_last: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return flat (excerpt_id, position) arrays over all tokens.

    Parameters
    ----------
    drop_last : bool
        If True, exclude the last token of each sequence (often a newline
        whose CE target is EOS, causing artificial spikes).

    Returns
    -------
    excerpt_ids, positions : 1-D int arrays of the same length
    n_total : total number of tokens
    """
    eids, poss = [], []
    for ti, t in enumerate(trajectories):
        seq_len = np.asarray(t["cross_entropy"]).shape[1]
        end = seq_len - 1 if drop_last else seq_len
        poss.append(np.arange(end))
        eids.append(np.full(end, ti, dtype=np.intp))
    return np.concatenate(eids), np.concatenate(poss), len(np.concatenate(poss))


def stable_by_cross_entropy(
    trajectories: list[dict],
    *,
    layer: int = -1,
    threshold: float = 2.0,
    drop_last: bool = True,
) -> TokenMask:
    """Tokens whose cross-entropy at *layer* is below *threshold*.

    Parameters
    ----------
    layer : int
        Layer index (negative indexing supported; -1 = final layer).
    threshold : float
        Maximum CE (nats) to be considered "stable".
    """
    eids, poss, _ = _build_flat_arrays(trajectories, drop_last=drop_last)
    vals = np.concatenate([
        np.asarray(t["cross_entropy"], dtype=np.float32)[layer, :(-1 if drop_last else None)]
        for t in trajectories
    ])
    return TokenMask(eids, poss, vals < threshold)


def stable_by_forward_kl(
    trajectories: list[dict],
    *,
    layer: int = -1,
    threshold: float = 0.5,
    drop_last: bool = True,
) -> TokenMask:
    """Tokens whose forward KL at *layer* is below *threshold*.

    Forward KL measures divergence from the final-layer distribution, so
    small values mean the lens prediction at *layer* already approximates
    the model output well.

    Parameters
    ----------
    layer : int
        Layer to evaluate (negative indexing supported).
    threshold : float
        Maximum KL (nats).
    """
    eids, poss, _ = _build_flat_arrays(trajectories, drop_last=drop_last)
    vals = np.concatenate([
        np.asarray(t["forward_kl"], dtype=np.float32)[layer, :(-1 if drop_last else None)]
        for t in trajectories
    ])
    return TokenMask(eids, poss, vals < threshold)


def persistent_top1_layer(
    trajectories: list[dict],
    *,
    drop_last: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the earliest persistent top-1 match layer for every token.

    For token position *p*, persistent_layer[p] is the smallest layer index
    *l* such that ``argmax(logits[l]) == argmax(logits[l+1]) == … ==
    argmax(logits[-1])``.  Since the final layer trivially matches itself,
    every token gets a value in ``[0, n_layers]``.

    Returns
    -------
    excerpt_ids : ndarray, shape (N,)
    positions : ndarray, shape (N,)
    persistent_layers : ndarray, shape (N,)
        The earliest layer at which the top-1 prediction matches all
        subsequent layers through the final layer.
    """
    eids, poss, _ = _build_flat_arrays(trajectories, drop_last=drop_last)
    parts = []
    for t in trajectories:
        top = np.asarray(t["top_token_ids"])  # (n_layers+1, seq_len)
        end = top.shape[1] - 1 if drop_last else top.shape[1]
        matches = top[:, :end] == top[-1:, :end]
        cum = np.cumprod(matches[::-1], axis=0)[::-1]
        parts.append(np.argmax(cum, axis=0))
    return eids, poss, np.concatenate(parts)


def stable_by_persistent_top1(
    trajectories: list[dict],
    *,
    by_layer: int | None = None,
    drop_last: bool = True,
) -> TokenMask:
    """Tokens whose top-1 prediction persistently matches the final layer.

    Parameters
    ----------
    by_layer : int or None
        If given, selects tokens that stabilize *at or before* this layer.
        If None, selects tokens that stabilize at *any* layer before the
        final one (i.e. ``persistent_layer < n_layers``).
    """
    eids, poss, player = persistent_top1_layer(
        trajectories, drop_last=drop_last,
    )
    n_layers_plus_one = np.asarray(trajectories[0]["top_token_ids"]).shape[0]
    if by_layer is not None:
        mask = player <= by_layer
    else:
        mask = player < n_layers_plus_one - 1
    return TokenMask(eids, poss, mask)
