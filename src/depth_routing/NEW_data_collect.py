"""
token_evolution_data_collect.py
────────────────────────────────────────────────────────────────────────────────
Runs inference on a transformer language model over text excerpts, collecting
per-token loss and TunedLens intermediate representations at every layer.

HOW TUNED LENS OUTPUTS ARE COMPUTED
─────────────────────────────────────
From reading the TunedLens source directly:

    # TunedLens.forward (lenses.py):
    def forward(self, h, idx):
        h = self.transform_hidden(h, idx)
        return self.unembed.forward(h)

    # TunedLens.transform_hidden (lenses.py):
    def transform_hidden(self, h, idx):
        return h + self[idx](h)        # residual affine via nn.Linear

    # Unembed.forward (unembed.py):
    def forward(self, h):
        return self.unembedding(self.final_norm(h))

    # Unembed stores DEEP COPIES of the model's final norm and unembedding:
    #   self.final_norm  = copy.deepcopy(model's final LayerNorm)
    #   self.unembedding = copy.deepcopy(model's unembedding Linear)

So the full pipeline for layer l is:

    h_translated = h + tuned_lens[l](h)                        # residual affine
    h_ln         = tuned_lens.unembed.final_norm(h_translated) # layer norm
    logits       = tuned_lens.unembed.unembedding(h_ln)        # unembed

For the final layer (l = L-1), transform_hidden returns h unchanged (no
translator for the last layer), and unembed applies norm + unembedding as usual.

We call tuned_lens.transform_hidden() and tuned_lens.unembed.final_norm() and
tuned_lens.unembed.unembedding() directly, intercepting h_ln between the norm
and the unembedding call.  This is equivalent to tuned_lens.forward() but gives
us access to h_ln for downstream cosine-similarity analysis.  No hooks, no numpy
reimplementation, no risk of discrepancy.

SANITY CHECK
─────────────
After computing tl_data for each excerpt, we verify (on the first N excerpts):
  1. Our logits match tuned_lens.forward() output exactly (100% top-token)
  2. Our h_ln matches re-computed tuned_lens.unembed.final_norm(h_translated)
  3. Cosine-sim matrices are re-derivable from stored proj_ln_states

WHAT IS SAVED
──────────────
Per excerpt  (excerpts/excerpt_NNNN.npz):

  Always:
    input_ids           (seq_len,)                  int64
    token_strings       (seq_len,)                  object
    token_losses        (seq_len-1,)                float32
    pred_token_ids      (seq_len-1,)                int64
    model_name                                      str
    avg_loss                                        float32
    perplexity                                      float32
    forward_pass_ms                                 float32

  With --enable-tuned-lens:
    layer_labels        (L,)                        object
    tl_cross_entropy    (L, seq_len-1)              float16
    tl_entropy          (L, seq_len-1)              float16
    tl_forward_kl       (L, seq_len-1)              float16
    tl_top_token_ids    (L, seq_len-1)              int32
    tl_proj_ln_states   (L, seq_len-1, hidden_dim)  float16
    tl_cossim_hidden    (seq_len-1, L, L)           float16
    tl_cossim_vocab     (seq_len-1, L, L)           float16

  With --save-hidden-states:
    hidden_states       (L, seq_len, hidden_dim)    float16

  Summary:
    excerpt_level.csv   one row per excerpt with scalar metrics
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME    = "EleutherAI/pythia-160m-deduped"
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
PROMPT_LEN    = 64
MIN_TOTAL_LEN = 100
BATCH_SIZE    = 4
OUTPUT_DIR    = "token_evolution_data"
MAX_SEQ_LEN   = 512

EXCERPT_FIELDS = [
    "excerpt_id", "prompt", "reference", "generated",
    "avg_loss", "perplexity", "bertscore",
    "forward_pass_ms", "generation_ms", "tokens_generated", "tokens_per_sec",
]

_EXCERPTS_SUBDIR = "excerpts"


# ─────────────────────────────────────────────────────────────────────────────
# §1.  Text quality filters
# ─────────────────────────────────────────────────────────────────────────────

def is_valid(text: str) -> bool:
    """Heuristic quality filter.  Rejects headings and numeric/punct-heavy text."""
    text = text.strip()
    if not text or text.startswith("="):
        return False
    tokens = text.split()
    if len(tokens) < 50:
        return False
    if not tokens[0]:
        return False
    punct_ratio = (text.count(",") + text.count(";")) / len(tokens)
    if punct_ratio > 0.15:
        return False
    num_ratio = sum(1 for w in tokens if any(c.isdigit() for c in w)) / len(tokens)
    if num_ratio > 0.20:
        return False
    cap_ratio = sum(1 for w in tokens if w and w[0].isupper()) / len(tokens)
    if cap_ratio > 0.40:
        return False
    return True


def passes_length_filter(text: str, tokenizer, min_tokens: int = MIN_TOTAL_LEN) -> bool:
    """Return True if the tokenized text meets the minimum token count."""
    return len(tokenizer(text)["input_ids"]) >= min_tokens


# ─────────────────────────────────────────────────────────────────────────────
# §2.  Cosine-similarity helper
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the (L, L) cosine-similarity matrix over the rows of X.

    Parameters
    ----------
    X : (L, d) float32

    Returns
    -------
    (L, L) float16  — arithmetic in float32, stored as float16.
    """
    norms = np.linalg.norm(X, axis=-1, keepdims=True)   # (L, 1)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Xn    = X / norms
    return np.clip(Xn @ Xn.T, -1.0, 1.0).astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# §3.  TunedLens data computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_tuned_lens_data(
    tuned_lens,
    hidden_states: tuple[torch.Tensor, ...],
    model_logits: torch.Tensor,
    targets: list[int],
) -> dict[str, np.ndarray]:
    """
    Compute TunedLens projected hidden states and derived metrics for one excerpt.

    For each layer l, the pipeline (verified from TunedLens source) is:

        h_translated = tuned_lens.transform_hidden(h, l)
                     = h + tuned_lens[l](h)                        # residual
        h_ln         = tuned_lens.unembed.final_norm(h_translated) # layer norm
        logits       = tuned_lens.unembed.unembedding(h_ln)        # unembed

    We call these three steps explicitly so we can capture h_ln, which is
    identical to what tuned_lens.forward(h, l) computes internally.

    Parameters
    ----------
    tuned_lens    : fitted TunedLens instance
    hidden_states : tuple of (1, seq_len, hidden_dim) tensors, one per layer
    model_logits  : (1, seq_len, vocab_size) — final model logits
    targets       : length-seq_len list of next-token target IDs

    Returns
    -------
    dict with numpy arrays:
        cross_entropy   (L, seq_len)              float16
        entropy         (L, seq_len)              float16
        forward_kl      (L, seq_len)              float16
        top_token_ids   (L, seq_len)              int32
        proj_ln_states  (L, seq_len, hidden_dim)  float16  — the h_ln vectors
        cossim_hidden   (seq_len, L, L)           float16  — cos-sim over h_ln
        cossim_vocab    (seq_len, L, L)           float16  — cos-sim over logits
    """
    device  = model_logits.device
    seq_len = model_logits.shape[1]
    L       = len(hidden_states)

    targets_th = torch.as_tensor(targets, device=device, dtype=torch.long)
    pos_idx    = torch.arange(seq_len, device=device)

    # Final-layer model log-probs — reference distribution for KL divergence
    model_lp = model_logits.squeeze(0).log_softmax(-1)   # (seq_len, V)
    model_p  = model_lp.exp()

    all_ce:      list[np.ndarray] = []
    all_ent:     list[np.ndarray] = []
    all_fkl:     list[np.ndarray] = []
    all_top:     list[np.ndarray] = []
    all_proj_ln: list[np.ndarray] = []   # (seq_len, hidden_dim) per layer

    with torch.no_grad():
        for l, h in enumerate(hidden_states):
            # Step 1 — translate into final-layer basis (residual affine).
            # For l = L-1 (final layer), transform_hidden returns h unchanged.
            h_translated = tuned_lens.transform_hidden(h, l)        # (1, seq_len, hidden_dim)

            # Step 2 — apply final layer norm.
            # tuned_lens.unembed.final_norm is a deep copy of the model's
            # final LayerNorm, stored inside the Unembed module.
            h_ln = tuned_lens.unembed.final_norm(h_translated)      # (1, seq_len, hidden_dim)

            # Step 3 — unembed to logits.
            # tuned_lens.unembed.unembedding is a deep copy of the model's
            # unembedding nn.Linear (no bias for most architectures).
            tl_logits = tuned_lens.unembed.unembedding(h_ln)        # (1, seq_len, vocab_size)

            # Compute metrics from log-probs
            lp = tl_logits.squeeze(0).log_softmax(-1)               # (seq_len, V)
            p  = lp.exp()

            all_ce.append((-lp[pos_idx, targets_th]).cpu().float().numpy())
            all_ent.append((-(p * lp).sum(-1)).cpu().float().numpy())
            all_fkl.append(((model_p * (model_lp - lp)).sum(-1)).cpu().float().numpy())
            all_top.append(lp.argmax(-1).cpu().numpy().astype(np.int32))
            all_proj_ln.append(h_ln.squeeze(0).cpu().float().numpy())   # (seq_len, hidden_dim)

    # Stack: (L, seq_len, hidden_dim) float32
    proj_ln_np = np.stack(all_proj_ln, axis=0)

    # Unembedding weight from tuned_lens (same deep copy used above)
    # shape: (vocab_size, hidden_dim) float32
    U_np = tuned_lens.unembed.unembedding.weight.detach().cpu().float().numpy()

    # Cosine-similarity matrices — computed per token position j.
    # We compute logits on-the-fly and discard them (too large to store).
    cossim_hidden_list: list[np.ndarray] = []
    cossim_vocab_list:  list[np.ndarray] = []

    for j in range(seq_len):
        H_j      = proj_ln_np[:, j, :]    # (L, hidden_dim) float32
        logits_j = H_j @ U_np.T           # (L, vocab_size) float32
        cossim_hidden_list.append(cosine_sim_matrix(H_j))
        cossim_vocab_list.append(cosine_sim_matrix(logits_j))

    return {
        "cross_entropy":  np.array(all_ce,  dtype=np.float16),         # (L, seq_len)
        "entropy":        np.array(all_ent, dtype=np.float16),         # (L, seq_len)
        "forward_kl":     np.array(all_fkl, dtype=np.float16),         # (L, seq_len)
        "top_token_ids":  np.array(all_top, dtype=np.int32),           # (L, seq_len)
        "proj_ln_states": proj_ln_np.astype(np.float16),               # (L, seq_len, hidden_dim)
        "cossim_hidden":  np.stack(cossim_hidden_list, axis=0),        # (seq_len, L, L)
        "cossim_vocab":   np.stack(cossim_vocab_list,  axis=0),        # (seq_len, L, L)
    }


# ─────────────────────────────────────────────────────────────────────────────
# §4.  Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_check(
    tuned_lens,
    hidden_states: tuple[torch.Tensor, ...],
    model_logits: torch.Tensor,
    tl_data: dict[str, np.ndarray],
    targets: list[int],
    excerpt_id: int,
) -> None:
    """
    Verify tl_data is exactly consistent with tuned_lens.forward().

    Since compute_tuned_lens_data calls the same internal methods as
    tuned_lens.forward(), we expect 100% top-token agreement and near-zero
    CE error (float16 storage rounding only).

    Checks
    ──────
    1. TOP-TOKEN AGREEMENT
       argmax(tuned_lens.forward(h, l)) == tl_data["top_token_ids"][l, :]
       Expected: 100%.

    2. CROSS-ENTROPY CONSISTENCY
       -log_softmax(tuned_lens.forward(h, l))[target_j]
       ≈ tl_data["cross_entropy"][l, j]
       Expected: mean |error| < 0.01 nats.

    3. PROJ_LN_STATES CONSISTENCY
       tuned_lens.unembed.final_norm(tuned_lens.transform_hidden(h, l))
       ≈ tl_data["proj_ln_states"][l, :]
       Expected: max |error| < 0.001 (float16 storage rounding only).

    4. COSSIM_HIDDEN RE-DERIVATION
       cosine_sim_matrix(tl_data["proj_ln_states"][:, j, :])
       ≈ tl_data["cossim_hidden"][j]
       Expected: max |error| < 0.002.

    Raises RuntimeError with a descriptive message if any check fails.
    """
    print(f"\n  [Sanity check — excerpt {excerpt_id}]")

    device     = model_logits.device
    seq_len    = model_logits.shape[1]
    L          = len(hidden_states)
    targets_th = torch.as_tensor(targets, device=device, dtype=torch.long)
    pos_idx    = torch.arange(seq_len, device=device)

    # ── Checks 1 & 2: compare tuned_lens.forward() against stored arrays ──────
    # tuned_lens.forward() is the public API and is our ground truth.
    top_matches = 0
    ce_errors   = []
    n_total     = 0

    with torch.no_grad():
        for l, h in enumerate(hidden_states):
            tl_logits = tuned_lens.forward(h, l)                         # (1, seq_len, V)
            lp        = tl_logits.squeeze(0).log_softmax(-1)             # (seq_len, V)

            ref_top = lp.argmax(-1).cpu().numpy().astype(np.int32)       # (seq_len,)
            ref_ce  = (-lp[pos_idx, targets_th]).cpu().float().numpy()   # (seq_len,)

            stored_top = tl_data["top_token_ids"][l]                     # (seq_len,)
            stored_ce  = tl_data["cross_entropy"][l].astype(np.float32)  # (seq_len,)

            top_matches += int((ref_top == stored_top).sum())
            ce_errors.extend(np.abs(ref_ce - stored_ce).tolist())
            n_total += seq_len

    top_pct     = 100.0 * top_matches / n_total
    mean_ce_err = float(np.mean(ce_errors))
    max_ce_err  = float(np.max(ce_errors))

    print(f"    Check 1 — top-token agreement:    {top_pct:.2f}%    (expected 100%)")
    print(f"    Check 2 — mean |CE error| (nats): {mean_ce_err:.6f}  (expected < 0.01)")
    print(f"               max  |CE error| (nats): {max_ce_err:.6f}")

    if top_pct < 100.0:
        raise RuntimeError(
            f"SANITY CHECK FAILED (excerpt {excerpt_id}): "
            f"top-token agreement {top_pct:.2f}% < 100%. "
            "compute_tuned_lens_data does not match tuned_lens.forward(). "
            "Check for in-place tensor modifications or mismatched layer indices."
        )
    if mean_ce_err > 0.01:
        raise RuntimeError(
            f"SANITY CHECK FAILED (excerpt {excerpt_id}): "
            f"mean CE error {mean_ce_err:.6f} nats > 0.01. "
            "tl_cross_entropy is inconsistent with tuned_lens.forward()."
        )

    # ── Check 3: proj_ln_states vs direct re-computation ─────────────────────
    proj_errors = []
    with torch.no_grad():
        for l, h in enumerate(hidden_states):
            h_translated = tuned_lens.transform_hidden(h, l)
            h_ln_ref     = tuned_lens.unembed.final_norm(h_translated)
            h_ln_ref_np  = h_ln_ref.squeeze(0).cpu().float().numpy()        # (seq_len, hidden_dim)
            h_ln_stored  = tl_data["proj_ln_states"][l].astype(np.float32)  # (seq_len, hidden_dim)
            proj_errors.append(float(np.abs(h_ln_ref_np - h_ln_stored).max()))

    max_proj_err = float(np.max(proj_errors))
    print(f"    Check 3 — proj_ln_states max |err|: {max_proj_err:.6f}  (expected < 0.001)")

    if max_proj_err > 0.001:
        raise RuntimeError(
            f"SANITY CHECK FAILED (excerpt {excerpt_id}): "
            f"proj_ln_states max error {max_proj_err:.6f} > 0.001. "
            "Stored h_ln values do not match direct re-computation. "
            "This indicates a float16 storage issue or in-place modification."
        )

    # ── Check 4: cossim_hidden re-derivation ──────────────────────────────────
    cossim_errors = []
    for j in range(min(5, seq_len)):
        H_j     = tl_data["proj_ln_states"][:, j, :].astype(np.float32)
        derived = cosine_sim_matrix(H_j).astype(np.float32)
        stored  = tl_data["cossim_hidden"][j].astype(np.float32)
        cossim_errors.append(float(np.abs(derived - stored).max()))

    max_cossim_err = float(np.max(cossim_errors))
    print(f"    Check 4 — cossim_hidden max |err|: {max_cossim_err:.6f}  (expected < 0.002)")

    if max_cossim_err > 0.002:
        raise RuntimeError(
            f"SANITY CHECK FAILED (excerpt {excerpt_id}): "
            f"cossim_hidden max error {max_cossim_err:.6f} > 0.002. "
            "Cosine similarities are inconsistent with stored proj_ln_states."
        )

    print("    ✓ All sanity checks passed.")


# ─────────────────────────────────────────────────────────────────────────────
# §5.  File I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_excerpt(output_dir: str | Path, excerpt_id: int, data: dict) -> Path:
    """Save all data for one excerpt as a compressed .npz archive."""
    excerpts_dir = Path(output_dir) / _EXCERPTS_SUBDIR
    excerpts_dir.mkdir(parents=True, exist_ok=True)
    path   = excerpts_dir / f"excerpt_{excerpt_id:04d}.npz"
    arrays = {
        k: (np.asarray(v) if not isinstance(v, np.ndarray) else v)
        for k, v in data.items()
        if v is not None
    }
    np.savez_compressed(str(path), **arrays)
    return path


def load_excerpt(output_dir: str | Path, excerpt_id: int) -> dict:
    """Load one excerpt.  Scalar 0-d arrays are returned as Python scalars."""
    path = Path(output_dir) / _EXCERPTS_SUBDIR / f"excerpt_{excerpt_id:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No excerpt file at {path}")
    raw    = np.load(str(path), allow_pickle=True)
    result = {k: (raw[k].item() if raw[k].ndim == 0 else raw[k]) for k in raw.files}
    raw.close()
    return result


def list_excerpt_ids(output_dir: str | Path) -> list[int]:
    """Return sorted list of excerpt IDs present in output_dir."""
    excerpts_dir = Path(output_dir) / _EXCERPTS_SUBDIR
    if not excerpts_dir.is_dir():
        return []
    return sorted(
        int(p.stem.split("_", 1)[1])
        for p in excerpts_dir.glob("excerpt_*.npz")
    )


def load_excerpts(
    output_dir: str | Path,
    excerpt_ids: list[int] | None = None,
) -> list[dict]:
    """Load multiple excerpts (all available if excerpt_ids is None)."""
    if excerpt_ids is None:
        excerpt_ids = list_excerpt_ids(output_dir)
    return [load_excerpt(output_dir, eid) for eid in excerpt_ids]


# ─────────────────────────────────────────────────────────────────────────────
# §6.  Main collection function
# ─────────────────────────────────────────────────────────────────────────────

def collect_token_evolution_data(
    texts: list[str],
    *,
    output_dir: str = OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    device: str = DEVICE,
    prompt_len: int = PROMPT_LEN,
    batch_size: int = BATCH_SIZE,
    verbose: bool = True,
    enable_tuned_lens: bool = False,
    save_hidden_states: bool = False,
    enable_generation: bool = False,
    sanity_check_first_n: int = 3,
) -> None:
    """
    Run a full data-collection pass over the provided text excerpts.

    Parameters
    ----------
    texts                : Pre-filtered list of text strings.
    output_dir           : Root directory for output files.
    model_name           : HuggingFace model identifier.
    device               : Torch device string.
    prompt_len           : Tokens used as generation prompt (only with enable_generation).
    batch_size           : Forward-pass batch size for the base model.
                           TunedLens passes are always single-sample (unpadded).
    verbose              : Print per-excerpt progress.
    enable_tuned_lens    : Collect TunedLens data (proj_ln_states, cossims, etc.).
    save_hidden_states   : Also save raw per-layer hidden states (~10 MB/excerpt).
    enable_generation    : Run autoregressive generation + BERTScore.
    sanity_check_first_n : Run full sanity check on the first N excerpts.
                           Set 0 to skip (not recommended for new data runs).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Model and tokenizer ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    MAX_CTX = model.config.max_position_embeddings

    # ── TunedLens ─────────────────────────────────────────────────────────────
    tuned_lens = None
    if enable_tuned_lens:
        from tuned_lens.nn.lenses import TunedLens
        tuned_lens = TunedLens.from_model_and_pretrained(
            model, map_location=device,
        ).to(device)
        tuned_lens.eval()

        if verbose:
            print(f"TunedLens loaded: {len(tuned_lens)} translators")
            print(f"  unembed.final_norm:  {type(tuned_lens.unembed.final_norm).__name__}")
            print(f"  unembed.unembedding: {type(tuned_lens.unembed.unembedding).__name__} "
                  f"shape={tuple(tuned_lens.unembed.unembedding.weight.shape)}")

    if verbose:
        print(f"Model:   {model_name}  |  Device: {device}")
        print(f"Excerpts to process: {len(texts)}")
        if enable_tuned_lens:
            print(f"Sanity check: first {sanity_check_first_n} excerpts")
        if save_hidden_states:
            print("Saving raw hidden states: yes")

    # ── Timing helper ─────────────────────────────────────────────────────────
    def sync_and_time() -> float:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # ── Output CSV ────────────────────────────────────────────────────────────
    EXCERPT_CSV = os.path.join(output_dir, "excerpt_level.csv")
    pd.DataFrame(columns=EXCERPT_FIELDS).to_csv(EXCERPT_CSV, index=False)

    excerpt_id      = 0
    tl_layer_labels = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for batch_start in tqdm(
        range(0, len(texts), batch_size),
        desc="Batches",
        disable=not verbose,
    ):
        batch_texts = texts[batch_start : batch_start + batch_size]

        # Tokenise each text in the batch
        batch_token_lists      = []
        batch_prompt_tokens    = []
        batch_reference_tokens = []
        batch_prompt_texts     = []
        batch_reference_texts  = []

        for text in batch_texts:
            tokens = tokenizer(text)["input_ids"]
            batch_token_lists.append(torch.tensor(tokens[:MAX_CTX]))
            batch_prompt_tokens.append(tokens[:prompt_len])
            batch_reference_tokens.append(tokens[prompt_len:])
            batch_prompt_texts.append(tokenizer.decode(tokens[:prompt_len]))
            batch_reference_texts.append(tokenizer.decode(tokens[prompt_len:]))

        if not batch_token_lists:
            continue

        # Pad to equal length for batched model forward pass
        seq_lens = [len(t) for t in batch_token_lists]
        max_len  = max(seq_lens)

        input_ids_list      = []
        attention_mask_list = []
        for t in batch_token_lists:
            pad = max_len - len(t)
            input_ids_list.append(
                torch.cat([t, torch.full((pad,), tokenizer.pad_token_id)])
            )
            attention_mask_list.append(
                torch.cat([torch.ones(len(t)), torch.zeros(pad)])
            )

        input_ids      = torch.stack(input_ids_list).to(device)
        attention_mask = torch.stack(attention_mask_list).to(device)

        # ── Batched model forward pass ────────────────────────────────────────
        t0 = sync_and_time()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        forward_ms = (sync_and_time() - t0) * 1000

        logits              = outputs.logits          # (B, max_len, vocab_size)
        hidden_states_batch = outputs.hidden_states   # tuple of (B, max_len, hidden_dim), L entries

        # Per-token cross-entropy losses (next-token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fct     = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=tokenizer.pad_token_id,
        )
        token_losses_batch = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(len(batch_token_lists), -1)            # (B, max_len-1)

        # ── Per-excerpt processing ────────────────────────────────────────────
        for i in range(len(batch_token_lists)):
            actual_len = seq_lens[i]

            # Core arrays (always saved)
            ids_np    = input_ids[i, :actual_len].cpu().numpy().astype(np.int64)
            tok_strs  = np.array(
                [tokenizer.decode([int(t)]) for t in ids_np], dtype=object,
            )
            losses_np = token_losses_batch[i, :actual_len - 1].cpu().float().numpy()
            preds_np  = logits[i, :actual_len - 1].argmax(-1).cpu().numpy().astype(np.int64)
            avg_loss  = float(losses_np.mean())

            excerpt_data: dict = {
                "model_name":      model_name,
                "input_ids":       ids_np,
                "token_strings":   tok_strs,
                "token_losses":    losses_np.astype(np.float32),
                "pred_token_ids":  preds_np,
                "avg_loss":        avg_loss,
                "perplexity":      float(np.exp(avg_loss)),
                "forward_pass_ms": forward_ms,
            }

            # ── Raw hidden states (optional) ──────────────────────────────────
            if save_hidden_states:
                excerpt_data["hidden_states"] = torch.stack(
                    [h[i, :actual_len, :] for h in hidden_states_batch]
                ).cpu().half().numpy()                # (L, actual_len, hidden_dim) float16

            # ── TunedLens data (optional) ─────────────────────────────────────
            if tuned_lens is not None:
                # Extract this sample's hidden states unpadded, one per layer.
                # hidden_states_batch has n_layers+1 entries: index 0 is the
                # embedding output, indices 1..n_layers are transformer layer
                # outputs.  TunedLens translator[l] expects the output of
                # transformer layer l, so we skip index 0.
                # Shape of each: (1, actual_len, hidden_dim) — stays on device.
                sample_hidden = tuple(
                    h[i : i + 1, :actual_len, :]
                    for h in hidden_states_batch[1:]
                )
                sample_logits = logits[i : i + 1, :actual_len, :]

                # Next-token targets:
                #   target[j] = input_ids[j+1]  for j = 0 … actual_len-2
                # The last entry uses EOS as a dummy (no real next token);
                # it is sliced away below before saving.
                target_ids = ids_np[1:].tolist() + [tokenizer.eos_token_id or 0]

                tl_data = compute_tuned_lens_data(
                    tuned_lens    = tuned_lens,
                    hidden_states = sample_hidden,
                    model_logits  = sample_logits,
                    targets       = target_ids,
                )

                # Run sanity check on the first sanity_check_first_n excerpts
                if excerpt_id < sanity_check_first_n:
                    run_sanity_check(
                        tuned_lens    = tuned_lens,
                        hidden_states = sample_hidden,
                        model_logits  = sample_logits,
                        tl_data       = tl_data,
                        targets       = target_ids,
                        excerpt_id    = excerpt_id,
                    )

                # Build layer labels once
                if tl_layer_labels is None:
                    L = tl_data["cross_entropy"].shape[0]
                    tl_layer_labels = [f"layer_{l}" for l in range(L - 1)] + ["output"]

                # Slice to n = actual_len-1 positions to align with token_losses.
                # tl_data has seq_len = actual_len; we drop the last position
                # (which used the dummy EOS target).
                n = actual_len - 1
                excerpt_data["layer_labels"]      = np.array(tl_layer_labels)
                excerpt_data["tl_cross_entropy"]  = tl_data["cross_entropy"][:, :n]
                excerpt_data["tl_entropy"]        = tl_data["entropy"][:, :n]
                excerpt_data["tl_forward_kl"]     = tl_data["forward_kl"][:, :n]
                excerpt_data["tl_top_token_ids"]  = tl_data["top_token_ids"][:, :n]
                excerpt_data["tl_proj_ln_states"] = tl_data["proj_ln_states"][:, :n, :]
                excerpt_data["tl_cossim_hidden"]  = tl_data["cossim_hidden"][:n]
                excerpt_data["tl_cossim_vocab"]   = tl_data["cossim_vocab"][:n]

            # ── Generation + BERTScore (optional) ────────────────────────────
            generated_text   = None
            bertscore        = None
            generation_ms    = 0.0
            tokens_generated = 0
            tokens_per_sec   = 0.0

            if enable_generation:
                prompt_ids  = torch.tensor(batch_prompt_tokens[i]).unsqueeze(0).to(device)
                prompt_mask = torch.ones_like(prompt_ids)
                ref_tokens  = batch_reference_tokens[i]
                ref_text    = batch_reference_texts[i]
                gen_len     = min(len(ref_tokens), MAX_CTX - prompt_len)

                t0 = sync_and_time()
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=gen_len,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                generation_ms    = (sync_and_time() - t0) * 1000
                tokens_generated = generated_ids.shape[1] - prompt_ids.shape[1]
                tokens_per_sec   = (
                    tokens_generated / (generation_ms / 1000)
                    if generation_ms > 0 else 0.0
                )
                generated_text = tokenizer.decode(
                    generated_ids[0][len(batch_prompt_tokens[i]):],
                    skip_special_tokens=True,
                )

                try:
                    from bert_score import score as bert_score_fn
                    _, _, F1 = bert_score_fn(
                        [generated_text], [ref_text[:1000]],
                        lang="en", verbose=False,
                    )
                    bertscore = F1.item()
                except Exception as exc:
                    if verbose:
                        tqdm.write(f"  BERTScore failed (excerpt {excerpt_id}): {exc}")

                excerpt_data.update({
                    "prompt_text":      batch_prompt_texts[i],
                    "reference_text":   ref_text,
                    "generated_text":   generated_text or "",
                    "bertscore":        bertscore if bertscore is not None else float("nan"),
                    "generation_ms":    generation_ms,
                    "tokens_generated": tokens_generated,
                    "tokens_per_sec":   tokens_per_sec,
                })

            # ── Save ──────────────────────────────────────────────────────────
            save_excerpt(output_dir, excerpt_id, excerpt_data)

            pd.DataFrame([{
                "excerpt_id":       excerpt_id,
                "prompt":           batch_prompt_texts[i],
                "reference":        batch_reference_texts[i] if enable_generation else "",
                "generated":        generated_text or "",
                "avg_loss":         avg_loss,
                "perplexity":       excerpt_data["perplexity"],
                "bertscore":        bertscore,
                "forward_pass_ms":  forward_ms,
                "generation_ms":    generation_ms,
                "tokens_generated": tokens_generated,
                "tokens_per_sec":   tokens_per_sec,
            }]).to_csv(EXCERPT_CSV, mode="a", header=False, index=False)

            if verbose:
                parts = [
                    f"Excerpt {excerpt_id:04d}",
                    f"loss={avg_loss:.3f}",
                    f"ppl={excerpt_data['perplexity']:.1f}",
                    f"fwd={forward_ms:.0f}ms",
                ]
                if enable_tuned_lens:
                    parts.append("tl=ok")
                if enable_generation and bertscore is not None:
                    parts.append(f"bert={bertscore:.3f}")
                tqdm.write("  " + " | ".join(parts))

            excerpt_id += 1

    if verbose:
        print(f"\nDone. {excerpt_id} excerpts saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# §7.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # import numpy as np

    # d = np.load("NEW_DATA_TEST/excerpts/excerpt_0000.npz", allow_pickle=True)

    # # See everything that was saved
    # print(list(d.files))

    # # Check shapes and dtypes
    # for k in d.files:
    #     v = d[k]
    #     print(f"{k:25s}  shape={str(v.shape):25s}  dtype={v.dtype}")

    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Collect token-evolution data from a causal LM over WikiText."
    )
    parser.add_argument("--model",                default=MODEL_NAME)
    parser.add_argument("--device",               default=None)
    parser.add_argument("--output-dir",           default=OUTPUT_DIR)
    parser.add_argument("--dataset",              default="wikitext")
    parser.add_argument("--dataset-config",       default="wikitext-2-raw-v1")
    parser.add_argument("--split",                default="test")
    parser.add_argument("--min-total-len",        type=int, default=MIN_TOTAL_LEN)
    parser.add_argument("--max-seq-len",          type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--prompt-len",           type=int, default=PROMPT_LEN)
    parser.add_argument("--batch-size",           type=int, default=BATCH_SIZE)
    parser.add_argument("--enable-tuned-lens",    action="store_true")
    parser.add_argument("--save-hidden-states",   action="store_true")
    parser.add_argument("--generation",           action="store_true")
    parser.add_argument("--sanity-check-first-n", type=int, default=3,
                        help="Run sanity check on first N excerpts (default: 3)")
    parser.add_argument("--verbose",              action="store_true")
    parser.add_argument("--small-run",            action="store_true",
                        help="Process only 4 excerpts for quick testing")
    args = parser.parse_args()

    device    = args.device or DEVICE
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset   = load_dataset(args.dataset, args.dataset_config, split=args.split)

    texts = [x["text"] for x in dataset if is_valid(x["text"])]
    texts = [t for t in texts if passes_length_filter(t, tokenizer, args.min_total_len)]
    if args.max_seq_len:
        texts = [
            tokenizer.decode(tokenizer(t)["input_ids"][: args.max_seq_len])
            for t in texts
        ]
    if args.small_run:
        texts = texts[:4]

    collect_token_evolution_data(
        texts,
        output_dir           = args.output_dir,
        model_name           = args.model,
        device               = device,
        prompt_len           = args.prompt_len,
        batch_size           = args.batch_size,
        verbose              = args.verbose,
        enable_tuned_lens    = args.enable_tuned_lens,
        save_hidden_states   = args.save_hidden_states,
        enable_generation    = args.generation,
        sanity_check_first_n = args.sanity_check_first_n,
    )
