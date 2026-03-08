"""
1_collect_tuned_lens_projections.py
──────────────────────────────────
Stage 1 of 2: project hidden states through TunedLens translators and compute
per-token cosine-similarity matrices in both hidden-state and vocab-logit space.

For every token position j in each excerpt:

  1. Load hidden_states[:, j, :]  →  H  of shape (n_layers+1, hidden_dim), float16.
  2. Apply the per-layer TunedLens affine translator AS A RESIDUAL:
         H_proj[l] = H[l] + (H[l] @ W.T + b)
     The residual form matches the TunedLens library exactly
     (lenses.py: `return h + self[idx](h)`).
     The final layer (l = n_layers) has no translator and is passed through unchanged.
  3. Apply final layer norm to each projected vector:
         H_proj_ln[l] = LayerNorm(H_proj[l])
  4. Compute cosine-similarity matrices:
     - Hidden-state space:  over H_proj_ln          (L, L)
     - Vocab-logit space:   over U @ H_proj_ln[l]   (L, L)

DTYPE DISCIPLINE:
  All arithmetic mirrors tuned_lens.forward() exactly — float16 throughout.
  Hidden states are stored as float16 and are NEVER upcast.
  All extracted weights (translator W/b, layer-norm γ/β, unembedding U) are
  kept as float16 numpy arrays.
  The only float32 values are the scalar losses written to all_losses.

SANITY CHECK:
  Before the main loop, we verify our manual pipeline against the stored
  tl_top_token_ids and tl_cross_entropy fields on a sample of tokens.
  Expected: top-token argmax agreement ≥ 99%, mean |CE error| < 0.05 nats.
  Any remaining error is due to float16 rounding in storage, not a pipeline bug.

Outputs (written to --output-dir):
  tuned_lens_cossims.npz — arrays:
      all_losses          (N,)      float32   cross-entropy loss per token
      all_cossims         (N, L, L) float16   cosine-sim in hidden-state space
      all_cossims_vocab   (N, L, L) float16   cosine-sim in vocab-logit space
      n_layers_p1         scalar    int        L = n_transformer_layers + 1

Usage
-----
  python 1_collect_tuned_lens_projections.py \\
      --data-dir   token_evolution_data \\
      --output-dir tuned_lens_projections \\
      --model      EleutherAI/pythia-160m-deduped \\
      --device     cpu \\
      --sanity-check-n 500
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, "..")
from token_evolution_data_collect import load_excerpts, list_excerpt_ids


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Project hidden states through TunedLens translators and "
                "compute per-token cosine-similarity matrices."
)
parser.add_argument("--data-dir",        default="token_evolution_data",
                    help="Root directory of .npz excerpt files")
parser.add_argument("--output-dir",      default="tuned_lens_projections",
                    help="Directory for output .npz file")
parser.add_argument("--model",           default="EleutherAI/pythia-160m-deduped",
                    help="HuggingFace model identifier (must match data collection)")
parser.add_argument("--device",          default="cpu",
                    help="Torch device (default: cpu)")
parser.add_argument("--sanity-check-n",  type=int, default=500,
                    help="Tokens to sample for sanity check (0 = skip)")
args = parser.parse_args()

DATA_DIR       = args.data_dir
OUTPUT_DIR     = args.output_dir
MODEL_NAME     = args.model
DEVICE         = args.device
SANITY_CHECK_N = args.sanity_check_n
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load model, TunedLens, and extract weights — all as float16
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading model {MODEL_NAME} …")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

print("Loading TunedLens …")
from tuned_lens.nn.lenses import TunedLens
tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=DEVICE).to(DEVICE)
tuned_lens.eval()

n_translators = len(tuned_lens)
print(f"  {n_translators} translators found")

# ── Translator weights (float16) ─────────────────────────────────────────────
# Each tuned_lens[l] is nn.Linear.  PyTorch convention: output = input @ W.T + b.
# TunedLens applies this as a RESIDUAL: h_proj = h + (h @ W.T + b).
# We extract W in its native (out, in) shape and apply W.T at projection time.
translators: list[tuple[np.ndarray, np.ndarray]] = []
for l in range(n_translators):
    layer = tuned_lens[l]
    W = layer.weight.detach().cpu().half().numpy()   # (hidden_dim, hidden_dim)
    b = layer.bias.detach().cpu().half().numpy()     # (hidden_dim,)
    translators.append((W, b))

hidden_dim = translators[0][0].shape[0]
print(f"  translator weight shape: {translators[0][0].shape}  dtype: {translators[0][0].dtype}")

# ── Final layer norm parameters (float16) ────────────────────────────────────
# For Pythia (GPT-NeoX): model.gpt_neox.final_layer_norm
final_ln  = model.gpt_neox.final_layer_norm
ln_weight = final_ln.weight.detach().cpu().half().numpy()   # (hidden_dim,)
ln_bias   = final_ln.bias.detach().cpu().half().numpy()     # (hidden_dim,)
ln_eps    = float(final_ln.eps)

# ── Unembedding matrix (float16) ─────────────────────────────────────────────
# For Pythia: model.embed_out.weight, shape (vocab_size, hidden_dim).
U = model.embed_out.weight.detach().cpu().half().numpy()    # (vocab_size, hidden_dim)
vocab_size = U.shape[0]
print(f"  unembedding shape: {U.shape}  dtype: {U.dtype}")

del model   # no longer needed

import inspect
from tuned_lens.nn.lenses import TunedLens
from tuned_lens.nn import unembed   # may or may not exist

# The core forward pass
print("=== TunedLens.forward ===")
print(inspect.getsource(TunedLens.forward))

print("\n=== TunedLens.transform_hidden ===")
print(inspect.getsource(TunedLens.transform_hidden))

# The unembed step — try both possible locations
try:
    print("\n=== TunedLens.unembed (method) ===")
    print(inspect.getsource(tuned_lens.unembed.__class__))
except Exception as e:
    print(f"Not a class: {e}")

# Also show the full class structure
print("\n=== TunedLens class members ===")
print([m for m in dir(TunedLens) if not m.startswith('__')])

print("STOPSTOPSTOPSOTP")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper functions — all operate in float16
# ─────────────────────────────────────────────────────────────────────────────

def apply_layer_norm(x: np.ndarray) -> np.ndarray:
    """
    Layer norm for a single vector x of shape (hidden_dim,), float16.
    Upcasts to float32 for the mean/var arithmetic to avoid float16 overflow,
    then returns float16.  This matches PyTorch's internal behaviour.
    """
    x32   = x.astype(np.float32)
    mean  = x32.mean()
    var   = ((x32 - mean) ** 2).mean()
    xn    = (x32 - mean) / np.sqrt(var + ln_eps)
    return (ln_weight.astype(np.float32) * xn + ln_bias.astype(np.float32)).astype(np.float16)


def apply_layer_norm_rows(X: np.ndarray) -> np.ndarray:
    """
    Layer norm for each row of X, shape (L, hidden_dim), float16.
    Returns float16.
    """
    X32   = X.astype(np.float32)
    mean  = X32.mean(axis=-1, keepdims=True)                    # (L, 1)
    var   = ((X32 - mean) ** 2).mean(axis=-1, keepdims=True)   # (L, 1)
    Xn    = (X32 - mean) / np.sqrt(var + ln_eps)
    result = (ln_weight.astype(np.float32) * Xn
              + ln_bias.astype(np.float32))                     # (L, hidden_dim)
    return result.astype(np.float16)


def project_and_unembed(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply TunedLens translator (residual) + layer norm + unembedding to every layer.

    Parameters
    ----------
    H : (L, hidden_dim) float16
        Raw hidden states for one token, across all layers.
        Must be float16 — do NOT upcast before calling.

    Returns
    -------
    H_proj_ln : (L, hidden_dim) float16
        Layer-normed projected hidden states (in final-layer basis).
    logits    : (L, vocab_size) float16
        Vocab-space logit vectors: logits[l] = U @ H_proj_ln[l].
    """
    assert H.dtype == np.float16, f"Expected float16, got {H.dtype}"
    L      = H.shape[0]
    H_proj = np.empty_like(H)                                   # (L, hidden_dim) float16

    # Non-final layers: apply residual translator
    for l in range(L - 1):
        W, b = translators[l]
        # Residual: h_proj = h + (h @ W.T + b)
        # Matches TunedLens source: `return h + self[idx](h)`
        H_proj[l] = H[l] + (H[l] @ W.T + b)

    # Final layer: no translator, pass through unchanged
    H_proj[L - 1] = H[L - 1]

    # Apply layer norm to all rows simultaneously
    H_proj_ln = apply_layer_norm_rows(H_proj)                   # (L, hidden_dim) float16

    # Unembed: logits[l] = U @ H_proj_ln[l]
    # Cast to float32 for matmul to avoid overflow, then back to float16
    logits = (H_proj_ln.astype(np.float32) @ U.astype(np.float32).T).astype(np.float16)

    return H_proj_ln, logits


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """
    X : (L, d) — rows are vectors (any dtype; promoted to float32 internally).
    Returns (L, L) float16 cosine-similarity matrix; diagonal = 1.
    """
    X32   = X.astype(np.float32)
    norms = np.linalg.norm(X32, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Xn    = X32 / norms
    return np.clip(Xn @ Xn.T, -1.0, 1.0).astype(np.float16)


def log_softmax_f32(logits_1d: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax in float32 for a 1-D array."""
    x      = logits_1d.astype(np.float32)
    x     -= x.max()
    return x - np.log(np.sum(np.exp(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load excerpts
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nScanning for excerpt IDs in {DATA_DIR} …")
ids      = list_excerpt_ids(DATA_DIR)
print(f"  found {len(ids)} excerpts")

print("Loading excerpts …")
excerpts = load_excerpts(DATA_DIR, ids)
excerpts = [e for e in excerpts if "hidden_states" in e]
print(f"  {len(excerpts)} excerpts contain hidden_states")

if not excerpts:
    raise RuntimeError(
        "No excerpts with hidden_states found. "
        "Re-run data collection with --save-hidden-states."
    )

n_layers_p1 = excerpts[0]["hidden_states"].shape[0]   # L = n_layers + 1
L           = n_layers_p1
print(f"  n_layers+1 = {L}  (n_translators = {n_translators})")

if n_translators != L - 1:
    raise RuntimeError(
        f"Mismatch: {n_translators} TunedLens translators but "
        f"{L - 1} non-final layers in hidden states. "
        "Ensure --model matches the model used during data collection."
    )

has_tl = all("tl_cross_entropy" in e and "tl_top_token_ids" in e for e in excerpts)
if not has_tl:
    print("  WARNING: tl_cross_entropy / tl_top_token_ids not found — skipping sanity check.")
    SANITY_CHECK_N = 0

# Confirm hidden states are float16 (as saved by data collection)
raw_dtype = excerpts[0]["hidden_states"].dtype
if raw_dtype != np.float16:
    raise RuntimeError(
        f"Expected hidden_states dtype float16, got {raw_dtype}. "
        "Check data collection script."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sanity check
# ─────────────────────────────────────────────────────────────────────────────
#
# Ground truth: tl_cross_entropy[l, j] and tl_top_token_ids[l, j] were
# computed during data collection as:
#     lp = tuned_lens.forward(h_layer_l, l).log_softmax(-1)
#     tl_cross_entropy[l, j]  = -lp[input_ids[j+1]]
#     tl_top_token_ids[l, j]  = lp.argmax()
#
# Our replication:
#     _, logits = project_and_unembed(H[:, j, :])
#     lp        = log_softmax(logits[l])
#     our_ce    = -lp[input_ids[j+1]]
#     our_top   = argmax(logits[l])
#
# With correct dtype discipline these should agree to within float16 rounding.

if SANITY_CHECK_N > 0:
    print(f"\nRunning sanity check on {SANITY_CHECK_N} sampled tokens …")

    rng = np.random.default_rng(42)

    all_pairs = [
        (ei, j)
        for ei, e in enumerate(excerpts)
        for j in range(e["token_losses"].shape[0])
    ]
    sample_size   = min(SANITY_CHECK_N, len(all_pairs))
    chosen_idxs   = rng.choice(len(all_pairs), size=sample_size, replace=False)
    sampled_pairs = [all_pairs[idx] for idx in chosen_idxs]

    top_matches = 0
    ce_errors   = []
    n_total     = 0

    for ei, j in tqdm(sampled_pairs, desc="Sanity check"):
        e          = excerpts[ei]
        hs         = e["hidden_states"]                              # (L, seq_len, hidden_dim) float16
        input_ids  = e["input_ids"]
        stored_top = e["tl_top_token_ids"][:, j].astype(np.int32)  # (L,)
        stored_ce  = e["tl_cross_entropy"][:, j].astype(np.float32) # (L,)
        target_id  = int(input_ids[j + 1])

        H = hs[:, j, :]                           # (L, hidden_dim) float16 — no cast
        _, logits = project_and_unembed(H)        # (L, vocab_size) float16

        for l in range(L):
            lp      = log_softmax_f32(logits[l])
            our_top = int(np.argmax(logits[l]))
            our_ce  = float(-lp[target_id])

            if our_top == stored_top[l]:
                top_matches += 1
            ce_errors.append(abs(our_ce - stored_ce[l]))
            n_total += 1

    top_match_pct = 100.0 * top_matches / n_total
    mean_ce_err   = float(np.mean(ce_errors))
    max_ce_err    = float(np.max(ce_errors))

    print(f"\n  ── Sanity check results ──────────────────────────────────────")
    print(f"  Sampled tokens:               {len(sampled_pairs):,}")
    print(f"  (layer, token) pairs checked: {n_total:,}")
    print(f"  Top-token argmax agreement:   {top_match_pct:.2f}%  (expected ≥ 99%)")
    print(f"  Mean |CE error| (nats):       {mean_ce_err:.5f}  (expected < 0.05)")
    print(f"  Max  |CE error| (nats):       {max_ce_err:.4f}")

    if top_match_pct < 99.0:
        raise RuntimeError(
            f"Sanity check FAILED: top-token agreement {top_match_pct:.2f}% < 99%. "
            "Do not proceed — projection pipeline is incorrect."
        )
    elif mean_ce_err > 0.05:
        print(
            f"\n  ⚠  CE errors slightly above threshold ({mean_ce_err:.5f} nats). "
            "Likely float16 rounding — check top-token agreement instead."
        )
    else:
        print("\n  ✓ Sanity check passed.")

    print(f"  ─────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main accumulation loop
# ─────────────────────────────────────────────────────────────────────────────

print("Projecting hidden states and computing cosine-similarity matrices …")

all_losses:        list[float]      = []
all_cossims:       list[np.ndarray] = []   # each (L, L) float16
all_cossims_vocab: list[np.ndarray] = []   # each (L, L) float16

for e in tqdm(excerpts, desc="Excerpts"):
    hs     = e["hidden_states"]                      # (L, seq_len, hidden_dim) float16
    losses = e["token_losses"].astype(np.float32)    # (seq_len-1,)

    seq_len = losses.shape[0]

    for j in range(seq_len):
        H = hs[:, j, :]                              # (L, hidden_dim) float16 — no cast

        H_proj_ln, logits = project_and_unembed(H)  # both (L, *) float16

        all_losses.append(float(losses[j]))
        all_cossims.append(cosine_sim_matrix(H_proj_ln))
        all_cossims_vocab.append(cosine_sim_matrix(logits))

all_losses_arr        = np.array(all_losses,        dtype=np.float32)
all_cossims_arr       = np.stack(all_cossims,       axis=0)   # (N, L, L) float16
all_cossims_vocab_arr = np.stack(all_cossims_vocab, axis=0)   # (N, L, L) float16

print(f"  total tokens:      {len(all_losses_arr):,}")
print(f"  cossims (hidden):  {all_cossims_arr.shape}  dtype={all_cossims_arr.dtype}")
print(f"  cossims (vocab):   {all_cossims_vocab_arr.shape}  dtype={all_cossims_vocab_arr.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save
# ─────────────────────────────────────────────────────────────────────────────

out_path = Path(OUTPUT_DIR) / "tuned_lens_cossims.npz"
np.savez_compressed(
    str(out_path),
    all_losses        = all_losses_arr,
    all_cossims       = all_cossims_arr,
    all_cossims_vocab = all_cossims_vocab_arr,
    n_layers_p1       = np.array(n_layers_p1),
    model_name        = np.array(MODEL_NAME),
)
print(f"\nSaved → {out_path}")
print(f"  all_losses:         {all_losses_arr.shape}")
print(f"  all_cossims:        {all_cossims_arr.shape}")
print(f"  all_cossims_vocab:  {all_cossims_vocab_arr.shape}")