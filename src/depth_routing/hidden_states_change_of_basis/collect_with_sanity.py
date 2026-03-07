"""
1_collect_tuned_lens_projections.py
──────────────────────────────────
Stage 1 of 2 in pipeline to analyze cosine similarities of hidden state vectors
after applying TunedLens affine mapping projections to convert each representation
into the final layer's basis.

For every token position j in each excerpt (paired with its cross-entropy
loss at position j):

  1. Load hidden_states[:, j, :]  →  H  of shape (n_layers+1, hidden_dim)
  2. Apply the per-layer TunedLens affine translator:
         H_proj[l] = W_l @ H[l] + b_l
     putting every layer's representation into the final layer's basis.
     The final layer (l = n_layers) is left as-is (identity mapping).
  3. Apply final layer norm to each projected vector:
         H_proj_ln[l] = LayerNorm(H_proj[l])
     and compute the (n_layers+1, n_layers+1) cosine-similarity matrix over
     those LN-normalized vectors. LN is required here for the same reason it
     is applied before unembedding: the translators map into the pre-LN
     final-layer basis, so LN must be applied before comparing directions.
  4. The logit vectors logits[l] = U @ H_proj_ln[l] (already computed inside
     project_and_unembed) are used to compute the vocab-space cosine-similarity
     matrix. Since cosine similarity is scale-invariant, this is equivalent to
     comparing directions in the post-LN vocab projection space.

SANITY CHECK (run before the main accumulation loop):
  On a small sample of tokens, we re-derive the TunedLens predictions
  (argmax token and cross-entropy) from scratch and compare against the
  stored tl_top_token_ids and tl_cross_entropy fields. These were computed
  during original data collection via tuned_lens.forward(h, l), which applies
  translator + layer_norm + unembedding in one call. Our manual re-derivation
  must match to confirm the pipeline is correct.

  The check passes if:
    - top-token argmax agreement >= 99% across sampled (layer, token) pairs
    - mean absolute CE error < 0.05 nats (some error expected from float16
      storage of hidden states and tl_cross_entropy)

Outputs (written to --output-dir):
  tuned_lens_cossims.npz   — arrays:
      all_losses          (N,)           float32  cross-entropy loss per token
      all_cossims         (N, L, L)      float16  cosine-sim in hidden-state space
      all_cossims_vocab   (N, L, L)      float16  cosine-sim in vocab logit space
      n_layers_p1         scalar         int      L = n_transformer_layers + 1

Usage
-----
  python collect_with_sanity.py \\
      --data-dir   token_evolution_data \\
      --output-dir tuned_lens_projections \\
      --model      EleutherAI/pythia-160m-deduped \\
      --device     cpu \\
      --sanity-check-n 500

Notes
-----
  • Requires hidden states AND TunedLens data saved during original collection
    (collect_token_evolution_data with save_hidden_states=True and
    enable_tuned_lens=True).
  • TunedLens weights are downloaded automatically from HuggingFace the
    first time; they are cached by the huggingface_hub library.
  • The final layer (l = L-1) has no TunedLens translator (it IS the target
    basis), so its hidden state is passed through unchanged and then
    layer-normed + unembedded directly.
  • For Pythia/GPT-NeoX: layer norm is model.gpt_neox.final_layer_norm and
    the unembedding is model.embed_out. Both are applied inside
    tuned_lens.forward(), so we replicate them here exactly.
  • Cosine-sim matrices are stored as float16 to keep the output file small.
    All arithmetic is done in float32.
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
    description="Project hidden states through TunedLens translators, run a "
                "sanity check against stored tl_* fields, and compute "
                "per-token cosine-similarity matrices in both hidden-state "
                "and vocab-logit space."
)
parser.add_argument("--data-dir",        default="token_evolution_data",
                    help="Root directory of .npz excerpt files")
parser.add_argument("--output-dir",      default="tuned_lens_projections",
                    help="Directory for output .npz file")
parser.add_argument("--model",           default="EleutherAI/pythia-160m-deduped",
                    help="HuggingFace model identifier matching the saved hidden states")
parser.add_argument("--device",          default="cpu",
                    help="Torch device (default: cpu)")
parser.add_argument("--sanity-check-n", type=int, default=500,
                    help="Number of tokens to sample for the sanity check "
                         "(default: 500; set 0 to skip)")
args = parser.parse_args()

DATA_DIR       = args.data_dir
OUTPUT_DIR     = args.output_dir
MODEL_NAME     = args.model
DEVICE         = args.device
SANITY_CHECK_N = args.sanity_check_n
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load model, TunedLens, and extract weights
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading model {MODEL_NAME} …")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

print("Loading TunedLens …")
from tuned_lens.nn.lenses import TunedLens
tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=DEVICE).to(DEVICE)
tuned_lens.eval()

n_translators = len(tuned_lens)   # one per non-final layer
print(f"  {n_translators} translators found")

# ── Extract affine translator weights ────────────────────────────────────────
# Each tuned_lens[l] is an nn.Linear: output = input @ W.T + b
# For a hidden-state row vector h of shape (hidden_dim,):
#   tuned_lens[l](h) = h @ W.T + b
# Which is equivalent to: W @ h + b  (h treated as column vector).
# We store W as (hidden_dim, hidden_dim) so we can do np.dot(W, h).
# translators = []
# for l in range(n_translators):
#     t = tuned_lens[l]
#     W = t.weight.detach().cpu().float().numpy()   # (hidden_dim, hidden_dim)
#     b = t.bias.detach().cpu().float().numpy()     # (hidden_dim,)
#     translators.append((W, b))

translators = []
for l in range(n_translators):
    t = tuned_lens[l]
    W = t.weight.detach().cpu().half().numpy()   # float16, not float32
    b = t.bias.detach().cpu().half().numpy()
    translators.append((W, b))

print(f"  translator weight shape: {translators[0][0].shape}")

# ── Extract final layer norm parameters ──────────────────────────────────────
# For Pythia (GPT-NeoX architecture): model.gpt_neox.final_layer_norm
# tuned_lens.forward(h, l) applies: translator → final_layer_norm → embed_out
final_ln  = model.gpt_neox.final_layer_norm
# ln_weight = final_ln.weight.detach().cpu().float().numpy()   # (hidden_dim,)
# ln_bias   = final_ln.bias.detach().cpu().float().numpy()     # (hidden_dim,)
# ln_eps    = float(final_ln.eps)


ln_weight = final_ln.weight.detach().cpu().half().numpy()
ln_bias   = final_ln.bias.detach().cpu().half().numpy()
ln_eps    = float(final_ln.eps)


# ── Extract unembedding matrix ───────────────────────────────────────────────
# For Pythia: model.embed_out.weight, shape (vocab_size, hidden_dim)
# This is NOT tied to the input embedding matrix.
# U = model.embed_out.weight.detach().cpu().float().numpy()   # (vocab_size, hidden_dim)
U = model.embed_out.weight.detach().cpu().half().numpy()
print(f"  unembedding shape: {U.shape}  (vocab_size={U.shape[0]}, hidden_dim={U.shape[1]})")

del model   # free memory — we no longer need the full model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm(x):
    """
    Apply final layer norm to a 1-D hidden-state vector x of shape (hidden_dim,).
    Uses the extracted ln_weight, ln_bias, ln_eps from the model.
    """
    mean = x.mean()
    var  = ((x - mean) ** 2).mean()
    xn   = (x - mean) / np.sqrt(var + ln_eps)
    return ln_weight * xn + ln_bias


def layer_norm_rows(X):
    """
    Apply final layer norm to each row of X, shape (L, hidden_dim).
    Vectorized: computes mean/var per row simultaneously.
    Returns (L, hidden_dim).
    """
    mean = X.mean(axis=-1, keepdims=True)                    # (L, 1)
    var  = ((X - mean) ** 2).mean(axis=-1, keepdims=True)   # (L, 1)
    Xn   = (X - mean) / np.sqrt(var + ln_eps)               # (L, hidden_dim)
    return ln_weight * Xn + ln_bias                          # (L, hidden_dim)


def project_and_unembed(H):
    """
    Apply TunedLens translator + layer norm + unembedding to every layer.

    Parameters
    ----------
    H : (L, hidden_dim) float32
        Raw hidden states for one token, across all layers.

    Returns
    -------
    H_proj : (L, hidden_dim) float32
        Projected hidden states in the final-layer basis.
        Layer L-1 (the final layer) is returned unchanged.
    logits : (L, vocab_size) float32
        Vocab-space logit vectors. logits[l] = U @ LayerNorm(H_proj[l]).
    """
    # L = H.shape[0]
    # H_proj = np.empty_like(H)                                 # (L, hidden_dim)
    # logits = np.empty((L, U.shape[0]), dtype=np.float32)      # (L, vocab_size)

    # for l in range(L - 1):
    #     # Step 1: apply affine translator → final-layer basis
    #     W, b = translators[l]
    #     # h_proj    = W @ H[l] + b              # (hidden_dim,)
    #     h_proj = H[l] @ W.T + b    # match PyTorch's convention exactly
    #     H_proj[l] = h_proj
    #     # Step 2: apply final layer norm
    #     h_ln      = layer_norm(h_proj)         # (hidden_dim,)
    #     # Step 3: unembed → logits
    #     logits[l] = U @ h_ln                   # (vocab_size,)

    # # Final layer: no translator; still apply LN + unembed
    # H_proj[L - 1] = H[L - 1]
    # h_ln           = layer_norm(H[L - 1])
    # logits[L - 1]  = U @ h_ln

    # return H_proj, logits

    H = H.astype(np.float16)   # match tuned_lens dtype
    L = H.shape[0]
    H_proj = np.empty_like(H)
    logits = np.empty((L, U.shape[0]), dtype=np.float16)

    for l in range(L - 1):
        # W, b = translators[l]
        # h_proj    = H[l] @ W.T + b        # W.T fix applied
        # H_proj[l] = h_proj
        # h_ln      = layer_norm(h_proj)
        # logits[l] = U @ h_ln
        W, b = translators[l]
        h_linear = H[l] @ W.T + b
        h_proj    = H[l] + h_linear   # residual add
        if l == 0:
            print(f"  [DEBUG] linear norm: {np.linalg.norm(h_linear):.4f}")
            print(f"  [DEBUG] H[0] norm:   {np.linalg.norm(H[l]):.4f}")
            print(f"  [DEBUG] h_proj norm: {np.linalg.norm(h_proj):.4f}")
        H_proj[l] = h_proj
        h_ln      = layer_norm(h_proj)
        logits[l] = U @ h_ln

    H_proj[L - 1] = H[L - 1]
    h_ln           = layer_norm(H[L - 1])
    logits[L - 1]  = U @ h_ln

    return H_proj, logits


def cosine_sim_matrix(X):
    """
    X : (L, d) float32 — rows are vectors.
    Returns (L, L) cosine-similarity matrix; diagonal = 1 by construction.
    """
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Xn    = X / norms
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def log_softmax_stable(logits_1d):
    """Numerically stable log-softmax for a 1-D logit array."""
    shifted = logits_1d - logits_1d.max()
    return shifted - np.log(np.sum(np.exp(shifted)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load excerpt data
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

# Check TunedLens fields are present
has_tl = all("tl_cross_entropy" in e and "tl_top_token_ids" in e for e in excerpts)
if not has_tl:
    print("  WARNING: tl_cross_entropy / tl_top_token_ids not found in excerpts.")
    print("           Sanity check will be skipped.")
    SANITY_CHECK_N = 0

n_layers_p1 = excerpts[0]["hidden_states"].shape[0]   # L
L           = n_layers_p1
print(f"  n_layers+1 = {L}  (n_translators = {n_translators})")

if n_translators != L - 1:
    raise RuntimeError(
        f"Mismatch: {n_translators} TunedLens translators but "
        f"{L - 1} non-final layers in hidden states. "
        "Check that --model matches the model used during data collection."
    )

# XXXXXX
# ── DEBUG: single-token end-to-end check ─────────────────────────────────
e = excerpts[0]
hs = e["hidden_states"].astype(np.float32)   # (L, seq_len, hidden_dim)
j  = 5   # arbitrary token position

H = hs[:, j, :]   # (L, hidden_dim)

# Check 1: does tuned_lens.forward() on the raw hidden state match stored tl_top?
import torch
stored_top_l0 = int(e["tl_top_token_ids"][0, j])
stored_ce_l0  = float(e["tl_cross_entropy"][0, j])
target_id     = int(e["input_ids"][j + 1])

h0_torch = torch.tensor(H[0]).unsqueeze(0).unsqueeze(0).half()  # (1, 1, hidden_dim) — float16
with torch.no_grad():
    tl_out = tuned_lens.forward(h0_torch, 0)             # (1, 1, vocab_size)
    lp     = tl_out.squeeze().log_softmax(-1).cpu().numpy()

our_top = int(np.argmax(lp))
our_ce  = float(-lp[target_id])

print(f"\n── DEBUG layer 0, token {j} ──")
print(f"  stored top token:  {stored_top_l0}   ours: {our_top}   match: {our_top == stored_top_l0}")
print(f"  stored CE:         {stored_ce_l0:.4f}   ours: {our_ce:.4f}   err: {abs(our_ce - stored_ce_l0):.5f}")

# Check 2: manually replicate W.T fix
W, b = translators[0]
h_proj_correct   = H[0] @ W.T + b          # PyTorch convention
h_proj_incorrect = W @ H[0] + b            # old (wrong) convention
h_ln_correct     = layer_norm(h_proj_correct)
logits_correct   = U @ h_ln_correct
logits_incorrect = U @ layer_norm(h_proj_incorrect)

print(f"\n  W.T argmax:  {int(np.argmax(logits_correct))}  (stored: {stored_top_l0})")
print(f"  W   argmax:  {int(np.argmax(logits_incorrect))}")

# Check 3: does tuned_lens.forward match W.T version?
tl_argmax = int(np.argmax(lp))
print(f"  tuned_lens.forward argmax: {tl_argmax}")
print(f"  W.T matches tuned_lens.forward: {int(np.argmax(logits_correct)) == tl_argmax}")

print(f"  H[0] dtype: {H[0].dtype}")
print(f"  W dtype:    {translators[0][0].dtype}")
print(f"  b dtype:    {translators[0][1].dtype}")
print(f"  ln_weight dtype: {ln_weight.dtype}")
print(f"  U dtype:    {U.dtype}")

# Also: what does tuned_lens.forward get as input dtype?
print(f"  h0_torch dtype: {h0_torch.dtype}")

# And check the raw stored hidden state value
raw_hs = e["hidden_states"]
print(f"  raw hidden_states dtype: {raw_hs.dtype}")
print(f"  raw H[0,5] norm: {np.linalg.norm(raw_hs[:, j, :][0]):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sanity check
# ─────────────────────────────────────────────────────────────────────────────
#
# What we're checking:
#   The stored tl_cross_entropy[l, j] and tl_top_token_ids[l, j] were
#   computed during data collection as:
#       lp = tuned_lens.forward(h_layer_l, l).log_softmax(-1)
#       tl_cross_entropy[l, j]  = -lp[target_id]
#       tl_top_token_ids[l, j]  = lp.argmax()
#   where target_id = input_ids[j+1]  (next-token target).
#
#   Our re-derivation: project_and_unembed(H[:, j, :]) → logits[l]
#   then: log_probs = log_softmax(logits[l])
#         our_ce    = -log_probs[target_id]
#         our_top   = argmax(logits[l])
#
#   These should agree within float16 rounding (hidden states stored as
#   float16, tl_cross_entropy stored as float16).

if SANITY_CHECK_N > 0:
    print(f"\nRunning sanity check on {SANITY_CHECK_N} sampled tokens …")

    rng = np.random.default_rng(42)

    # Build flat list of all valid (excerpt_idx, token_j) pairs
    all_pairs = [
        (ei, j)
        for ei, e in enumerate(excerpts)
        for j in range(e["token_losses"].shape[0])
    ]

    sample_size   = min(SANITY_CHECK_N, len(all_pairs))
    chosen_idxs   = rng.choice(len(all_pairs), size=sample_size, replace=False)
    sampled_pairs = [all_pairs[i] for i in chosen_idxs]

    top_token_matches = 0
    ce_errors         = []
    n_total           = 0

    for ei, j in tqdm(sampled_pairs, desc="Sanity check"):
        e          = excerpts[ei]
        hs         = e["hidden_states"].astype(np.float32)     # (L, seq_len, hidden_dim)
        input_ids  = e["input_ids"]                            # (seq_len_full,)
        stored_top = e["tl_top_token_ids"][:, j].astype(np.int32)    # (L,)
        stored_ce  = e["tl_cross_entropy"][:, j].astype(np.float32)  # (L,)

        H = hs[:, j, :]                        # (L, hidden_dim)
        _, logits_all = project_and_unembed(H) # (L, vocab_size)
        print(f"  project_and_unembed argmax l=0: {int(np.argmax(logits_all[0]))}")

        # Target: input_ids[j+1] is the next token at position j.
        # token_losses[j] = CE(model output at j, input_ids[j+1]), so this
        # is the same target used when storing tl_cross_entropy.
        target_id = int(input_ids[j + 1])

        for l in range(L):
            log_probs = log_softmax_stable(logits_all[l].astype(np.float64))
            our_top   = int(np.argmax(logits_all[l]))
            our_ce    = float(-log_probs[target_id])

            if our_top == stored_top[l]:
                top_token_matches += 1

            ce_errors.append(abs(our_ce - stored_ce[l]))
            n_total += 1

    top_match_pct = 100.0 * top_token_matches / n_total
    mean_ce_err   = float(np.mean(ce_errors))
    max_ce_err    = float(np.max(ce_errors))

    print(f"\n  ── Sanity check results ──────────────────────────────────────")
    print(f"  Sampled tokens:                  {len(sampled_pairs):,}")
    print(f"  (layer, token) pairs checked:    {n_total:,}")
    print(f"  Top-token argmax agreement:      {top_match_pct:.2f}%  (expected ≥ 99%)")
    print(f"  Mean |CE error| (nats):          {mean_ce_err:.5f}  (expected < 0.05)")
    print(f"  Max  |CE error| (nats):          {max_ce_err:.4f}")

    if top_match_pct < 95.0:
        print("\n  ⚠ WARNING: top-token agreement is unexpectedly low.")
        print("    Possible causes:")
        print("    - --model does not match the model used during data collection")
        print("    - Layer norm extracted from wrong submodule")
        print("    - Unembedding extracted from wrong submodule")
        print("    Check: model.gpt_neox.final_layer_norm  and  model.embed_out")
    elif mean_ce_err > 0.05:
        print("\n  ⚠ WARNING: CE errors are larger than expected.")
        print("    This may indicate a pipeline issue beyond float16 rounding.")
        print(f"    Top-token agreement ({top_match_pct:.1f}%) is the more reliable check.")
    else:
        print("\n  ✓ Sanity check passed.")

    print(f"  ─────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main accumulation loop
# ─────────────────────────────────────────────────────────────────────────────

print("Projecting hidden states and computing cosine-similarity matrices …")

all_losses        = []
all_cossims       = []   # hidden-state space  (L, L)
all_cossims_vocab = []   # vocab logit space   (L, L)

for e in tqdm(excerpts, desc="Excerpts"):
    hs     = e["hidden_states"].astype(np.float32)   # (L, seq_len, hidden_dim)
    losses = e["token_losses"].astype(np.float32)    # (seq_len-1,)

    seq_len = losses.shape[0]

    for j in range(seq_len):
        H_raw = hs[:, j, :]                          # (L, hidden_dim)

        H_proj, logits = project_and_unembed(H_raw)  # (L, hidden_dim), (L, vocab_size)

        # Apply layer norm to projected hidden states before cosine similarity.
        # The TunedLens translators map h_l into the pre-LN final-layer basis;
        # LN must be applied before comparing directions, for the same reason it
        # is applied before unembedding: it places every vector in the normalized
        # space where the geometry is semantically meaningful.
        H_proj_ln  = layer_norm_rows(H_proj)          # (L, hidden_dim)

        sim_hidden = cosine_sim_matrix(H_proj_ln)     # (L, L)
        sim_vocab  = cosine_sim_matrix(logits)        # (L, L)

        all_losses.append(losses[j])
        all_cossims.append(sim_hidden.astype(np.float16))
        all_cossims_vocab.append(sim_vocab.astype(np.float16))

all_losses        = np.array(all_losses,        dtype=np.float32)
all_cossims       = np.stack(all_cossims,       axis=0)   # (N, L, L) float16
all_cossims_vocab = np.stack(all_cossims_vocab, axis=0)   # (N, L, L) float16

print(f"  total tokens:      {len(all_losses):,}")
print(f"  cossims (hidden):  {all_cossims.shape}  dtype={all_cossims.dtype}")
print(f"  cossims (vocab):   {all_cossims_vocab.shape}  dtype={all_cossims_vocab.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save
# ─────────────────────────────────────────────────────────────────────────────

out_path = Path(OUTPUT_DIR) / "tuned_lens_cossims.npz"
np.savez_compressed(
    str(out_path),
    all_losses        = all_losses,
    all_cossims       = all_cossims,
    all_cossims_vocab = all_cossims_vocab,
    n_layers_p1       = np.array(n_layers_p1),
    model_name        = np.array(MODEL_NAME),
)
print(f"\nSaved → {out_path}")
print(f"  all_losses:         {all_losses.shape}")
print(f"  all_cossims:        {all_cossims.shape}")
print(f"  all_cossims_vocab:  {all_cossims_vocab.shape}")
