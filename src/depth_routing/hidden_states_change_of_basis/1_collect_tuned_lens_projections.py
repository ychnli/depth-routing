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
  3. Compute the (n_layers+1, n_layers+1) cosine-similarity matrix over
     the projected vectors.
  4. Accumulate losses + cosine-similarity matrices for downstream plotting.

Outputs (written to --output-dir):
  tuned_lens_cossims.npz   — arrays:
      all_losses   (N,)           float32  cross-entropy loss per token
      all_cossims  (N, L, L)      float16  per-token projected cosine-sim matrix
      n_layers_p1  scalar         int      L = n_transformer_layers + 1

Usage
-----
  python collect_tuned_lens_projections.py \\
      --data-dir  token_evolution_data \\
      --output-dir tuned_lens_projections \\
      --model      EleutherAI/pythia-160m-deduped \\
      --device     cpu

Notes
-----
  • Requires hidden states saved from running model inference
  • TunedLens weights are downloaded automatically from HuggingFace the
    first time; they are cached by the transformers/huggingface_hub library.
  • The final layer has no TunedLens translator (it IS the target basis),
    so its hidden state is passed through unchanged.
  • Memory: cosine-sim matrices are stored as float16 to keep the output
    file small.  All arithmetic is done in float32.
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
parser.add_argument("--data-dir",   default="token_evolution_data",
                    help="Root directory of .npz excerpt files (default: token_evolution_data)")
parser.add_argument("--output-dir", default="tuned_lens_projections",
                    help="Directory for output .npz file (default: tuned_lens_projections)")
parser.add_argument("--model",      default="EleutherAI/pythia-160m-deduped",
                    help="HuggingFace model identifier matching the saved hidden states")
parser.add_argument("--device",     default="cpu",
                    help="Torch device for TunedLens forward pass (default: cpu)")
args = parser.parse_args()

DATA_DIR   = args.data_dir
OUTPUT_DIR = args.output_dir
MODEL_NAME = args.model
DEVICE     = args.device
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# 1.  Load TunedLens affine mapping translators
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading model {MODEL_NAME} (needed to locate TunedLens weights) …")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

print("Loading TunedLens probes …")
from tuned_lens.nn.lenses import TunedLens
tuned_lens = TunedLens.from_model_and_pretrained(model, map_location=DEVICE).to(DEVICE)
tuned_lens.eval()

# Extract W and b for each translator as plain numpy arrays (float32).
# tuned_lens.unembed is the shared unembedding; the per-layer translators
# live in tuned_lens[l] and expose .weight (linear) and .bias.
#
# Each translator maps  hidden_dim → hidden_dim  (same space; the unembedding
# is applied separately).  We just want the affine map itself.

n_translators = len(tuned_lens)          # one per non-final layer
print(f"  {n_translators} translators found")

translators = []   # list of (W, b) tuples, one per layer 0..n_translators-1
for l in range(n_translators):
    t = tuned_lens[l]
    # The translator is an nn.Linear (or similar); weight shape: (hidden_dim, hidden_dim)
    W = t.weight.detach().cpu().float().numpy()   # (hidden_dim, hidden_dim)
    b = t.bias.detach().cpu().float().numpy()     # (hidden_dim,)
    translators.append((W, b))

print(f"  translator weight shape: {translators[0][0].shape}")
del model   # free memory — we only needed it to load the lens weights


# 2.  Load excerpt hidden states
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

n_layers_p1 = excerpts[0]["hidden_states"].shape[0]   # L
L           = n_layers_p1
print(f"  n_layers+1 = {L}  (n_translators = {n_translators})")

if n_translators != L - 1:
    raise RuntimeError(
        f"Mismatch: {n_translators} TunedLens translators but "
        f"{L - 1} non-final layers in hidden states. "
        "Check that --model matches the model used for data collection."
    )


# 3.  Project hidden states and compute cosine-similarity matrices
# ─────────────────────────────────────────────────────────────────────────────

def project(H):
    """
    Apply TunedLens affine translators to every layer of a single token.

    Parameters
    ----------
    H : (L, hidden_dim) float32
        Raw hidden states for one token, across all layers.

    Returns
    -------
    H_proj : (L, hidden_dim) float32
        Projected hidden states, all in the final layer's basis.
        Layer l = L-1 (the final layer) is returned unchanged.
    """
    H_proj = np.empty_like(H)
    for l in range(L - 1):
        W, b = translators[l]
        H_proj[l] = W @ H[l] + b      # affine map → final-layer basis
    H_proj[L - 1] = H[L - 1]          # final layer: identity
    return H_proj


def cosine_sim_matrix(H):
    """
    H : (L, hidden_dim) float32 — already projected.
    Returns (L, L) cosine-similarity matrix (full, diagonal = 1).
    """
    norms = np.linalg.norm(H, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    Hn    = H / norms
    sim   = np.clip(Hn @ Hn.T, -1.0, 1.0)
    return sim


print("\nProjecting hidden states and computing cosine-similarity matrices …")

all_losses  = []
all_cossims = []

for e in tqdm(excerpts, desc="Excerpts"):
    hs     = e["hidden_states"].astype(np.float32)   # (L, seq_len, hidden_dim)
    losses = e["token_losses"].astype(np.float32)    # (seq_len-1,)

    seq_len = losses.shape[0]

    for j in range(seq_len):
        H_raw  = hs[:, j, :]            # (L, hidden_dim)
        H_proj = project(H_raw)         # (L, hidden_dim) — in final-layer basis
        sim    = cosine_sim_matrix(H_proj)

        all_losses.append(losses[j])
        all_cossims.append(sim.astype(np.float16))   # store as float16

all_losses  = np.array(all_losses,  dtype=np.float32)
all_cossims = np.stack(all_cossims, axis=0)           # (N, L, L) float16

print(f"  total tokens: {len(all_losses):,}")
print(f"  cossims array: {all_cossims.shape}  dtype={all_cossims.dtype}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Save
# ─────────────────────────────────────────────────────────────────────────────

out_path = Path(OUTPUT_DIR) / "tuned_lens_cossims.npz"
np.savez_compressed(
    str(out_path),
    all_losses   = all_losses,
    all_cossims  = all_cossims,
    n_layers_p1  = np.array(n_layers_p1),
    model_name   = np.array(MODEL_NAME),
)
print(f"\nSaved → {out_path}")
print(f"  all_losses:  {all_losses.shape}")
print(f"  all_cossims: {all_cossims.shape}")