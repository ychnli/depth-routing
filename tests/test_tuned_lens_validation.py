"""
Validate TunedLens results from token_evolution_data_collect.py against the
TunedLens PredictionTrajectory API (ground truth).

Runs collect_token_evolution_data with --enable-tuned-lens on a small set of
WikiText excerpts (batch_size=1, CPU for determinism), then compares the saved
cross-entropy, entropy, forward KL, and top-token predictions against what
PredictionTrajectory.from_lens_and_model produces on the same inputs.

Usage:
    uv run python scripts/test_tuned_lens_validation.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tuned_lens.nn.lenses import TunedLens
from tuned_lens.plotting import PredictionTrajectory

from depth_routing.token_evolution_data_collect import (
    collect_token_evolution_data,
    load_excerpt,
    list_excerpt_ids,
    is_valid,
    passes_length_filter,
)

MODEL_NAME = "EleutherAI/pythia-160m-deduped"
DEVICE = "cpu"  # Force CPU for deterministic results
MAX_SEQ_LEN = 128  # Short sequences for fast testing
NUM_EXCERPTS = 4
ATOL = 1e-4  # Absolute tolerance for float comparisons


def get_test_texts(tokenizer, num=NUM_EXCERPTS):
    """Get a small set of test texts from WikiText."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if is_valid(x["text"])]
    texts = [
        t for t in texts
        if passes_length_filter(t, tokenizer, min_tokens=32)
    ]
    texts = [
        tokenizer.decode(tokenizer(t)["input_ids"][:MAX_SEQ_LEN])
        for t in texts[:num]
    ]
    return texts


def run_api_reference(model, tokenizer, tuned_lens, texts):
    """Run PredictionTrajectory.from_lens_and_model as ground truth."""
    results = []
    for text in texts:
        input_ids = tokenizer.encode(text)
        input_ids = input_ids[:MAX_SEQ_LEN]
        targets = input_ids[1:] + [tokenizer.eos_token_id or 0]

        with torch.no_grad():
            traj = PredictionTrajectory.from_lens_and_model(
                tuned_lens, model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                targets=targets,
            )

        results.append({
            "input_ids": input_ids,
            "cross_entropy": traj.cross_entropy().stats,   # (n_layers+1, seq_len)
            "entropy": traj.entropy().stats,               # (n_layers+1, seq_len)
            "forward_kl": traj.forward_kl().stats,         # (n_layers+1, seq_len)
            "top_token_ids": np.argmax(traj.log_probs, axis=-1).astype(np.int32),
        })
    return results


def compare_array(name, saved, api, atol=ATOL):
    """Compare two arrays, print diagnostics, return True if pass."""
    saved = saved.astype(np.float32)
    api = api.astype(np.float32)

    if saved.shape != api.shape:
        print(f"  {name}: FAIL — shape mismatch: saved={saved.shape} api={api.shape}")
        return False

    diff = np.abs(saved - api)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    passed = max_diff < atol

    status = "PASS" if passed else "FAIL"
    print(f"  {name:20s}  max|diff|={max_diff:.2e}  mean|diff|={mean_diff:.2e}  [{status}]")

    if not passed:
        # Per-layer breakdown
        for l in range(saved.shape[0]):
            layer_max = float(diff[l].max())
            layer_mean = float(diff[l].mean())
            print(f"    layer {l:2d}: max={layer_max:.2e}  mean={layer_mean:.2e}")

    return passed


def main():
    print(f"Model:        {MODEL_NAME}")
    print(f"Device:       {DEVICE}")
    print(f"Max seq len:  {MAX_SEQ_LEN}")
    print(f"Num excerpts: {NUM_EXCERPTS}")
    print(f"Tolerance:    {ATOL}")
    print()

    # Load model, tokenizer, TunedLens once (shared between both methods)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    tuned_lens = TunedLens.from_model_and_pretrained(
        model, map_location=DEVICE,
    ).to(DEVICE)

    # Get test texts
    texts = get_test_texts(tokenizer, NUM_EXCERPTS)
    print(f"Test texts:   {len(texts)} excerpts")
    for i, t in enumerate(texts):
        n_tok = len(tokenizer(t)["input_ids"])
        print(f"  [{i}] {n_tok} tokens: {t[:60]}...")
    print()

    # --- Ground truth: PredictionTrajectory API ---
    print("=" * 70)
    print("Running PredictionTrajectory.from_lens_and_model (ground truth)...")
    print("=" * 70)
    api_results = run_api_reference(model, tokenizer, tuned_lens, texts)
    print(f"API produced {len(api_results)} results\n")

    # --- Our implementation: collect_token_evolution_data ---
    print("=" * 70)
    print("Running collect_token_evolution_data (batch_size=1, enable_tuned_lens)...")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        collect_token_evolution_data(
            texts,
            output_dir=tmpdir,
            model_name=MODEL_NAME,
            device=DEVICE,
            batch_size=1,           # No padding — eliminates batching artifacts
            verbose=True,
            enable_tuned_lens=True,
            save_hidden_states=True,
            enable_generation=False,
        )

        excerpt_ids = list_excerpt_ids(tmpdir)
        print(f"\nSaved {len(excerpt_ids)} excerpts to {tmpdir}\n")

        # --- Comparison ---
        print("=" * 70)
        print("Comparing saved results vs API ground truth")
        print("=" * 70)

        all_passed = True

        for idx, eid in enumerate(excerpt_ids):
            saved = load_excerpt(tmpdir, eid)
            api = api_results[idx]

            seq_len = len(api["input_ids"])
            print(f"\n--- Excerpt {eid} ({seq_len} tokens) ---")

            # 1. Input IDs must match exactly
            saved_ids = saved["input_ids"].tolist()
            api_ids = api["input_ids"]
            if saved_ids != api_ids:
                print(f"  FAIL: input_ids mismatch!")
                print(f"    saved: {saved_ids[:10]}...")
                print(f"    api:   {api_ids[:10]}...")
                all_passed = False
                continue
            print(f"  input_ids:   MATCH ({len(saved_ids)} tokens)")

            # 2. Cross-entropy
            if not compare_array(
                "cross_entropy",
                saved["tl_cross_entropy"],
                api["cross_entropy"],
            ):
                all_passed = False

            # 3. Entropy
            if not compare_array(
                "entropy",
                saved["tl_entropy"],
                api["entropy"],
            ):
                all_passed = False

            # 4. Forward KL
            if not compare_array(
                "forward_kl",
                saved["tl_forward_kl"],
                api["forward_kl"],
            ):
                all_passed = False

            # 5. Top token IDs (exact match)
            saved_top = saved["tl_top_token_ids"]
            api_top = api["top_token_ids"]
            match_pct = 100.0 * float((saved_top == api_top).mean())
            top_pass = match_pct == 100.0
            status = "PASS" if top_pass else "FAIL"
            print(f"  {'top_token_ids':20s}  {match_pct:.1f}% agreement  [{status}]")
            if not top_pass:
                all_passed = False
                for l in range(saved_top.shape[0]):
                    layer_pct = 100.0 * float((saved_top[l] == api_top[l]).mean())
                    if layer_pct < 100.0:
                        print(f"    layer {l:2d}: {layer_pct:.1f}%")

            # 6. Hidden states: compare against a fresh unbatched forward pass
            if "hidden_states" in saved:
                input_th = torch.tensor(
                    api_ids, dtype=torch.long, device=DEVICE,
                ).unsqueeze(0)
                with torch.no_grad():
                    out = model(input_th, output_hidden_states=True)
                api_hs = (
                    torch.stack(list(out.hidden_states))
                    .squeeze(1)
                    .cpu()
                    .float()
                    .numpy()
                )
                if not compare_array(
                    "hidden_states",
                    saved["hidden_states"],
                    api_hs,
                ):
                    all_passed = False

    # --- Summary ---
    print()
    print("=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED")
        print("=" * 70)
    else:
        print("SOME CHECKS FAILED — see details above")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
