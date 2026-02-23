"""
Runs inference on a transformer language model over text excerpts, collecting
per-token loss, hidden states, and (optionally) TunedLens trajectory metrics
(cross-entropy, entropy, forward KL across all layers). Can also run
autoregressive generation with BERTScore evaluation.
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

MODEL_NAME = "EleutherAI/pythia-160m-deduped"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT_LEN = 64
MIN_TOTAL_LEN = 100
BATCH_SIZE = 4
OUTPUT_DIR = "token_evolution_data"
MAX_SEQ_LEN = 512

EXCERPT_FIELDS = [
    "excerpt_id", "prompt", "reference", "generated",
    "avg_loss", "perplexity", "bertscore",
    "forward_pass_ms", "generation_ms", "tokens_generated", "tokens_per_sec",
]
TOKEN_FIELDS = [
    "excerpt_id", "token_idx",
    "true_token_id", "true_token_text",
    "pred_token_id", "pred_token_text",
    "loss",
]

def is_valid(text):
    """
    Return True if the excerpt passes heuristic quality filters.

    Rejects empty lines, section headings, and texts with excessive
    punctuation, numeric tokens, or capitalisation.
    """
    text = text.strip()
    if len(text) == 0 or text.startswith("="):
        return False
    tokens = text.split()
    if len(tokens) < 50:
        return False
    punct_count = text.count(",") + text.count(";")
    if punct_count / len(tokens) > 0.15:
        return False
    num_count = sum(1 for w in tokens if any(c.isdigit() for c in w))
    if num_count / len(tokens) > 0.2:
        return False
    cap_count = sum(1 for w in tokens if w[0].isupper())
    if cap_count / len(tokens) > 0.4:
        return False
    return True

def passes_length_filter(text, tokenizer, min_tokens=MIN_TOTAL_LEN):
    """Return True if the tokenised text meets the minimum token count."""
    tokens = tokenizer(text)["input_ids"]
    return len(tokens) >= min_tokens


# ---------------------------------------------------------------------------
# TunedLens helpers
# ---------------------------------------------------------------------------

def _compute_tuned_lens_metrics(tuned_lens, hidden_states, model_logits, targets):
    """
    Applies the TunedLens probe to each intermediate hidden-state layer and
    computes cross-entropy, entropy, forward KL divergence (relative to the
    final layer), and the top predicted token ID.

    Parameters
    ----------
    tuned_lens :
        A fitted ``TunedLens`` instance compatible with the model.
    hidden_states : sequence of Tensor
        Per-layer hidden-state tensors of shape ``(1, seq_len, hidden_dim)``
        from a model forward pass with ``output_hidden_states=True``.
    model_logits : Tensor
        Final-layer logits, shape ``(1, seq_len, vocab_size)``.
    targets : array-like
        Target token IDs of length ``seq_len`` (typically the next-token
        targets for each position).

    Returns
    -------
    dict
        ``cross_entropy``  — ``(n_layers+1, seq_len)`` float16
        ``entropy``        — ``(n_layers+1, seq_len)`` float16
        ``forward_kl``     — ``(n_layers+1, seq_len)`` float16
        ``top_token_ids``  — ``(n_layers+1, seq_len)`` int32
    """
    device = model_logits.device
    seq_len = model_logits.shape[1]
    targets_th = torch.as_tensor(targets, device=device, dtype=torch.long)

    # Final-layer (model) log-probabilities are 
    model_lp = model_logits.squeeze(0).log_softmax(-1)  # (seq_len, V)
    model_p = model_lp.exp()

    pos_idx = torch.arange(seq_len, device=device)
    all_ce, all_ent, all_fkl, all_top = [], [], [], []

    # Lens layers: hidden_states[0] ... hidden_states[-2]
    for i, h in enumerate(hidden_states[:-1]):
        lp = tuned_lens.forward(h, i).squeeze(0).log_softmax(-1)  # (seq_len, V)
        p = lp.exp()

        all_ce.append((-lp[pos_idx, targets_th]).detach().cpu().float().numpy())
        all_ent.append((-(p * lp).sum(-1)).detach().cpu().float().numpy())
        all_fkl.append(((model_p * (model_lp - lp)).sum(-1)).detach().cpu().float().numpy())
        all_top.append(lp.argmax(-1).detach().cpu().numpy())

        del lp, p  # free accelerator memory

    # Final layer (true model predictions)
    all_ce.append((-model_lp[pos_idx, targets_th]).detach().cpu().float().numpy())
    all_ent.append((-(model_p * model_lp).sum(-1)).detach().cpu().float().numpy())
    all_fkl.append(np.zeros(seq_len, dtype=np.float32)) # final layer KL is zero by definition
    all_top.append(model_logits.squeeze(0).argmax(-1).detach().cpu().numpy())

    return {
        "cross_entropy": np.array(all_ce, dtype=np.float16),
        "entropy": np.array(all_ent, dtype=np.float16),
        "forward_kl": np.array(all_fkl, dtype=np.float16),
        "top_token_ids": np.array(all_top, dtype=np.int32),
    }


_EXCERPTS_SUBDIR = "excerpts"


def save_excerpt(output_dir: str | Path, excerpt_id: int, data: dict) -> Path:
    """Save all data for a single excerpt as a compressed ``.npz`` archive.

    Parameters
    ----------
    output_dir :
        Root output directory (the ``excerpts/`` subdirectory is created
        automatically).
    excerpt_id :
        Integer ID for this excerpt.
    data :
        Dictionary of arrays / scalars to save.  Expected keys (all optional
        except ``input_ids``):

        Always present:
            ``input_ids`` (seq_len,), ``token_strings`` (seq_len,),
            ``token_losses`` (seq_len-1,), ``pred_token_ids`` (seq_len-1,),
            ``model_name``, ``avg_loss``, ``perplexity``,
            ``forward_pass_ms``.

        TunedLens (when enabled):
            ``layer_labels`` (n_layers+1,),
            ``tl_cross_entropy`` (n_layers+1, seq_len),
            ``tl_entropy`` (n_layers+1, seq_len),
            ``tl_forward_kl`` (n_layers+1, seq_len),
            ``tl_top_token_ids`` (n_layers+1, seq_len).

        Hidden states (when enabled):
            ``hidden_states`` (n_layers+1, seq_len, hidden_dim).

        Generation (when enabled):
            ``prompt_text``, ``reference_text``, ``generated_text``,
            ``bertscore``, ``generation_ms``, ``tokens_generated``,
            ``tokens_per_sec``.

    Returns
    -------
    Path
        The path to the saved ``.npz`` file.
    """
    excerpts_dir = Path(output_dir) / _EXCERPTS_SUBDIR
    excerpts_dir.mkdir(parents=True, exist_ok=True)
    path = excerpts_dir / f"excerpt_{excerpt_id:04d}.npz"

    arrays: dict[str, np.ndarray] = {}
    for key, val in data.items():
        if val is None:
            continue
        arrays[key] = np.asarray(val) if not isinstance(val, np.ndarray) else val

    np.savez_compressed(str(path), **arrays)
    return path


def load_excerpt(output_dir: str | Path, excerpt_id: int) -> dict:
    """Load all saved data for a single excerpt.

    Returns a dict whose keys mirror those written by :func:`save_excerpt`.
    Scalar values (e.g. ``avg_loss``) are returned as Python floats/ints;
    string values as Python strings; arrays as NumPy arrays.

    Example
    -------
    >>> data = load_excerpt("output_dir", 42)
    >>> data["tl_cross_entropy"]   # (n_layers+1, seq_len)  — TunedLens CE
    >>> data["hidden_states"]      # (n_layers+1, seq_len, hidden_dim)
    >>> data["token_losses"]       # (seq_len-1,) — per-token loss
    >>> data["input_ids"]          # (seq_len,)
    """
    path = Path(output_dir) / _EXCERPTS_SUBDIR / f"excerpt_{excerpt_id:04d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No excerpt file at {path}")
    raw = np.load(str(path), allow_pickle=True)

    result: dict = {}
    for key in raw.files:
        arr = raw[key]
        # 0-d arrays → scalars
        if arr.ndim == 0:
            val = arr.item()
            result[key] = val
        else:
            result[key] = arr
    raw.close()
    return result


def list_excerpt_ids(output_dir: str | Path) -> list[int]:
    """Return sorted list of excerpt IDs available in *output_dir*.

    Looks for ``excerpts/excerpt_NNNN.npz`` files.
    """
    excerpts_dir = Path(output_dir) / _EXCERPTS_SUBDIR
    if not excerpts_dir.is_dir():
        return []
    ids = sorted(
        int(p.stem.split("_", 1)[1])
        for p in excerpts_dir.glob("excerpt_*.npz")
    )
    return ids


def load_excerpts(
    output_dir: str | Path,
    excerpt_ids: list[int] | None = None,
) -> list[dict]:
    """Load data for multiple excerpts.

    Parameters
    ----------
    output_dir :
        Root output directory.
    excerpt_ids :
        Specific IDs to load.  If ``None``, loads all available excerpts.

    Returns
    -------
    list[dict]
        One dict per excerpt (same format as :func:`load_excerpt`).
    """
    if excerpt_ids is None:
        excerpt_ids = list_excerpt_ids(output_dir)
    return [load_excerpt(output_dir, eid) for eid in excerpt_ids]


def load_tuned_lens_from_excerpts(
    output_dir: str | Path,
    excerpt_ids: list[int] | None = None,
) -> tuple[list[dict], dict]:
    """Load TunedLens trajectories from per-excerpt files.

    Returns ``(trajectories, meta)`` in the same format as
    ``tuned_lens_analysis.load_trajectories``, so the result can be passed
    directly to any of the visualisation functions in that module.

    Parameters
    ----------
    output_dir :
        Root output directory containing ``excerpts/`` subdirectory.
    excerpt_ids :
        Specific IDs to load.  If ``None``, loads all available excerpts.

    Returns
    -------
    trajectories : list[dict]
        Each dict has keys ``index``, ``input_ids``, ``token_strings``,
        ``cross_entropy``, ``entropy``, ``forward_kl``, ``top_token_ids``.
    meta : dict
        ``model_name`` and ``layer_labels``.

    Raises
    ------
    ValueError
        If no TunedLens data is found in the loaded excerpts.
    """
    all_data = load_excerpts(output_dir, excerpt_ids)

    trajectories: list[dict] = []
    model_name: str | None = None
    layer_labels: list[str] | None = None

    for i, d in enumerate(all_data):
        if "tl_cross_entropy" not in d:
            continue
        eid = excerpt_ids[i] if excerpt_ids else i
        trajectories.append({
            "index": eid,
            "input_ids": d["input_ids"].tolist() if isinstance(d["input_ids"], np.ndarray) else d["input_ids"],
            "token_strings": d["token_strings"].tolist() if isinstance(d["token_strings"], np.ndarray) else d["token_strings"],
            "cross_entropy": d["tl_cross_entropy"],
            "entropy": d["tl_entropy"],
            "forward_kl": d["tl_forward_kl"],
            "top_token_ids": d["tl_top_token_ids"],
        })
        if model_name is None:
            model_name = d.get("model_name", "")
        if layer_labels is None and "layer_labels" in d:
            ll = d["layer_labels"]
            layer_labels = ll.tolist() if isinstance(ll, np.ndarray) else list(ll)

    if not trajectories:
        raise ValueError(
            f"No TunedLens data found in {output_dir}. "
            "Was enable_tuned_lens=True during collection?"
        )

    meta = {
        "model_name": model_name or "",
        "layer_labels": layer_labels or [],
    }
    logger.info(
        "Loaded %d TunedLens trajectories from per-excerpt files in %s",
        len(trajectories), output_dir,
    )
    return trajectories, meta


def load_token_level_df(
    output_dir: str | Path,
    excerpt_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Assemble a token-level DataFrame from per-excerpt ``.npz`` files.

    The returned DataFrame has the same columns as the legacy
    ``token_level.csv`` (``excerpt_id``, ``token_idx``, ``true_token_id``,
    ``true_token_text``, ``pred_token_id``, ``pred_token_text``, ``loss``)
    and can be used identically.

    Parameters
    ----------
    output_dir :
        Root output directory.
    excerpt_ids :
        Specific IDs to load.  If ``None``, loads all available excerpts.
    """
    if excerpt_ids is None:
        excerpt_ids = list_excerpt_ids(output_dir)

    rows: list[dict] = []
    for eid in excerpt_ids:
        d = load_excerpt(output_dir, eid)
        input_ids = np.asarray(d["input_ids"])
        token_strings = np.asarray(d["token_strings"]) if "token_strings" in d else None
        losses = np.asarray(d["token_losses"])
        preds = np.asarray(d["pred_token_ids"])
        for j in range(len(losses)):
            rows.append({
                "excerpt_id": eid,
                "token_idx": j + 1,
                "true_token_id": int(input_ids[j + 1]),
                "true_token_text": (
                    str(token_strings[j + 1]) if token_strings is not None else ""
                ),
                "pred_token_id": int(preds[j]),
                "pred_token_text": "",  # would require tokenizer to decode
                "loss": float(losses[j]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

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
    enable_generation: bool = True,
):
    """
    Run a full data-collection pass over the provided text excerpts.

    Parameters
    ----------
    texts :
        Pre-processed list of text strings to use as excerpts.
    output_dir :
        Directory in which to write output files.
    model_name :
        HuggingFace model identifier to load.
    device :
        Torch device string (e.g. ``"mps"``, ``"cuda"``, ``"cpu"``).
    prompt_len :
        Number of tokens to use as the prompt; the remainder becomes the
        reference continuation (used only when *enable_generation* is True).
    batch_size :
        Number of excerpts to process in each forward-pass batch.
    verbose :
        Whether to print progress information during the run.
    enable_tuned_lens :
        If True, load TunedLens probes and compute per-layer cross-entropy,
        entropy, forward KL, and top-token predictions from the hidden states
        that already come out of the single forward pass.  No additional model
        forward pass is performed.  Results are stored in each per-excerpt
        ``.npz`` file and can be loaded via
        :func:`load_tuned_lens_from_excerpts`.
    save_hidden_states :
        If True, include raw per-layer hidden states (float16) in each
        per-excerpt ``.npz`` file.  Disabled by default because hidden states
        are large (``n_layers × seq_len × hidden_dim × 2`` bytes each).
    enable_generation :
        If True (default), run autoregressive generation from the prompt
        prefix, compute BERTScore against the reference, and record
        generation timing.  Set to False to skip the slow generation step.

    Outputs
    -------
    Per-excerpt files (primary format):
        ``excerpts/excerpt_{id:04d}.npz`` — self-contained archive per
        excerpt.  Always contains: ``input_ids``, ``token_strings``,
        ``token_losses``, ``pred_token_ids``, ``model_name``, ``avg_loss``,
        ``perplexity``, ``forward_pass_ms``.  Conditionally contains
        TunedLens arrays (``tl_cross_entropy``, ``tl_entropy``,
        ``tl_forward_kl``, ``tl_top_token_ids``, ``layer_labels``), hidden
        states (``hidden_states``), and generation results
        (``prompt_text``, ``reference_text``, ``generated_text``,
        ``bertscore``, ``generation_ms``, ``tokens_generated``,
        ``tokens_per_sec``).

    Lightweight summary:
        ``excerpt_level.csv`` — one row per excerpt with scalar metrics and
        generation text (if enabled).

    Loading helpers:
        Use :func:`load_excerpt`, :func:`load_excerpts`,
        :func:`load_tuned_lens_from_excerpts`, and
        :func:`load_token_level_df` to read the results back.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MAX_CTX = model.config.max_position_embeddings

    tuned_lens = None
    if enable_tuned_lens:
        from tuned_lens.nn.lenses import TunedLens
        tuned_lens = TunedLens.from_model_and_pretrained(
            model, map_location=device,
        ).to(device)
        logger.info("TunedLens probes loaded for %s", model_name)

    if verbose:
        print(f"Total excerpts to process: {len(texts)}")
        if enable_tuned_lens:
            print("TunedLens depth analysis: enabled")
        if not enable_generation:
            print("Autoregressive generation: disabled")
        if save_hidden_states:
            print("Raw hidden-state saving: enabled")

    def sync_and_time():
        """Synchronise the active accelerator and return the current wall time."""
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    EXCERPT_CSV = os.path.join(output_dir, "excerpt_level.csv")

    # Overwrite any existing output files with fresh headers.
    pd.DataFrame(columns=EXCERPT_FIELDS).to_csv(EXCERPT_CSV, index=False)

    excerpt_id = 0
    tl_layer_labels: list[str] | None = None

    for batch_start in tqdm(
        range(0, len(texts), batch_size),
        desc="Processing batches",
        disable=not verbose,
    ):
        batch_texts = texts[batch_start:batch_start + batch_size]

        batch_token_lists = []
        batch_prompt_tokens = []
        batch_reference_tokens = []
        batch_prompt_texts = []
        batch_reference_texts = []

        for text in batch_texts:
            tokens = tokenizer(text)["input_ids"]
            prompt_tokens = tokens[:prompt_len]
            reference_tokens = tokens[prompt_len:]
            full_tokens = tokens[:MAX_CTX]
            batch_token_lists.append(torch.tensor(full_tokens))
            batch_prompt_tokens.append(prompt_tokens)
            batch_reference_tokens.append(reference_tokens)
            batch_prompt_texts.append(tokenizer.decode(prompt_tokens))
            batch_reference_texts.append(tokenizer.decode(reference_tokens))

        if not batch_token_lists:
            continue

        seq_lens = [len(t) for t in batch_token_lists]
        max_len = max(seq_lens)

        input_ids_list = []
        attention_mask_list = []

        for t in batch_token_lists:
            pad_len = max_len - len(t)
            input_ids_list.append(torch.cat([t, torch.full((pad_len,), tokenizer.pad_token_id)]))
            attention_mask_list.append(torch.cat([torch.ones(len(t)), torch.zeros(pad_len)]))

        input_ids = torch.stack(input_ids_list).to(device)
        attention_mask = torch.stack(attention_mask_list).to(device)

        cur_batch_size, seq_len = input_ids.shape

        t_forward_start = sync_and_time()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        t_forward_end = sync_and_time()
        forward_ms = (t_forward_end - t_forward_start) * 1000

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = losses.view(cur_batch_size, -1)

        for i in range(cur_batch_size):
            actual_len = seq_lens[i]

            # ---- Build the per-excerpt data dict ---------------------------
            ids_np = input_ids[i, :actual_len].cpu().numpy().astype(np.int64)
            tok_strs = np.array(
                [tokenizer.decode([int(tid)]) for tid in ids_np], dtype=object,
            )
            losses_np = token_losses[i, :actual_len - 1].cpu().float().numpy()
            preds_np = logits[i, :actual_len - 1, :].argmax(-1).cpu().numpy().astype(np.int64)

            avg_loss = float(losses_np.mean())
            ppl = float(np.exp(avg_loss))

            excerpt_data: dict = {
                "model_name": model_name,
                "input_ids": ids_np,
                "token_strings": tok_strs,
                "token_losses": losses_np.astype(np.float32),
                "pred_token_ids": preds_np,
                "avg_loss": avg_loss,
                "perplexity": ppl,
                "forward_pass_ms": forward_ms,
            }

            # ---- Hidden states (optional — stored as float16) --------------
            if save_hidden_states:
                hs = torch.stack(
                    [h[i, :actual_len, :] for h in hidden_states]
                ).cpu().half().numpy()   # (n_layers+1, seq_len, hidden_dim)
                excerpt_data["hidden_states"] = hs

            # ---- TunedLens trajectory (optional) ---------------------------
            if tuned_lens is not None:
                # Next-token targets (same convention as tuned_lens_analysis)
                target_ids = input_ids[i, 1:actual_len].cpu().tolist()
                target_ids.append(tokenizer.eos_token_id or 0)

                # Extract this sample's unpadded hidden states (still on device)
                sample_hidden = [hs_layer[i:i + 1, :actual_len, :] for hs_layer in hidden_states]
                sample_logits = logits[i:i + 1, :actual_len, :]

                tl_metrics = _compute_tuned_lens_metrics(
                    tuned_lens, sample_hidden, sample_logits, target_ids,
                )

                if tl_layer_labels is None:
                    n = tl_metrics["cross_entropy"].shape[0]
                    tl_layer_labels = [f"layer_{j}" for j in range(n - 1)] + ["output"]

                # Store in per-excerpt data (prefixed with tl_)
                excerpt_data["layer_labels"] = np.array(tl_layer_labels)
                excerpt_data["tl_cross_entropy"] = tl_metrics["cross_entropy"]
                excerpt_data["tl_entropy"] = tl_metrics["entropy"]
                excerpt_data["tl_forward_kl"] = tl_metrics["forward_kl"]
                excerpt_data["tl_top_token_ids"] = tl_metrics["top_token_ids"]

            # ---- Generation and BERTScore (optional) -----------------------
            generated_text = None
            bertscore = None
            generation_ms = 0.0
            tokens_generated = 0
            tokens_per_sec = 0.0

            if enable_generation:
                prompt_ids = torch.tensor(batch_prompt_tokens[i]).unsqueeze(0).to(device)
                prompt_mask = torch.ones_like(prompt_ids).to(device)
                reference_tokens_i = batch_reference_tokens[i]
                reference_text_i = batch_reference_texts[i]

                gen_len = min(len(reference_tokens_i), MAX_CTX - prompt_len)

                t_gen_start = sync_and_time()

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=gen_len,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                t_gen_end = sync_and_time()
                generation_ms = (t_gen_end - t_gen_start) * 1000
                tokens_generated = generated_ids.shape[1] - prompt_ids.shape[1]
                tokens_per_sec = (
                    tokens_generated / (generation_ms / 1000)
                    if generation_ms > 0 else 0.0
                )

                generated_new = generated_ids[0][len(batch_prompt_tokens[i]):]
                generated_text = tokenizer.decode(generated_new, skip_special_tokens=True)

                try:
                    from bert_score import score as bert_score_fn
                    P, R, F1 = bert_score_fn(
                        [generated_text], [reference_text_i[:1000]],
                        lang="en", verbose=False,
                    )
                    bertscore = F1.item()
                except Exception as e:
                    if verbose:
                        tqdm.write(f"BERTScore failed for excerpt {excerpt_id}: {e}")

                excerpt_data["prompt_text"] = batch_prompt_texts[i]
                excerpt_data["reference_text"] = reference_text_i
                excerpt_data["generated_text"] = generated_text or ""
                excerpt_data["bertscore"] = bertscore if bertscore is not None else float("nan")
                excerpt_data["generation_ms"] = generation_ms
                excerpt_data["tokens_generated"] = tokens_generated
                excerpt_data["tokens_per_sec"] = tokens_per_sec

            # ---- Save per-excerpt .npz (primary format) --------------------
            save_excerpt(output_dir, excerpt_id, excerpt_data)

            # ---- Append lightweight summary row to CSV ---------------------
            pd.DataFrame([{
                "excerpt_id": excerpt_id,
                "prompt": batch_prompt_texts[i],
                "reference": batch_reference_texts[i] if enable_generation else "",
                "generated": generated_text or "",
                "avg_loss": avg_loss,
                "perplexity": ppl,
                "bertscore": bertscore,
                "forward_pass_ms": forward_ms,
                "generation_ms": generation_ms,
                "tokens_generated": tokens_generated,
                "tokens_per_sec": tokens_per_sec,
            }]).to_csv(EXCERPT_CSV, mode='a', header=False, index=False)

            if verbose:
                bertscore_str = f"{bertscore:.4f}" if bertscore is not None else "N/A"
                parts = [
                    f"Excerpt {excerpt_id}",
                    f"Loss: {avg_loss:.4f}",
                    f"PPL: {ppl:.2f}",
                    f"Forward: {forward_ms:.1f}ms",
                ]
                if enable_generation:
                    parts.append(f"BERTScore: {bertscore_str}")
                    parts.append(f"Gen: {generation_ms:.1f}ms ({tokens_per_sec:.1f} tok/s)")
                if tuned_lens is not None:
                    parts.append("TL: ok")
                tqdm.write(" | ".join(parts))

            excerpt_id += 1

    if verbose:
        print(f"\nDone! {excerpt_id} excerpts saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Collect token-evolution data from a causal LM over WikiText."
    )
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: mps if available, else cpu)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Root directory for output files")
    parser.add_argument("--dataset", default="wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1",
                        help="HuggingFace dataset config")
    parser.add_argument("--split", default="test",
                        help="Dataset split to use")
    parser.add_argument("--min-total-len", type=int, default=MIN_TOTAL_LEN,
                        help="Minimum token count for an excerpt to be included")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help="Truncate sequences to this many tokens")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN,
                        help="Number of tokens used as the generation prompt")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Forward-pass batch size")
    parser.add_argument("--enable-tuned-lens", action="store_true",
                        help="Compute TunedLens trajectory metrics")
    parser.add_argument("--save-hidden-states", action="store_true",
                        help="Save raw per-layer hidden states (large)")
    parser.add_argument("--generation", action="store_true",
                        help="Do autoregressive generation and BERTScore")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-excerpt progress output")
    parser.add_argument("--small-run", action="store_true",
                        help="Use a small subset of the data for quick testing")
    args = parser.parse_args()

    device = args.device or DEVICE

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    texts = [x["text"] for x in dataset if is_valid(x["text"])]
    texts = [
        t for t in texts
        if passes_length_filter(t, tokenizer, min_tokens=args.min_total_len)
    ]
    if args.max_seq_len:
        texts = [tokenizer.decode(
            tokenizer(t)["input_ids"][:args.max_seq_len]
        ) for t in texts]

    if args.small_run:
        texts = texts[:4]

    collect_token_evolution_data(
        texts,
        output_dir=args.output_dir,
        model_name=args.model,
        device=device,
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
        verbose=args.verbose,
        enable_tuned_lens=args.enable_tuned_lens,
        save_hidden_states=args.save_hidden_states,
        enable_generation=args.generation,
    )