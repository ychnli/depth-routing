import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import os
import time
import random

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "EleutherAI/pythia-160m"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPT_LEN = 64
MIN_TOTAL_LEN = 100
MAX_EXCERPTS = 20   # for testing, we can increase
BATCH_SIZE = 4      # batch forward pass size (# excerpts)
OUTPUT_DIR = "TEST2/token_evolution_data"

def collect_token_evolution_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------
    # Load model + tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MAX_CTX = model.config.max_position_embeddings

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # def is_valid(text):
    #     text = text.strip()
    #     return len(text) > 0 and not text.startswith("=")
    
    def is_valid(text):
        text = text.strip()

        if len(text) == 0 or text.startswith("="):
            return False

        tokens = text.split()

        # 1. Minimum length: short passages lack sufficient context
        #    for meaningful perplexity / attention analysis
        if len(tokens) < 50:
            return False

        # 2. Reject list-heavy text: high ratio of comma/semicolons suggests
        #    enumeration rather than coherent prose — uninteresting for attention plots
        punct_count = text.count(",") + text.count(";")
        if punct_count / len(tokens) > 0.15:
            return False

        # 3. Reject passages with too many numbers: dates/stats produce
        #    artificially high token loss for uninteresting reasons
        num_count = sum(1 for w in tokens if any(c.isdigit() for c in w))
        if num_count / len(tokens) > 0.2:
            return False

        # 4. Reject passages that are mostly capitalized words: usually
        #    proper noun lists (awards, film titles, etc.)
        cap_count = sum(1 for w in tokens if w[0].isupper())
        if cap_count / len(tokens) > 0.4:
            return False

        return True
    
        # After loading and shuffling texts, filter BEFORE slicing
    def passes_length_filter(text, tokenizer, min_tokens=MIN_TOTAL_LEN):
        tokens = tokenizer(text)["input_ids"]
        return len(tokens) >= min_tokens

    texts = [x["text"] for x in dataset if is_valid(x["text"])]
    texts = [t for t in texts if passes_length_filter(t, tokenizer)]
    random.seed(42)
    random.shuffle(texts)
    texts = texts[:10]

    # ---------------------------
    # Timing helper
    # ---------------------------
    def sync_and_time():
        """
        Returns current time in seconds, but first flushes the GPU command queue.
        This ensures the clock starts/stops only after the GPU is truly done —
        not just after the CPU has dispatched the work.
        On CPU, this is a plain perf_counter() call.
        """
        if DEVICE == "mps":
            torch.mps.synchronize()
        elif DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # ---------------------------
    # Storage
    # ---------------------------
    excerpt_rows = []
    token_rows = []
    excerpt_id = 0

    # ---------------------------
    # Batch processing
    # ---------------------------
    for batch_start in range(0, len(texts), BATCH_SIZE):
        print(f"Starting batch containing exercpts [{batch_start}, {batch_start + BATCH_SIZE})")
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

        batch_token_lists = []
        batch_prompt_tokens = []
        batch_reference_tokens = []
        batch_prompt_texts = []
        batch_reference_texts = []

        # ---------------------------
        # Tokenize and filter per excerpt
        # ---------------------------
        for text in batch_texts:
            tokens = tokenizer(text)["input_ids"]
        
            print(f"Tokenized text {text}...")

            # Split prompt + reference
            prompt_tokens = tokens[:PROMPT_LEN]
            reference_tokens = tokens[PROMPT_LEN:]

            prompt_text = tokenizer.decode(prompt_tokens)
            reference_text = tokenizer.decode(reference_tokens)

            # Truncate full sequence to model context
            full_tokens = tokens[:MAX_CTX]

            batch_token_lists.append(torch.tensor(full_tokens))
            batch_prompt_tokens.append(prompt_tokens)
            batch_reference_tokens.append(reference_tokens)
            batch_prompt_texts.append(prompt_text)
            batch_reference_texts.append(reference_text)

        if not batch_token_lists:
            continue  # skip empty batch

        # ---------------------------
        # Pad batch sequences
        # ---------------------------
        seq_lens = [len(t) for t in batch_token_lists]
        max_len = max(seq_lens)

        input_ids_list = []
        attention_mask_list = []

        for t in batch_token_lists:
            pad_len = max_len - len(t)
            input_ids_list.append(torch.cat([t, torch.full((pad_len,), tokenizer.pad_token_id)]))
            attention_mask_list.append(torch.cat([torch.ones(len(t)), torch.zeros(pad_len)]))

        input_ids = torch.stack(input_ids_list).to(DEVICE)          # (batch_size, seq_len)
        attention_mask = torch.stack(attention_mask_list).to(DEVICE)

        batch_size, seq_len = input_ids.shape

        # ---------------------------
        # Forward pass — TIMED
        # ---------------------------
        print("Starting forward pass on batch...")
        t_forward_start = sync_and_time()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        t_forward_end = sync_and_time()
        forward_ms = (t_forward_end - t_forward_start) * 1000  # whole batch, one pass

        logits = outputs.logits                               # (batch_size, seq_len, vocab_size)
        hidden_states = outputs.hidden_states                 # tuple: (num_layers+1, batch_size, seq_len, hidden_dim)
        num_layers = len(hidden_states)
        print("Forward pass on batch finished!")

        # ---------------------------
        # Compute per-token loss
        # ---------------------------
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id) # ignore padded tokens -- model never trained to predict them
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = losses.view(batch_size, -1)  # (batch_size, seq_len-1)

        # ---------------------------
        # Save hidden states and token-level info
        # ---------------------------
        hidden_np = torch.stack(hidden_states).cpu().numpy()  # (num_layers, batch_size, seq_len, hidden_dim)

        for i in range(batch_size):
            # Save hidden states per excerpt
            np.save(os.path.join(OUTPUT_DIR, f"hidden_states_excerpt_{excerpt_id}.npy"), hidden_np[:, i, :, :])
            print(f"Conducitng per-token logging for entry {i} of batch")

            # Token-level logging
            for j in range(seq_len - 1):
                true_token_id = input_ids[i, j + 1].item()
                true_token_text = tokenizer.decode([true_token_id])
                pred_token_id = torch.argmax(logits[i, j, :]).item()
                pred_token_text = tokenizer.decode([pred_token_id])
                loss_val = token_losses[i, j].item()

                token_rows.append({
                    "excerpt_id": excerpt_id,
                    "token_idx": j + 1,
                    "true_token_id": true_token_id,
                    "true_token_text": true_token_text,
                    "pred_token_id": pred_token_id,
                    "pred_token_text": pred_token_text,
                    "loss": loss_val
                })

            # ---------------------------
            # Generation — TIMED
            # ---------------------------
            print("Now starting generation task...")
            prompt_ids = torch.tensor(batch_prompt_tokens[i]).unsqueeze(0).to(DEVICE)
            prompt_mask = torch.ones_like(prompt_ids).to(DEVICE)
            reference_tokens_i = batch_reference_tokens[i]
            reference_text_i = batch_reference_texts[i]

            gen_len = min(len(reference_tokens_i), MAX_CTX - PROMPT_LEN)

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
            tokens_per_sec = tokens_generated / (generation_ms / 1000)

            generated_new = generated_ids[0][len(batch_prompt_tokens[i]):]
            generated_text = tokenizer.decode(generated_new, skip_special_tokens=True)
            print(f"Text model generated: {generated_text}")
            print(f"True Reference Text: {reference_text_i[:1000]}")

            # ---------------------------
            # BERTScore
            # ---------------------------

            try:
                P, R, F1 = score([generated_text], [reference_text_i[:1000]], lang="en", verbose=False)
                bertscore = F1.item()
            except Exception as e:
                print(f"BERTScore failed for excerpt {excerpt_id}: {e}")
                bertscore = None

            avg_loss = token_losses[i].mean().item()
            perplexity = torch.exp(token_losses[i].mean()).item()

            print("Now writing excerpt-level logs")
            # Excerpt-level logging
            excerpt_rows.append({
                # --- original fields ---
                "excerpt_id": excerpt_id,
                "prompt": batch_prompt_texts[i],
                "reference": reference_text_i,
                "generated": generated_text,
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "bertscore": bertscore,
                # --- timing fields ---
                "forward_pass_ms": forward_ms,      # one forward pass over the whole batch
                "generation_ms": generation_ms,      # autoregressive decoding for this excerpt
                "tokens_generated": tokens_generated,
                "tokens_per_sec": tokens_per_sec,    # generation throughput
            })

            print(
                f"Excerpt {excerpt_id} | "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | BERTScore: {bertscore:.4f} | "
                f"Forward: {forward_ms:.1f}ms | "
                f"Generation: {generation_ms:.1f}ms ({tokens_per_sec:.1f} tok/s) | "
            )

            excerpt_id += 1

            if excerpt_id >= MAX_EXCERPTS:
                break

        if excerpt_id >= MAX_EXCERPTS:
            break

    # ---------------------------
    # Save CSVs
    # ---------------------------
    excerpt_df = pd.DataFrame(excerpt_rows)
    token_df = pd.DataFrame(token_rows)

    excerpt_df.to_csv(os.path.join(OUTPUT_DIR, "excerpt_level.csv"), index=False)
    token_df.to_csv(os.path.join(OUTPUT_DIR, "token_level.csv"), index=False)

    print("\nDone!")
    print(f"Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    collect_token_evolution_data()
