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
MAX_EXCERPTS = 1000
BATCH_SIZE = 4
OUTPUT_DIR = "token_evolution_data"

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

def collect_token_evolution_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------
    # Load model + tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MAX_CTX = model.config.max_position_embeddings

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def is_valid(text):
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
        tokens = tokenizer(text)["input_ids"]
        return len(tokens) >= min_tokens

    texts = [x["text"] for x in dataset if is_valid(x["text"])]
    texts = [t for t in texts if passes_length_filter(t, tokenizer)]
    random.seed(42)
    random.shuffle(texts)
    texts = texts[:MAX_EXCERPTS]

    print(f"Total excerpts to process: {len(texts)}")

    # ---------------------------
    # Timing helper
    # ---------------------------
    def sync_and_time():
        if DEVICE == "mps":
            torch.mps.synchronize()
        elif DEVICE == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    # ---------------------------
    # Write CSV headers once upfront
    # Any existing files are overwritten cleanly at the start
    # ---------------------------
    EXCERPT_CSV = os.path.join(OUTPUT_DIR, "excerpt_level.csv")
    TOKEN_CSV   = os.path.join(OUTPUT_DIR, "token_level.csv")

    pd.DataFrame(columns=EXCERPT_FIELDS).to_csv(EXCERPT_CSV, index=False)
    pd.DataFrame(columns=TOKEN_FIELDS).to_csv(TOKEN_CSV, index=False)

    # ---------------------------
    # Batch processing
    # ---------------------------
    excerpt_id = 0

    for batch_start in range(0, len(texts), BATCH_SIZE):
        print(f"Starting batch containing excerpts [{batch_start}, {batch_start + BATCH_SIZE})")
        batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

        batch_token_lists = []
        batch_prompt_tokens = []
        batch_reference_tokens = []
        batch_prompt_texts = []
        batch_reference_texts = []

        # ---------------------------
        # Tokenize
        # ---------------------------
        for text in batch_texts:
            tokens = tokenizer(text)["input_ids"]

            prompt_tokens    = tokens[:PROMPT_LEN]
            reference_tokens = tokens[PROMPT_LEN:]
            full_tokens      = tokens[:MAX_CTX]

            batch_token_lists.append(torch.tensor(full_tokens))
            batch_prompt_tokens.append(prompt_tokens)
            batch_reference_tokens.append(reference_tokens)
            batch_prompt_texts.append(tokenizer.decode(prompt_tokens))
            batch_reference_texts.append(tokenizer.decode(reference_tokens))

        if not batch_token_lists:
            continue

        # ---------------------------
        # Pad batch sequences
        # ---------------------------
        seq_lens = [len(t) for t in batch_token_lists]
        max_len  = max(seq_lens)

        input_ids_list      = []
        attention_mask_list = []

        for t in batch_token_lists:
            pad_len = max_len - len(t)
            input_ids_list.append(torch.cat([t, torch.full((pad_len,), tokenizer.pad_token_id)]))
            attention_mask_list.append(torch.cat([torch.ones(len(t)), torch.zeros(pad_len)]))

        input_ids      = torch.stack(input_ids_list).to(DEVICE)
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
        forward_ms = (t_forward_end - t_forward_start) * 1000

        logits        = outputs.logits
        hidden_states = outputs.hidden_states
        print("Forward pass on batch finished!")

        # ---------------------------
        # Compute per-token loss
        # ---------------------------
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fct     = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
        losses       = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = losses.view(batch_size, -1)

        # ---------------------------
        # Save hidden states
        # ---------------------------
        hidden_np = torch.stack(hidden_states).cpu().numpy()

        for i in range(batch_size):
            np.save(
                os.path.join(OUTPUT_DIR, f"hidden_states_excerpt_{excerpt_id}.npy"),
                hidden_np[:, i, :, :]
            )

            # ---------------------------
            # Token rows — all tokens for this excerpt written in one shot
            # ---------------------------
            print(f"Writing token-level rows for excerpt {excerpt_id}...")
            token_df_batch = pd.DataFrame([{
                "excerpt_id":      excerpt_id,
                "token_idx":       j + 1,
                "true_token_id":   input_ids[i, j + 1].item(),
                "true_token_text": tokenizer.decode([input_ids[i, j + 1].item()]),
                "pred_token_id":   torch.argmax(logits[i, j, :]).item(),
                "pred_token_text": tokenizer.decode([torch.argmax(logits[i, j, :]).item()]),
                "loss":            token_losses[i, j].item(),
            } for j in range(seq_len - 1)])

            token_df_batch.to_csv(TOKEN_CSV, mode='a', header=False, index=False)

            # ---------------------------
            # Generation — TIMED
            # ---------------------------
            print("Now starting generation task...")
            prompt_ids         = torch.tensor(batch_prompt_tokens[i]).unsqueeze(0).to(DEVICE)
            prompt_mask        = torch.ones_like(prompt_ids).to(DEVICE)
            reference_tokens_i = batch_reference_tokens[i]
            reference_text_i   = batch_reference_texts[i]

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

            t_gen_end        = sync_and_time()
            generation_ms    = (t_gen_end - t_gen_start) * 1000
            tokens_generated = generated_ids.shape[1] - prompt_ids.shape[1]
            tokens_per_sec   = tokens_generated / (generation_ms / 1000)

            generated_new  = generated_ids[0][len(batch_prompt_tokens[i]):]
            generated_text = tokenizer.decode(generated_new, skip_special_tokens=True)
            print(f"Text model generated: {generated_text}")
            print(f"True reference text:  {reference_text_i[:200]}")

            # ---------------------------
            # BERTScore
            # ---------------------------
            try:
                P, R, F1 = score([generated_text], [reference_text_i[:1000]], lang="en", verbose=False)
                bertscore = F1.item()
            except Exception as e:
                print(f"BERTScore failed for excerpt {excerpt_id}: {e}")
                bertscore = None

            avg_loss   = token_losses[i].mean().item()
            perplexity = torch.exp(token_losses[i].mean()).item()

            # ---------------------------
            # Excerpt row — written immediately
            # ---------------------------
            pd.DataFrame([{
                "excerpt_id":       excerpt_id,
                "prompt":           batch_prompt_texts[i],
                "reference":        reference_text_i,
                "generated":        generated_text,
                "avg_loss":         avg_loss,
                "perplexity":       perplexity,
                "bertscore":        bertscore,
                "forward_pass_ms":  forward_ms,
                "generation_ms":    generation_ms,
                "tokens_generated": tokens_generated,
                "tokens_per_sec":   tokens_per_sec,
            }]).to_csv(EXCERPT_CSV, mode='a', header=False, index=False)

            bertscore_str = f"{bertscore:.4f}" if bertscore is not None else "N/A"
            print(
                f"Excerpt {excerpt_id} | "
                f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                f"BERTScore: {bertscore_str} | "
                f"Forward: {forward_ms:.1f}ms | "
                f"Generation: {generation_ms:.1f}ms ({tokens_per_sec:.1f} tok/s)"
            )

            excerpt_id += 1
            print(f"Finished excerpt #{excerpt_id}!")

    print("\nDone!")
    print(f"Saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    collect_token_evolution_data()
    # hidden = np.load("token_evolution_data/hidden_states_excerpt_0.npy")
    # print(hidden.shape)  # expect (num_layers, seq_len, hidden_dim)
    # print(hidden)