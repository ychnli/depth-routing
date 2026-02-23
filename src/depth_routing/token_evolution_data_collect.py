import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score
import os

# ---------------------------
# Config
# ---------------------------
MODEL_NAME = "EleutherAI/pythia-160m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_LEN = 64
MIN_TOTAL_LEN = 100
MAX_EXCERPTS = 20   # for testing, we can increase
BATCH_SIZE = 4      # batch forward pass size (# excerpts)
OUTPUT_DIR = "outputs"
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

def is_valid(text):
    text = text.strip()
    return len(text) > 0 and not text.startswith("=")

texts = [x["text"] for x in dataset if is_valid(x["text"])]

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

        if len(tokens) < MIN_TOTAL_LEN:
            continue

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
    # Forward pass for batch
    # ---------------------------
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    logits = outputs.logits                               # (batch_size, seq_len, vocab_size)
    hidden_states = outputs.hidden_states                 # tuple: (num_layers, batch_size, seq_len, hidden_dim)
    num_layers = len(hidden_states)

    # ---------------------------
    # Compute per-token loss
    # ---------------------------
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    token_losses = losses.view(batch_size, -1)  # (batch_size, seq_len-1)

    # ---------------------------
    # Save hidden states and token-level info
    # ---------------------------
    hidden_np = torch.stack(hidden_states).cpu().numpy()  # (num_layers, batch_size, seq_len, hidden_dim)
    
    for i in range(batch_size):
        # Save hidden states per excerpt
        np.save(os.path.join(OUTPUT_DIR, f"hidden_states_excerpt_{excerpt_id}.npy"), hidden_np[:, i, :, :])

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
        # Generation + BERTScore
        # ---------------------------
        prompt_ids = torch.tensor(batch_prompt_tokens[i]).unsqueeze(0).to(DEVICE)
        prompt_mask = torch.ones_like(prompt_ids).to(DEVICE)
        reference_tokens_i = batch_reference_tokens[i]
        reference_text_i = batch_reference_texts[i]

        gen_len = min(len(reference_tokens_i), MAX_CTX - PROMPT_LEN)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=gen_len,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated_new = generated_ids[0][len(batch_prompt_tokens[i]):]
        generated_text = tokenizer.decode(generated_new, skip_special_tokens=True)

        # BERTScore
        try:
            P, R, F1 = score([generated_text], [reference_text_i[:1000]], lang="en", verbose=False)
            bertscore = F1.item()
        except Exception as e:
            print(f"BERTScore failed for excerpt {excerpt_id}: {e}")
            bertscore = None

        avg_loss = token_losses[i].mean().item()
        perplexity = torch.exp(token_losses[i].mean()).item()

        # Excerpt-level logging
        excerpt_rows.append({
            "excerpt_id": excerpt_id,
            "prompt": batch_prompt_texts[i],
            "reference": reference_text_i,
            "generated": generated_text,
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "bertscore": bertscore
        })

        print(f"Processed excerpt {excerpt_id} | Avg Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | BERTScore: {bertscore}")
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