# this will be testing the last model

import torch
import torch.nn.functional as F
import json
from models.GPTModel import *
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

special_vocab = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
new_chars = "012456789<|>[]{}()_\""
all_chars = special_vocab + new_chars
print(len(all_chars))

def encode(text):
    return [all_chars.index(ch) for ch in text if ch in all_chars]

def decode(indices):
    return ''.join(all_chars[i] for i in indices if 0 <= i < len(all_chars))

# test encode and decode
# context = "Tamim"
# dec = encode(context)
# print(dec)
# print(decode(dec))



model = GPTModelStyle(
    vocab_size=len(all_chars),
    n_embed=384,
    block_size=256,
    n_head=6,
    n_layer=6,
    dropout=0.2,
    device=device
)

model.load_state_dict(torch.load("GPT-FineTunned.pt", map_location=device))
model.to(device)
model.eval()

def generate(model, context, max_tokens=100, temperature=0.5, top_k=50, rep_penalty=1.2):
    generated = context.clone()
    for _ in range(max_tokens):
        idx_cond = generated[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        recent_tokens = generated[0, -20:].tolist() if generated.shape[1] > 20 else generated[0].tolist()
        for token in set(recent_tokens):
            logits[0, token] /= rep_penalty
        logits = logits / temperature
        if top_k and top_k > 0:
            top_logits, top_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(1, next_token_idx)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        text_so_far = decode(generated[0].tolist())
        if "\nHuman:" in text_so_far:
            break
    return generated

if __name__ == "__main__":
    question = input("You: ")
    while question.lower() != "q":
        prompt = f"Human: {question}\nAssistant:"
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

        with torch.no_grad():
            output = generate(model, context)

        full_text = decode(output[0].tolist())
        word_to_strip = '<|endoftext|>'


        if "Assistant:" in full_text:
            assistant_response = full_text.split("Assistant:", 1)[1]
            if "\nHuman:" in assistant_response:
                assistant_response = assistant_response.split("\nHuman:")[0]
            assistant_response = assistant_response.strip()
            cleaned_response = assistant_response.replace(word_to_strip, "")
            print(f"AI: {cleaned_response}")
        else:
            print("No assistant response generated.")

        question = input("You: ")
