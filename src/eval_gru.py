import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer
from torch.optim import Adam
import evaluate


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm



def val_loop(dataloader, model, criterion, tokenizer, device, num_samples=3):
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []
    rouge = evaluate.load("rouge")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

            if i == 0:
                for j in range(min(num_samples, x.size(0))):
 
                    ref_tokens = y[j].cpu().tolist()
                    ref_tokens = [t for t in ref_tokens if t != tokenizer.pad_token_id]
                    ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)

                    context = x[j:j+1] 
                    generated = model.generate(context, max_new_tokens=20, temperature=0.7)

                    gen_tokens = generated[0, context.size(1):].cpu().tolist()
                    gen_tokens = [t for t in gen_tokens if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

                    all_refs.append(ref_text)
                    all_preds.append(gen_text)

    avg_loss = total_loss / len(dataloader)
    rouge_scores = rouge.compute(predictions=all_preds, references=all_refs)
    return avg_loss, rouge_scores, list(zip(all_refs, all_preds))


