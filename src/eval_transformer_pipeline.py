import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.next_token_dataset import AutoTextDataset, collate_fn
from torch.nn import CrossEntropyLoss
import evaluate
import pandas as pd
from tqdm import tqdm

rouge_metric = evaluate.load("rouge")

def val_loop(dataloader, model, tokenizer, device, num_samples=3):
    model.eval()
    total_loss = 0
    all_preds, all_refs = [], []
    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            logits = model(x).logits
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

            if i == 0:
                for j in range(min(num_samples, x.size(0))):
                    ref_ids = [t for t in y[j].tolist() if t != tokenizer.pad_token_id]
                    ref = tokenizer.decode(ref_ids, skip_special_tokens=True)

                    context = x[j:j+1]
                    generated = model.generate(
                        context,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    gen_ids = generated[0, context.size(1):].tolist()
                    gen_ids = [t for t in gen_ids if t not in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
                    gen = tokenizer.decode(gen_ids, skip_special_tokens=True)

                    all_refs.append(ref)
                    all_preds.append(gen)

    avg_loss = total_loss / len(dataloader)
    rouge_scores = rouge_metric.compute(predictions=all_preds, references=all_refs)
    return avg_loss, rouge_scores, list(zip(all_refs, all_preds))


def run_gpt2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

    val, test = pd.read_csv('data/val.csv'), pd.read_csv('data/test.csv')
    valds, testds = AutoTextDataset(val), AutoTextDataset(test)
    val_loader = DataLoader(valds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(valds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    val_loss, val_rouge, _ = val_loop(val_loader, model, tokenizer, device)
    print(f"Val — Loss: {val_loss:.4f}, ROUGE-1: {val_rouge['rouge1']:.4f}, ROUGE-2: {val_rouge['rouge2']:.4f},  ROUGE-L: {val_rouge['rougeL']:.4f}")
    
    test_loss, test_rouge, _ = val_loop(test_loader, model, tokenizer, device)
    print(f"Test — Loss: {test_loss:.4f}, ROUGE-1: {test_rouge['rouge1']:.4f}, ROUGE-2: {test_rouge['rouge2']:.4f}, ROUGE-L: {test_rouge['rougeL']:.4f}")