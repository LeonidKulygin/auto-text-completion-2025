import torch 
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class GRUmodel(nn.Module):
    def __init__(self, vocab_size=50257, embed_dim=128, hidden_dim=64, padding_idx=50256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, lengths=None):
        emb = self.emb(x)
        if lengths is not None:
            packed_emb = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed_emb)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(emb)
        logits = self.fc(out)
        return logits

    def generate(self, context_tokens, max_new_tokens=10, temperature=1.0):
        self.eval()
        with torch.no_grad():
            generated = context_tokens.clone()

            for _ in range(max_new_tokens):
                logits = self(generated)  
                next_logits = logits[0, -1, :] / temperature 
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == self.emb.padding_idx:
                    break

            return generated