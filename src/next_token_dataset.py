import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class AutoTextDataset(Dataset):
    def __init__(self, data):
        self.tokens = data.tokens
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, index):
        token = eval(self.tokens[index])
        return {
            'x': torch.tensor(token[:-1]),
            'y': torch.tensor(token[1:])
        }

        
def collate_fn(batch):
    l = torch.tensor([len(item['x']) for item in batch])
    sort_ind = torch.argsort(l, descending=True)
    x = [item['x'] for item in batch]
    y = [item['y'] for item in batch]
    
    sort_x = [x[i] for i in sort_ind]
    sort_y = [y[i] for i in sort_ind]

    pad_x = pad_sequence(sort_x, batch_first=True, padding_value=50256)
    pad_y = pad_sequence(sort_y, batch_first=True, padding_value=50256)
    return {
        'lengths': [l[i] for i in sort_ind], 
        'x': pad_x, 
        'y': pad_y, 
    }