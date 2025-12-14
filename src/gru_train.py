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



def train_loop(dataloader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)