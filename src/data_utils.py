import pandas as pd
from transformers import AutoTokenizer
import re



def preprocessing_function(s):
    s = s.lower()
    s = re.sub(r'@\w+', '', s)
    s = re.sub(r'\b(https?://|www\.)\S+', '', s)
    # s = regex.sub(r'[^\p{L}\p{N}\s]', ' ', s)
    s = re.sub(r' +', ' ',s).strip()
    s += '<|endoftext|>'
    return s


def preprocessing_data(data: pd.DataFrame, label_text: str = 'text'):
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    data = data[label_text].apply(preprocessing_function)
    data = tokenizer(data.tolist()).input_ids
    return data