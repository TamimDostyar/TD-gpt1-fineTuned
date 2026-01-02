'''
    Main Archeticture
    
'''

import torch, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset


class ByteEncodingTokenizer:
    def __init__(self):
        original_chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        new_chars = "012456789<|>[]{}()_\""
        self.chars = original_chars + new_chars
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    def encode(self, text):
        return [self.stoi.get(ch, self.stoi['\n']) for ch in text]

    def decode(self, indices):
        return ''.join([self.itos.get(i, '') for i in indices])
def collate_fn(batch):
    """Collate function to pad sequences to the same length."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(len(seq) for seq in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    
    for inp, lab in zip(input_ids, labels):
        pad_length = max_len - len(inp)
        if pad_length > 0:
            padded_inp = torch.cat([inp, torch.zeros(pad_length, dtype=inp.dtype)])
            padded_lab = torch.cat([lab, torch.zeros(pad_length, dtype=lab.dtype)])
        else:
            padded_inp = inp
            padded_lab = lab
        padded_input_ids.append(padded_inp)
        padded_labels.append(padded_lab)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels)
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    pred_tokens = np.argmax(predictions, axis=-1)

    mask = labels != -100

    correct = (pred_tokens == labels) * mask
    accuracy = correct.sum() / mask.sum()

    loss_fct = torch.nn.CrossEntropyLoss()
    predictions_tensor = torch.from_numpy(predictions[:, :-1, :])
    labels_tensor = torch.from_numpy(labels[:, 1:])

    shift_logits = predictions_tensor.reshape(-1, predictions_tensor.shape[-1])
    shift_labels = labels_tensor.reshape(-1)

    loss = loss_fct(shift_logits, shift_labels).item()
    perplexity = np.exp(loss)

    return {
        "accuracy": float(accuracy),
        "perplexity": float(perplexity),
        "loss": float(loss)
    }



from sklearn.model_selection import train_test_split
from datasets import Dataset
from datasets import Dataset as HFDataset
def split_conversations(conversations: list, test_size: float = 0.1, random_state: int = 42):
    train_conv, test_conv = train_test_split(
        conversations,
        test_size=test_size,
        random_state=random_state
    )
    

    train_dataset = HFDataset.from_dict({"text": train_conv})
    test_dataset = HFDataset.from_dict({"text": test_conv})
    
    return train_dataset, test_dataset
  

import torch
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer, block_size):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        # Safety Check: If idx is a tensor, convert to int
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Safety Check: If idx is a list (rare, but happens with some samplers), take the first
        if isinstance(idx, list):
             # If this happens, your DataLoader is misconfigured, 
             # but this hack fixes the crash for single-item debugging
            idx = idx[0]

        text = self.conversations[idx]
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        encoded = self.tokenizer.encode(text)

        required_len = self.block_size + 1
        if len(encoded) > required_len:
            encoded = encoded[:required_len]
        else:
            pad_len = required_len - len(encoded)
            encoded = encoded + [0] * pad_len 

        data = torch.tensor(encoded, dtype=torch.long)
        x = data[:-1]
        y = data[1:]
        
        return x, y



