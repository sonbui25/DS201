import torch
from utils.vocab import Vocab
from torch.utils.data import Dataset
import json
class UIT_VSFCDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, config: dict):
        self.path = path
        self.vocab = vocab
        self.config = config
        self.load_data()
    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract text and label
        text = item['sentence']
        label = item['topic']
        
        # Encode text using the provided vocabulary
        input_ids = self.vocab.encode_sentence(text, max_len=self.config['vocab']['max_length'])
        label_id = self.vocab.l2i[label]
        
        return {
            "input_ids": input_ids,
            "label": label_id
        }
        
        