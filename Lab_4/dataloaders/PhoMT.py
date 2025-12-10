import torch
from torch.utils.data import Dataset
from utils.vocab import Vocab  
import json

class PhoMTDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, config: dict):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.load_dataset(path)
        
    def load_dataset(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        # Get keys for source and target based on vocab languages
        src_key_text = f"{self.vocab.src_language}"
        tgt_key_text = f"{self.vocab.tgt_language}"
        
        # Extract source and target text
        src_text = self.data[idx][src_key_text]
        tgt_text = self.data[idx][tgt_key_text]
        
        # Encode sentences
        src_encoded = self.vocab.encode_sentence(src_text, src_key_text)
        tgt_encoded = self.vocab.encode_sentence(tgt_text, tgt_key_text)
        
        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_encoded': src_encoded,
            'tgt_encoded': tgt_encoded
        }