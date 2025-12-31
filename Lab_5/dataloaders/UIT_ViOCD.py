import torch
from utils.vocab import Vocab
from torch.utils.data import Dataset
import json
class UIT_ViOCD_Dataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, config: dict):
        self.path = path
        self.vocab = vocab
        self.config = config
        self.load_data()
    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Thử đọc như dict JSON trước
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    self.data = list(data.values())
                else:
                    self.data = data
            except json.JSONDecodeError:
                # Nếu không phải single JSON, thử đọc như JSON Lines
                self.data = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {self.path}")
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract text and label
        text = item['review']
        label = item['domain']
      
        # Encode text using the provided vocabulary
        input_ids = self.vocab.encode_sentence(text)
        label_id = self.vocab.l2i[label]
        
        return {
            "input_ids": input_ids,
            "label": label_id
        }
        
        