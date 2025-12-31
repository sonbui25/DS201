import torch
from torch.utils.data import Dataset
import json
from utils.vocab import Vocab
class PhoNER_COVID19Dataset(Dataset):
    def __init__(self, path: str, vocab: Vocab, config: dict):
        self.path = path
        self.vocab = vocab
        self.config = config
        self.load_data(path)
    
    def load_data(self, path: str):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Bỏ qua các dòng trống
                    self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        words_list = item['words'] # Đây là list: ["Đồng", "thời", ...]
        tags = item['tags']
        
        input_ids = self.vocab.encode_sequence_labeling(words_list, max_len=None)
        label_ids = [self.vocab.encode_label(tag) for tag in tags]
    
        return {
            "input_ids": input_ids, # Tensor
            "label": torch.tensor(label_ids).long() # Tensor
        }

