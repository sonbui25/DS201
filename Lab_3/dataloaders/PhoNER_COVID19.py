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
        
        words_list = item['words']
        tags = item['tags']
        
        # Encode words
        joined_sentence = " ".join(words_list)
        input_ids = self.vocab.encode_sequence_labeling(joined_sentence, max_len=self.config['vocab']['max_length'])
        # print("Input IDs:", input_ids)
        # Decode (for debugging)
        # decoded_sentence = self.vocab.decode_sentence(input_ids.unsqueeze(0), join_words=False)
        # print("Decoded Sentence:", decoded_sentence)
        label_ids = [self.vocab.encode_label(tag) for tag in tags]

        # Padding labels to match input_ids length
        if len(label_ids) < self.config['vocab']['max_length']:
            label_ids.extend([-100] * (self.config['vocab']['max_length'] - len(label_ids)))
        elif len(label_ids) > self.config['vocab']['max_length']:
            label_ids = label_ids[:self.config['vocab']['max_length']]
        return {
            "input_ids": input_ids,
            "label": torch.tensor(label_ids).long()
        }

