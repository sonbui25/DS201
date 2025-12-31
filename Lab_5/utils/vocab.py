import os
import json
import torch
from typing import List
from .utils import preprocess_sentence
import torch.nn.functional as F 
import numpy as np
class Vocab(object):
    """
        A base Vocab class that is used to create vocabularies for particular tasks
    """
    def __init__(self, config):
        self.config = config
        
        self.vocab_prefix = config['vocab_prefix']
        # Special tokens and their IDs
        self.unk_piece = config['unk_piece']
        self.bos_piece = config['bos_piece']
        self.eos_piece = config['eos_piece']
        self.pad_piece = config['pad_piece']

        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.specials = [self.pad_piece, self.unk_piece, self.bos_piece, self.eos_piece]
        # Vocab size will be determined based on the config
        self.vocab_size = 0
        # Mappings from token to index and index to token
        self.stoi = {}
        self.itos = {}
        # Label mappings
        self.i2l = {}
        self.l2i = {}

        # Build vocabulary
        self.make_vocab(config)

    def make_vocab(self, config):
        """
        Gather text from JSON files, determine vocab size
        """
        path = config['path']['train']
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        words_set = set()
        labels = set()
        
        # Collect text data and labels
        try:
            # Thử đọc toàn bộ file như một khối JSON duy nhất
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)  # Nếu file là một khối JSON duy nhất
                if isinstance(data, list):  # Nếu file là một danh sách các đối tượng JSON
                    for item in data:
                        self.process_item(item, config, words_set, labels)
                elif isinstance(data, dict):  # Nếu file là một dict với numeric keys
                    for key, item in data.items():
                        self.process_item(item, config, words_set, labels)
                else:
                    raise ValueError("Unexpected JSON format: Expected a list of objects or a dictionary.")
        except json.JSONDecodeError:
            # Nếu không phải một khối JSON duy nhất, đọc từng dòng (JSON Lines)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Bỏ qua các dòng trống
                        item = json.loads(line)  # Parse từng dòng JSON
                        self.process_item(item, config, words_set, labels)
        # Use all words from the set
        num_words = len(words_set)
        self.vocab_size = num_words + len(self.specials) 
        
        # Build stoi/itos
        for idx, special_token in enumerate(self.specials):
            self.stoi[special_token] = idx
            self.itos[idx] = special_token
            
        # start=len(self.specials)
        for idx, word in enumerate(sorted(words_set), start=len(self.specials)):
            self.stoi[word] = idx
            self.itos[idx] = word
        
        # Build label dictionaries
        for idx, label in enumerate(sorted(list(labels))):
            self.l2i[label] = idx
            self.i2l[idx] = label

        #Save labels
        self.save_labels()
                    
        
    def encode_sentence(self, sentence: str, max_len: int = None) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence_list = preprocess_sentence(sentence)
        input_ids = [self.bos_id] + [self.stoi[token] if token in self.stoi else self.unk_id for token in sentence_list] + [self.eos_id]
        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids.extend([self.pad_id] * (max_len - len(input_ids)))
        return torch.tensor(input_ids).long()
    
    def decode_sentence(self, input_ids: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in input_ids:
            question = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())
         # Nếu sentence là 2D array thì bung thành 1D
        if np.array(sentences).ndim == 2:
            sentences = np.array(sentences).flatten()
        return sentences
    def encode_sequence_labeling(self, text, max_len: int = None) -> torch.Tensor:
        # 1. KIỂM TRA LOẠI DỮ LIỆU ĐẦU VÀO
        if isinstance(text, list):
            # Nếu là list (từ Dataset truyền vào), dùng luôn
            sentence_list = text
        else:
            # Nếu là string, mới dùng preprocess
            sentence_list = preprocess_sentence(text)

        # 2. Map sang ID (Giữ nguyên logic cũ)
        input_ids = [self.stoi[token] if token in self.stoi else self.unk_id for token in sentence_list]
        
        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids.extend([self.pad_id] * (max_len - len(input_ids)))
                
        return torch.tensor(input_ids).long()
    def align_labels_with_subwords(self, labels: List[str], word_to_subword_mapping: List[int]) -> List[str]:
        pass
    def subword_labels_to_word_labels(self, subword_labels: List[str], word_to_subword_mapping: List[int]) -> List[str]:
        pass
    
    def encode_label(self, label: str) -> int:
        return self.l2i[label]
    
    def decode_label(self, label_idx: int) -> str:
        return self.i2l[label_idx]
    
    def save_labels(self):
        """
        Save the label dictionaries (i2l, l2i) into a JSON file.
        """
        # Lấy đường dẫn thư mục hiện tại của file vocab.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        labels_file = os.path.join(current_dir, f"{self.vocab_prefix}_labels.json")
        
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump({"i2l": self.i2l, "l2i": self.l2i}, f, ensure_ascii=False)
        print(f"Labels saved to {labels_file}")
        
    def load_labels(self):
        """
        Load the label dictionaries (i2l, l2i) from a JSON file, if it exists.
        """
        # Lấy đường dẫn thư mục hiện tại của file vocab.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        labels_file = os.path.join(current_dir, f"{self.vocab_prefix}_labels.json")
        
        if not os.path.exists(labels_file):
            print(f"Labels file not found: {labels_file}.")
            return

        with open(labels_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            # Convert keys of i2l back to integers
            self.i2l = {int(k): v for k, v in label_data["i2l"].items()}
            self.l2i = label_data["l2i"]

        print(f"Labels loaded successfully from {labels_file}")
    @property
    def total_labels(self) -> int:
        """Number of distinct labels."""
        return len(self.l2i)

    @property
    def total_tokens(self) -> int:
        """Vocabulary size determined at training time."""
        return self.vocab_size

    @property
    def get_vocab_size(self) -> int:
        """
        Returns the vocab size. Named differently to avoid confusion
        """
        return self.vocab_size

    @property
    def get_pad_idx(self) -> int:
        """Get the ID of the padding token."""
        return self.pad_id
    def process_item(self, item, config, words_set, labels):
        """
        Process a single JSON object to extract tokens and labels
        """
        raw_text = item[config['text']]
        
        if isinstance(raw_text, list):
            tokens = raw_text 
            words_set.update(tokens)
        else:
            # Nếu là string thì mới cần preprocess
            tokens = preprocess_sentence(raw_text)
            words_set.update(tokens)        
        # Xử lý nhãn (labels) 
        if self.config.get("task_type", None) == "seq_labeling":
            for label in item[config['label']]:
                labels.add(label)
        else:
            labels.add(item[config['label']])