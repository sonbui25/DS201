import os
import json
import torch
from collections import Counter
from typing import List
from .utils import preprocess_sentence
import torch.nn.functional as F 
import numpy as np
import os
import json
import torch
from collections import Counter
from typing import List
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
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        # Mappings from token to index and index to token, source and target vocabularies (for seq2seq tasks)
        self.src_stoi = {}
        self.src_itos = {}
        self.tgt_stoi = {}
        self.tgt_itos = {}
        
        # Specified source and target language
        self.src_language = config['source_text']
        self.tgt_language = config['target_text']
        # Build vocabulary
        self.make_vocab(config)

    def make_vocab(self, config):
        """
        Gather text from JSON files, determine vocab size
        """
        json_paths = [config['path']['train'], config['path']['val'], config['path']['test']]
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        src_words = set()
        tgt_words = set()
        # Collect text data
        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) 
                if isinstance(data, list):
                    for item in data:
                        self.process_item(item, config, src_words, tgt_words)
                else:
                    raise ValueError("Unexpected JSON format: Expected a list of objects.")
        
        # Determine vocab sizes without special tokens
        num_src_words = len(src_words)
        num_tgt_words = len(tgt_words)
        
        # Update vocab sizes to include special tokens
        self.src_vocab_size = num_src_words + len(self.specials) 
        self.tgt_vocab_size = num_tgt_words + len(self.specials) 
        
        # Build stoi/itos for source and target for special tokens 
        for idx, special_token in enumerate(self.specials):
            self.src_stoi[special_token] = idx
            self.src_itos[idx] = special_token
            self.tgt_stoi[special_token] = idx
            self.tgt_itos[idx] = special_token
            
        # Build source vocabulary, continue from special tokens
        for idx, word in enumerate(src_words, start=len(self.specials)):
            self.src_stoi[word] = idx
            self.src_itos[idx] = word
        
        # Build target vocabulary, continue from special tokens
        for idx, word in enumerate(tgt_words, start=len(self.specials)):
            self.tgt_stoi[word] = idx
            self.tgt_itos[idx] = word
        
    def encode_sentence(self, sentence: str, language: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length based on source/target flag """
        sentence_list = preprocess_sentence(sentence)
        stoi_list = self.src_stoi if language == self.src_language else self.tgt_stoi
        input_ids = [self.bos_id] + [stoi_list[token] if token in stoi_list else self.unk_id for token in sentence_list] + [self.eos_id]
        return torch.tensor(input_ids).long()
    
    def decode_sentence(self, input_ids: torch.Tensor, language: str, join_words=True) -> List[str]:
        '''
            Turn a tensor of indices back into sentences based on source/target flag.
        '''
        sentences = []
        itos_list = self.src_itos if language == self.src_language else self.tgt_itos
        for vec in input_ids:
            sentence = " ".join([itos_list[idx] for idx in vec.tolist() if itos_list[idx] not in self.specials])
            if join_words:
                sentences.append(sentence)
            else:
                sentences.append(sentence.strip().split())
        return sentences
    
    def get_src_vocab_size(self) -> int:
        """Get the source vocabulary size."""
        return self.src_vocab_size
    def get_tgt_vocab_size(self) -> int:
        """Get the target vocabulary size."""
        return self.tgt_vocab_size
    @property
    def get_pad_idx(self) -> int:
        """Get the ID of the padding token."""
        return self.pad_id
    def process_item(self, item, config, src_words: set, tgt_words: set):
        '''
        Process a single JSON item to update source and target word counters
        '''
        # Process source and target texts
        src_tokens = preprocess_sentence(item[config['source_text']])
        src_words.update(src_tokens)
        tgt_tokens = preprocess_sentence(item[config['target_text']])
        tgt_words.update(tgt_tokens)