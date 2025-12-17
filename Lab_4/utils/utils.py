import os
import shutil
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path
import re
from unicodedata import normalize
def plot_metrics(results, epochs, model_name, dataset_name):
    metrics = [
        ("train_loss", "val_loss", "Loss"),
        ("train_rouge_L", "val_rouge_L", "ROUGE-L"),
    ]
    plt.figure(figsize=(18, 10))
    plt.suptitle(f"Metrics over Epochs of {model_name} model", fontsize=16)
    for i, (train_key, val_key, title) in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(range(epochs), results[train_key], label="Train")
        plt.plot(range(epochs), results[val_key], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    save_dir = "/kaggle/working/DS201/Lab_4/results"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_metrics.png")
    plt.savefig(save_path)
    print(f"Metrics figure saved at: {save_path}")
    plt.show()
def collate_fn(samples: list[dict]) -> torch.Tensor:
    src_encoded_list = [sample['src_encoded'] for sample in samples]
    tgt_encoded_list = [sample['tgt_encoded'] for sample in samples]
    src_encoded_padded = pad_sequence(src_encoded_list, batch_first=True, padding_value=0) # Shape(bs, max_len)
    tgt_encoded_padded = pad_sequence(tgt_encoded_list, batch_first=True, padding_value=0) # Shape(bs, max_len)
    return src_encoded_padded, tgt_encoded_padded

def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    sentence = normalize("NFKC", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    # tokenize the sentence
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens
