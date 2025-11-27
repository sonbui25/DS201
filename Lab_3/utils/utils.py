import os
import shutil
import torch
import kagglehub
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path
import re
from unicodedata import normalize

def download_data_and_clear_cache(dataset: str, target_dir: str):
    # If data already exists in target_dir, skip downloading
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Dataset already exists in: {target_dir}")
        return target_dir

    # Step 1: Download dataset to cache
    cache_path = kagglehub.dataset_download(dataset)
    # Step 2: Create target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Step 3: Copy data from cache to target directory
    for item in os.listdir(cache_path):
        s = os.path.join(cache_path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # Step 4: Remove the specific dataset cache (usually stored in C drive), does not affect other datasets
    dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(cache_path)))  # go up 3 levels to reach datasets folder
    shutil.rmtree(dataset_root, ignore_errors=True)

    print(f"Dataset copied to: {target_dir}")
    print(f"Cache for {dataset} removed at: {dataset_root}")
    return target_dir
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(20,30))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 5);        
        index += 1
    plt.show()

def plot_metrics(results, epochs, model_name, dataset_name):
    metrics = [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc", "val_acc", "Accuracy"),
        ("train_precision", "val_precision", "Precision"),
        ("train_recall", "val_recall", "Recall"),
        ("train_f1", "val_f1", "F1-score"),
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
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_metrics.png")
    plt.savefig(save_path)
    print(f"Metrics figure saved at: {save_path}")
    plt.show()
def collate_fn(samples: list[dict], task: str) -> torch.Tensor:
    sentences = [sample['input_ids'] for sample in samples]
    labels = [sample['label'] for sample in samples]
    if task == 'text_classification':
        sentences = torch.stack(sentences, dim=0) # Shape(bs, max_len)
        labels = torch.tensor(labels)
    elif task == 'seq_labeling':
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0) # Shape(bs, max_len)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # Shape(bs, max_len)
    return sentences, labels

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
