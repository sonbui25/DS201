import os
import shutil
import torch
import kagglehub
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path
import re
from unicodedata import normalize
import json

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
    
def collate_fn(samples: list[dict], task: str):
    sentences = [sample['input_ids'] for sample in samples]
    labels = [sample['label'] for sample in samples]
    # Lưu độ dài thực của mỗi sequence trước khi padding
    lengths = torch.tensor([len(s) for s in sentences], dtype=torch.long)
    # print(lengths)
    if task == 'text_classification':
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0) # Shape(bs, max_len)
        labels = torch.tensor(labels)
        return sentences, labels, lengths
    elif task == 'seq_labeling':
        sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0) # Shape(bs, max_len)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # Shape(bs, max_len)
        return sentences, labels, lengths

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
def extract_entities(bio_tags: List[int], idx2label: Dict[int, str]) -> List[Tuple[str, int, int]]:
    """
    Extract entities from BIO tags.
    Returns list of (entity_type, start_idx, end_idx)
    """
    entities = []
    current_entity = None
    start_idx = None
    
    for i, tag_idx in enumerate(bio_tags):
        tag = idx2label[tag_idx]
        
        if tag == 'O':
            if current_entity is not None:
                entities.append((current_entity, start_idx, i - 1))
                current_entity = None
                start_idx = None
        else:
            # Extract entity type (remove B- or I- prefix)
            entity_type = tag[2:]  # Skip "B-" or "I-"
            
            if tag.startswith('B-'):
                # Start of a new entity
                if current_entity is not None:
                    entities.append((current_entity, start_idx, i - 1))
                current_entity = entity_type
                start_idx = i
            elif tag.startswith('I-'):
                # Continue current entity or start new one if type changes
                if current_entity != entity_type:
                    if current_entity is not None:
                        entities.append((current_entity, start_idx, i - 1))
                    current_entity = entity_type
                    start_idx = i
    
    # Handle entity at the end
    if current_entity is not None:
        entities.append((current_entity, start_idx, len(bio_tags) - 1))
    
    return entities

def get_entity_type_labels(bio_tags: List[int], idx2label: Dict[int, str]) -> List[str]:
    """
    Convert BIO tags to entity type labels (gom lại B và I cùng loại).
    Returns list of entity type labels with same length as input.
    """
    entity_type_labels = []
    
    for tag_idx in bio_tags:
        tag = idx2label[tag_idx]
        if tag == 'O':
            entity_type_labels.append('O')
        else:
            # Extract entity type (remove B- or I- prefix)
            entity_type = tag[2:]  # Skip "B-" or "I-"
            entity_type_labels.append(entity_type)
    
    return entity_type_labels

def bio_tags_to_entity_types(y_true_bio: List[int], y_pred_bio: List[int], idx2label: Dict[int, str]) -> Tuple[List[str], List[str]]:
    """
    Convert BIO tag sequences to entity type sequences.
    Returns (y_true_entity_types, y_pred_entity_types) both as lists of strings
    """
    y_true_entity = get_entity_type_labels(y_true_bio, idx2label)
    y_pred_entity = get_entity_type_labels(y_pred_bio, idx2label)
    
    return y_true_entity, y_pred_entity

def convert_bio_indices_to_tags(tag_indices: List[int], idx2label: Dict[int, str]) -> List[str]:
    """
    Convert BIO tag indices to BIO tag strings.
    Input: [0, 10, 20] (indices)
    Output: ['B-AGE', 'I-AGE', 'O'] (strings)
    """
    return [idx2label[int(idx)] for idx in tag_indices]

def compute_seqeval_metrics_from_sequences(y_true_sequences: List[List[str]], 
                                          y_pred_sequences: List[List[str]]) -> Tuple[Dict, str]:
    """
    Compute entity-level metrics using seqeval from sequences of BIO tags.
    
    Args:
        y_true_sequences: List of sequences, each sequence is list of BIO tags (strings)
                         e.g., [['B-AGE', 'I-AGE', 'O', 'B-NAME'], ['O', 'B-DATE']]
        y_pred_sequences: List of predicted sequences (same format)
    
    Returns:
        (metrics_dict, report_str)
    """
    try:
        from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
        from seqeval.metrics import classification_report as seqeval_report
    except ImportError:
        raise ImportError("seqeval not installed. Install with: pip install seqeval")
    
    # Compute entity-level metrics
    accuracy = accuracy_score(y_true_sequences, y_pred_sequences)
    precision = precision_score(y_true_sequences, y_pred_sequences, average='macro', zero_division=0)
    recall = recall_score(y_true_sequences, y_pred_sequences, average='macro', zero_division=0)
    f1 = f1_score(y_true_sequences, y_pred_sequences, average='macro', zero_division=0)
    
    # Get detailed report
    report_str = seqeval_report(y_true_sequences, y_pred_sequences, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, report_str