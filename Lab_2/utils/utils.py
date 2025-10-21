import os
import shutil
import torch
import kagglehub
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from pathlib import Path
def download_data_and_clear_cache(dataset: str, target_dir: str):
    # Nếu đã tồn tại dữ liệu ở target_dir thì bỏ qua
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Dataset already exists in: {target_dir}")
        return target_dir

    # B1: Download dataset về cache
    cache_path = kagglehub.dataset_download(dataset)
    # B2: Tạo thư mục đích nếu chưa có
    os.makedirs(target_dir, exist_ok=True)

    # B3: Copy dữ liệu sang thư mục đích
    for item in os.listdir(cache_path):
        s = os.path.join(cache_path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # B4: Xóa đúng dataset (thường lưu trong ổ C) vừa tải trong cache (không ảnh hưởng dataset khác)
    dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(cache_path)))  # quay lại 3 cấp để đến thư mục datasets
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

def plot_metrics(results, epochs, model_name):
    metrics = [
        ("train_loss", "test_loss", "Loss"),
        ("train_acc", "test_acc", "Accuracy"),
        ("train_precision", "test_precision", "Precision"),
        ("train_recall", "test_recall", "Recall"),
        ("train_f1", "test_f1", "F1-score"),
    ]
    plt.figure(figsize=(18, 10))
    plt.suptitle(f"Metrics over Epochs of {model_name} model (turn off this window to continue training process of next model!)", fontsize=16)
    for i, (train_key, test_key, title) in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(range(1, epochs+1), results[train_key], label="Train")
        plt.plot(range(1, epochs+1), results[test_key], label="Test")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
def collate_fn(samples: list[dict]) -> torch.Tensor:
    images = [sample['image'] for sample in samples]
    labels = [sample['label'] for sample in samples]

    images = torch.stack(images, dim=0) # Shape(Batch_size, C, H, W)
    labels = torch.tensor(labels)
    return images, labels