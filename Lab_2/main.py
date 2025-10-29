# -*- coding: utf-8 -*-
import os
import random
import warnings
import argparse
import yaml
from collections import Counter

import numpy as np
import torch
# Import DataLoader and Subset/split tools
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Import models, dataset classes and task engine
from models import LeNet, GoogleNet, ResNet18, ResNet50
from task import classification_engine
from dataloader import MNIST, ViNaFood21
from utils.utils import plot_metrics, collate_fn

# Setup warnings and seeds for reproducibility
warnings.filterwarnings("ignore", message=".*number of unique classes.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images", category=UserWarning)
def seed_worker(worker_id):
    """Seed function for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# MAIN EXECUTION
if __name__ == "__main__":
    # Argument parser for CLI
    parser = argparse.ArgumentParser(description="Train a classification model based on a YAML config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Load configuration from YAML
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'experiments' not in config or not config['experiments']:
                raise ValueError("YAML must contain an 'experiments' list with at least one experiment.")
            exp_config = config['experiments'][0]
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error processing YAML file: {e}")
        exit(1)

    # Extract experiment details
    exp_name = exp_config['name']
    model_key = exp_config['model']
    dataset_key = exp_config['dataset']
    hp = exp_config['hyperparameters']
    
    # Checkpoint directory
    checkpoint_dir = config.get('checkpoint_dir', "./checkpoints")
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    
    print(f"\nRunning Experiment: {exp_name}")
    print(f"Model: {model_key}, Dataset: {dataset_key}")
    print(f"Hyperparameters: {hp}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Map model names to classes
    model_classes = {
        'LeNet': LeNet,
        'GoogleNet': GoogleNet,
        'ResNet18': ResNet18,
        'ResNet50': ResNet50
    }
    if model_key not in model_classes:
        print(f"Error: Model '{model_key}' not found in model_classes mapping.")
        exit(1)
    ModelClass = model_classes[model_key]

    #  Load and split dataset (generalized, keep comments) 
    print(f"Loading dataset: {dataset_key}")
    try:
        dataset_info = config['datasets'][dataset_key]
        num_classes = dataset_info['classes']

        # Chọn class dataset
        if dataset_key == 'mnist':
            DatasetClass = MNIST.MNISTDataset
            train_data = DatasetClass(
                images_filepath=dataset_info['train_images'],
                labels_filepath=dataset_info['train_labels']
            )
            test_data = DatasetClass(
                images_filepath=dataset_info['test_images'],
                labels_filepath=dataset_info['test_labels']
            )
            val_images = dataset_info.get('val_images')
            val_labels = dataset_info.get('val_labels')
            if val_images and val_labels:
                print(f"Loading validation data from specified path...")
                val_data = DatasetClass(
                    images_filepath=val_images,
                    labels_filepath=val_labels
                )
            else:
                # Split train data into train and validation (80/20) using stratified split
                print("No 'val_path' found. Splitting training data into 80% train / 20% validation using stratified split.")
                indices = list(range(len(train_data)))
                labels = train_data.labels
                train_idx, val_idx = train_test_split(
                    indices, test_size=0.2, stratify=labels, random_state=seed
                )
                train_data = Subset(train_data, train_idx)
                val_data = Subset(train_data.dataset, val_idx)
                print(f"Splitting train data: {len(train_data)} for training, {len(val_data)} for validation.")
        elif dataset_key == 'vinafood':
            DatasetClass = ViNaFood21.ViNaFood21Dataset
            train_data_full = DatasetClass(path=dataset_info['train_path'], is_train=True)
            test_data = DatasetClass(path=dataset_info['test_path'], is_train=False)
            val_path = dataset_info.get('val_path')
            if val_path:
                print(f"Loading validation data from: {val_path}")
                val_data = DatasetClass(path=val_path, is_train=False)
                train_data = train_data_full
            else:
                print("No 'val_path' found. Splitting training data into 80% train / 20% validation using stratified split.")
                indices = list(range(len(train_data_full)))
                labels = train_data_full.labels
                train_idx, val_idx = train_test_split(
                    indices, test_size=0.2, stratify=labels, random_state=seed
                )
                train_data = Subset(train_data_full, train_idx)
                val_data = Subset(train_data_full, val_idx)
                print(f"Splitting train data: {len(train_data)} for training, {len(val_data)} for validation.")
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        # Lấy thông tin gốc để tính class weights
        original_dataset = train_data.dataset if isinstance(train_data, Subset) else train_data
        original_labels = original_dataset.labels
        original_idx2label = getattr(original_dataset, "idx2label", None)
    except Exception as e:
        print(f"Error loading or splitting dataset: {e}")
        exit(1)

    # Handle edge case: get item from Subset or Dataset
    if isinstance(train_data, torch.utils.data.Subset):
        sample_item = train_data.dataset[train_data.indices[0]]
    else:
        sample_item = train_data[0]
    image_size = sample_item['image'].shape

    num_labels = len(set(original_labels))
    print(f"Sample image size: {image_size}, Number of classes: {num_labels}")

    # Initialize model
    model = ModelClass(in_channels=image_size[0], num_classes=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Print mapping label id to class name and distribution
    label_counts = Counter(original_labels)
    total_samples = len(original_labels)
    print(f"\nClass Distribution (from original train set):")
    print("="*70)

    if original_idx2label:
        for label_id in sorted(original_idx2label.keys()):
            count = label_counts.get(label_id, 1)
            class_name = original_idx2label[label_id]
            print(f"{label_id:<10d} {class_name:<30s} {count:<10d}")
    else:
        for label_id in sorted(label_counts.keys()):
            count = label_counts[label_id]
            print(f"{label_id:<10d} {'N/A':<30s} {count:<10d}")
    print("="*70)
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # DataLoaders
    batch_size = hp['batch_size']
    num_workers = min(os.cpu_count(), 8)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=g
    )

    # Optimizer
    optimizer_name = hp['optimizer']
    optimizer_params = hp.get('optimizer_params', {})
    learning_rate = hp['lr']
    weight_decay = hp['weight_decay']
    step_size = hp['step_size']
    gamma = hp['gamma']

    print(f"Using optimizer: {optimizer_name} with LR={learning_rate}, WeightDecay={weight_decay}")
    try:
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_params)
        elif optimizer_name.lower() == "sgd":
            if 'momentum' not in optimizer_params:
                optimizer_params['momentum'] = 0.9
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        exit(1)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Trainer initialization
    trainer = classification_engine.ClassificationTraining(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    # Training loop
print(f"START TRAINING {exp_name}...")
model_filename = f"{exp_name}.pth"
checkpoint_path = os.path.join(checkpoint_dir,  "last_checkpoint.pth")

# Load checkpoint if exists
start_epoch = trainer.load_checkpoint(checkpoint_path)
if start_epoch > 0:
    # Hiển thị best val loss và best val f1 nếu có
    if hasattr(trainer, "best_epoch") and trainer.best_epoch != -1:
        print(f"Best validation at epoch {trainer.best_epoch}: Val_Loss={trainer.best_val_loss:.4f}, Val_F1={trainer.best_val_f1:.4f}")
    print(f"Resuming training from epoch {start_epoch}")
    
# Get early stopping patience from config, default to 10
early_stop_patience = hp.get('early_stop_patience', 10)

# Train the model
results, actual_epochs_ran = trainer.train(
    epochs=hp['epochs'],
    target_dir=checkpoint_dir,
    model_name=model_filename,
    start_epoch=start_epoch,
    early_stop_epochs=early_stop_patience
)

print(f"DONE TRAINING {exp_name}. Ran for {actual_epochs_ran} epochs.")

# Run final evaluation on the (unseen) test set
print(f"\n[INFO] Starting final evaluation on the (unseen) test set...")

# Load the best model saved during training for final evaluation
best_model_path = os.path.join(checkpoint_dir, f"{exp_name}.pth")

print(f"Loading best model from {best_model_path} for final test evaluation...")
try:
    trainer.load_checkpoint(best_model_path)
    test_metrics, test_report_str = trainer.evaluate(test_dataloader)
    print(f"[INFO] Test Set Results: Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['acc']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    print("\nClassification report for final (unseen) test set:")
    print(test_report_str)
except Exception as e:
    print(f"[ERROR] Error during final test set evaluation: {e}")

# Plot training metrics
print(f"Plotting training/validation results for {exp_name}.")
plot_metrics(results, epochs=actual_epochs_ran, model_name=exp_name, dataset_name=dataset_key)

print(f"\nExperiment {exp_name} finished successfully!")