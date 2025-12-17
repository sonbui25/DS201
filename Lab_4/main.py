# -*- coding: utf-8 -*-
import os
import random
import warnings
import argparse
import yaml
import logging
from collections import Counter
from tabulate import tabulate
import numpy as np
import torch
# Import DataLoader and Subset/split tools
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Import models, dataset classes and task engine
from models import LSTM, LSTM_Bahdanau, LSTM_Local_Attention
from tasks import seq2seq_engine
from dataloaders import PhoMTDataset
from utils.utils import plot_metrics, collate_fn
from utils.vocab import Vocab
from functools import partial

# Setup warnings
warnings.filterwarnings("ignore", message=".*number of unique classes.*", category=UserWarning)

# --- NEW: Hàm thiết lập Logger dùng chung ---
def setup_main_logger(log_file_path):
    """Thiết lập logger cho hàm main, ghi vào file và console"""
    # Tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logger = logging.getLogger("Main_Logger")
    logger.setLevel(logging.INFO)

    # Clear handlers cũ nếu có để tránh duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. File Handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger
# ---------------------------------------------

def seed_worker(worker_id):
    """Seed function for DataLoader workers"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# MAIN EXECUTION
def main():
    # Argument parser for CLI
    parser = argparse.ArgumentParser(description="Train a classification model based on a YAML config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # --- Đọc config sơ bộ để lấy tên experiment và setup logger trước ---
    config_path = args.config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'experiments' not in config or not config['experiments']:
                raise ValueError("YAML must contain an 'experiments' list.")
            exp_config = config['experiments'][0]
    except Exception as e:
        print(f"Critical Error loading config for logging setup: {e}")
        exit(1)

    exp_name = exp_config['name']
    checkpoint_dir = config.get('checkpoint_dir', "./checkpoints")
    
    #Task
    task = config['vocab']['task_type']
    
    # Tạo thư mục cho từng experiment
    if task == "machine_translation":
        exp_dir = os.path.join('/kaggle/working/DS201/Lab_4/results', exp_name)
        log_file_path = os.path.join(exp_dir, f"{exp_name}_training.log")
        output_log_path = os.path.join(exp_dir, f"{exp_name}_test_predictions.jsonl")
    
    # KHỞI TẠO LOGGER
    logger = setup_main_logger(log_file_path)
    
    logger.info("="*50)
    logger.info(f"STARTING PROCESS FOR EXPERIMENT: {exp_name}")
    logger.info(f"Log file location: {log_file_path}")
    logger.info(f"Loading configuration from: {args.config}")
    # -------------------------------------------------------------------

    # Extract experiment details
    model_key = exp_config['model']
    dataset_key = exp_config['dataset']
    hp = exp_config['hyperparameters']
    
    # Set random seeds
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    
    logger.info(f"\nRunning Experiment: {exp_name}")
    logger.info(f"Model: {model_key}, Dataset: {dataset_key}")
    logger.info(f"Hyperparameters: {hp}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        
    # Load tokenizer
    vocab = Vocab(config['vocab'])
    
    
    # Map model names to classes
    model_classes = {
        'LSTM': LSTM,
        'LSTM_Bahdanau': LSTM_Bahdanau,
        'LSTM_Local_Attention': LSTM_Local_Attention
    }
    if model_key not in model_classes:
        logger.error(f"Error: Model '{model_key}' not found in model_classes mapping.")
        exit(1)
    ModelClass = model_classes[model_key]
    DatasetClass = None  # Placeholder for dataset class
    # Load and split dataset
    logger.info(f"Loading dataset: {dataset_key}")
    try:
        dataset_info = config['datasets'][dataset_key]

        if dataset_key == 'pho_mt':
            DatasetClass = PhoMTDataset
            
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        train_data = DatasetClass(path=dataset_info['train_path'], vocab=vocab, config=config)
        test_data = DatasetClass(path=dataset_info['test_path'], vocab=vocab, config=config)
        val_data = DatasetClass(path=dataset_info['val_path'], vocab=vocab, config=config)
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
                
    except Exception as e:
        logger.exception(f"Error loading or splitting dataset: {e}")
        exit(1)
        
    # Initialize model
    model = ModelClass(vocab=vocab, config=hp).to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
   
    # DataLoaders
    batch_size = hp['batch_size']
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer_name = hp['optimizer']
    optimizer_params = hp.get('optimizer_params', {})
    learning_rate = hp['lr']
    weight_decay = hp['weight_decay']
    step_size = hp['step_size']
    gamma = hp['gamma']

    logger.info(f"Using optimizer: {optimizer_name} with LR={learning_rate}, WeightDecay={weight_decay}")
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
        logger.error(f"Error initializing optimizer: {e}")
        exit(1)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Trainer initialization
    trainer_classes = {
        'machine_translation': seq2seq_engine.Seq2SeqTraining
    }

    if task not in trainer_classes:
        raise ValueError(f"Unsupported task type: {task}")

    TrainerClass = trainer_classes[task]
    trainer = TrainerClass(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        logger=logger,
        vocab=vocab
    )
    # Training loop
    logger.info(f"START TRAINING {exp_name}...")
    checkpoint_path = f"{exp_name}_last_checkpoint.pth"
    last_check_point_path = os.path.join(checkpoint_dir, checkpoint_path)

    # Load checkpoint if exists
    start_epoch = trainer.load_checkpoint_for_resume(last_check_point_path)
    if start_epoch > 0:
        if hasattr(trainer, "best_epoch") and trainer.best_epoch != -1:
            logger.info(f"Best validation at epoch {trainer.best_epoch}: Val_Loss={trainer.best_val_loss:.4f}, Val_F1={trainer.best_val_f1:.4f}")
        logger.info(f"Resuming training from epoch {start_epoch}")
        
    early_stop_patience = hp.get('early_stop_patience', 10)

    # --- LƯU Ý: trainer.train cũng sẽ ghi vào file log này vì chúng ta dùng mode='a' ---
    results, actual_epochs_ran = trainer.train(
        epochs=hp['epochs'],
        target_dir=checkpoint_dir,
        model_name=exp_name,
        start_epoch=start_epoch,
        early_stop_epochs=early_stop_patience
    )

    logger.info(f"DONE TRAINING {exp_name}. Ran for {actual_epochs_ran} epochs.")

    # Run final evaluation
    logger.info(f"\n[INFO] Starting final evaluation on the (unseen) test set...")
    best_model_path = os.path.join(checkpoint_dir, f"{exp_name}_best_model.pth")
    logger.info(f"Loading best model from {best_model_path} for final test evaluation...")

    try:
        trainer.load_best_model_for_eval(best_model_path)
        logger.info("[INFO] Best model loaded successfully.")
        test_metrics = trainer.evaluate(test_dataloader, output_log_path=output_log_path)
        logger.info("[INFO] Test evaluation completed successfully.")
        logger.info(f"[INFO] Test Set Results: Loss: {test_metrics['loss']:.4f}, ROUGE-L: {test_metrics['rouge_L']:.4f}")
    except Exception as e:
        logger.error(f"[ERROR] Error during final test set evaluation: {e}")

    logger.info(f"Plotting training/validation results for {exp_name}.")
    plot_metrics(results, epochs=actual_epochs_ran, model_name=exp_name, dataset_name=dataset_key)

    logger.info(f"\nExperiment {exp_name} finished successfully!")

if __name__ == "__main__":
    main()