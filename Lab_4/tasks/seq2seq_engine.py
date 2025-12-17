# -*- coding: utf-8 -*-
import os
import logging
import json
from pathlib import Path
import random
from tabulate import tabulate
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report
import numpy as np
from rouge_score import rouge_scorer
from utils.vocab import Vocab
"""
This module defines the ClassificationTraining engine responsible for handling
the training, validation, evaluation, and checkpointing logic.
"""

class Seq2SeqTraining():
    """
    A training engine for classification tasks.
    Manages the training loop, validation loop, metric calculation,
    early stopping, and model checkpointing.
    """
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn, optimizer, device, scheduler, logger, vocab: Vocab):
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.vocab = vocab
        self.best_rouge_L = 0.0
        self.best_epoch = -1
        
        # Logger placeholder
        self.logger = logger

    def log(self, message):
        """Helper to log to file if logger exists, else print to console"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_step(self, epoch_pbar=None) -> Tuple[float, float]:
        """Performs a single training step (one epoch)"""
        self.model.train()
        train_loss, train_rouge_L = [], []

        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X, y) # (batch_size, seq_len, vocab_size)
            # Reshape for loss computation
            y_pred_flat = y_pred.view(-1, y_pred.shape[-1]) # (batch_size*seq_len, vocab_size)
            
            # Compute loss
            # Shift y for loss computation (remove <BOS>)
            y_target = y[:, 1:] # (batch_size, seq_len)
            y_target_flat = y_target.reshape(-1) # (batch_size*seq_len)
            
            loss = self.loss_fn(y_pred_flat, y_target_flat)
                
            train_loss.append(loss.item()) # Loss for this batch
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            y_pred_class = torch.argmax(y_pred, dim=2) # (batch_size, seq_len)
            # print(y_target.shape)
            # print(y_pred_class.shape)
            # Loop qua từng sample trong batch
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            for i in range(y_target.shape[0]):  # Iterate batch size
                ref_list = self.vocab.decode_sentence(y_target[i].unsqueeze(0), self.vocab.tgt_language)
                references = ref_list[0]
                pred_list = self.vocab.decode_sentence(y_pred_class[i].unsqueeze(0), self.vocab.tgt_language)
                predictions = pred_list[0]
                rouge_L = scorer.score(references, predictions)['rougeL'].fmeasure
                train_rouge_L.append(rouge_L)
            
            # update postfix on outer epoch tqdm so loss can be monitored
            if epoch_pbar is not None:
                epoch_pbar.set_postfix({"train_loss":f"{np.mean(train_loss):.3f}"})

        return (np.mean(train_loss), np.mean(train_rouge_L))

    def val_step(self) -> Tuple[float, float]:
        """Performs a single validation step (one epoch)"""
        self.model.eval()
        y_true_all, y_pred_all = [], []
        val_loss = []
        val_rouge_L = []

        with torch.inference_mode():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                # Reshape for loss computation
                y_pred_flat = y_pred.view(-1, y_pred.shape[-1]) # (batch_size*seq_len, vocab_size)
                
                # Compute loss
                # Shift y for loss computation (remove <BOS>)
                y_target = y[:, 1:]
                y_target_flat = y_target.reshape(-1) # (batch_size*seq_len)
                
                loss = self.loss_fn(y_pred_flat, y_target_flat)
                    
                val_loss.append(loss.item()) # Loss for this batch
                
                y_pred_class = torch.argmax(y_pred, dim=2)
                
                # Compute ROUGE-L for each sample in batch
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                for i in range(y_target.shape[0]):
                    ref_list = self.vocab.decode_sentence(y_target[i].unsqueeze(0), self.vocab.tgt_language)
                    references = ref_list[0]
                    
                    pred_list = self.vocab.decode_sentence(y_pred_class[i].unsqueeze(0), self.vocab.tgt_language)
                    predictions = pred_list[0]
                    
                    rouge_L = scorer.score(references, predictions)['rougeL'].fmeasure
                    val_rouge_L.append(rouge_L)
                                    

        val_loss = np.mean(val_loss)
        val_rouge_L = np.mean(val_rouge_L)
        return (val_loss, val_rouge_L)

    def train(
        self,
        epochs: int,
        model_name: str,
        target_dir: str = "./checkpoints",
        start_epoch: int = 0,
        early_stop_epochs: int = 10
    ) -> Tuple[Dict[str, List], int]:
        """
        Main training loop.
        """
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # --------------------

        base_name = os.path.splitext(model_name)[0]
        best_model_path = os.path.join(target_dir, f"{base_name}_best_model.pth")
        last_ckpt_path = os.path.join(target_dir, f"{base_name}_last_checkpoint.pth")
        
        results = {
            "train_loss": [],
            "val_loss": [], "val_rouge_L": []
        }
        headers = [
            "Epoch", "Train Loss",
            "Val Loss", "Val ROUGE-L"
        ]
        
        epochs_no_improve = 0
        actual_epochs_ran = start_epoch
        pbar_total = epochs - start_epoch
        pbar = tqdm(range(start_epoch, epochs), desc="Epoch", total=pbar_total, initial=start_epoch)
        
        for epoch in pbar:
            
            train_loss, train_rouge_L = self.train_step(epoch_pbar=pbar)
            val_loss, val_rouge_L = self.val_step()

            row = [
                epoch, f"{train_loss:.4f}",
                f"{val_loss:.4f}", f"{val_rouge_L:.4f}"
            ]

            results["train_loss"].append(train_loss)
            results["train_rouge_L"].append(train_rouge_L)
            results["val_loss"].append(val_loss)
            results["val_rouge_L"].append(val_rouge_L)

            if self.scheduler is not None:
                self.scheduler.step()

            # --- LOGGING TABLE ---
            table_str = tabulate([row], headers=headers, tablefmt="github")
            self.log("\n" + table_str)
            # ---------------------

            actual_epochs_ran = epoch + 1

            # Early stopping & checkpoint logic
            if val_rouge_L > self.best_val_rouge_L:
                self.best_val_loss = val_loss
                self.best_val_rouge_L = val_rouge_L
                self.best_epoch = epoch
                self.save_best_model(best_model_path, epoch)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stop_epochs:
                self.log(f"TRIGGERED EARLY STOP AT {epoch}! (Val ROUGE-L not improved for {early_stop_epochs} epochs)")
                break
            # Always save last checkpoint
            self.save_last_checkpoint(last_ckpt_path, epoch)
            
        self.log("\nBest result based on ROUGE-L on validation set:")
        self.log(f"Best epoch: {self.best_epoch}, Best Val_Loss: {self.best_val_loss:.4f}, Val_ROUGE-L: {self.best_val_rouge_L:.4f}")
        
        return results, actual_epochs_ran

    def evaluate(self, dataloader: torch.utils.data.DataLoader, output_log_path: str = None) -> Tuple[Dict[str, float], str]:
        """
        Evaluates the model on a given dataloader (the final test set).
        """
        self.model.eval()
        test_loss = 0
        test_rouge_L = []
        y_true_all, y_pred_all = [], []
        predictions_log = []  # Store predictions for logging

        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Evaluating Test Set"):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                # Reshape for loss computation
                y_pred_flat = y_pred.view(-1, y_pred.shape[-1]) # (batch_size*seq_len, vocab_size)
                
                # Compute loss
                # Shift y for loss computation (remove <BOS>)
                y_target = y[:, 1:]
                y_target_flat = y_target.reshape(-1) # (batch_size*seq_len)
                
                loss = self.loss_fn(y_pred_flat, y_target_flat)
                    
                test_loss.append(loss.item()) # Loss for this batch
                
                y_pred_class = torch.argmax(y_pred, dim=2)
                
                # Compute ROUGE-L for each sample in batch
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                for i in range(y_target.shape[0]):
                    ref_list = self.vocab.decode_sentence(y_target[i].unsqueeze(0), self.vocab.tgt_language)
                    references = ref_list[0]
                    
                    pred_list = self.vocab.decode_sentence(y_pred_class[i].unsqueeze(0), self.vocab.tgt_language)
                    predictions = pred_list[0]
                    
                    rouge_L = scorer.score(references, predictions)['rougeL'].fmeasure
                    test_rouge_L.append(rouge_L)
                    
                    # Store prediction log if dataset has source/target text
                    if hasattr(dataloader.dataset, 'data'):
                        sample_data = dataloader.dataset.data[i] if i < len(dataloader.dataset.data) else {}
                        src_text = sample_data.get(self.vocab.src_language, "")
                        tgt_text = sample_data.get(self.vocab.tgt_language, "")
                    else:
                        src_text = ""
                        tgt_text = ""
                    
                    predictions_log.append({
                        f"{self.vocab.src_language}_source": src_text,
                        f"{self.vocab.tgt_language}_gold_label": references,
                        "prediction": predictions
                    })
                                  

        test_loss /= len(dataloader)
        test_metrics = {
            "loss": test_loss,
            "rouge_L": np.mean(test_rouge_L)
        }
        
        # Save predictions log to file
        if output_log_path:
            self._save_predictions_log(output_log_path, predictions_log)
        
        return test_metrics
    
    def _save_predictions_log(self, filepath: str, predictions: List[Dict]) -> None:
        """Save predictions to a JSON log file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        self.log(f"[INFO]: Predictions log saved to {filepath}")

    def save_best_model(self, path: str, epoch: int) -> None:
        dir_path = os.path.dirname(path) or "."
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model.state_dict()
        )

        self.log(f"\n[INFO]: Saving best model at epoch {epoch} to: {path}")
        torch.save({
            "model_state": model_state,
            "best_val_loss": self.best_val_loss,
            "best_val_rouge_L": self.best_val_rouge_L,
            "best_epoch": epoch
        }, path)

    def save_last_checkpoint(self, path: str, epoch: int) -> None:
        dir_path = os.path.dirname(path) or "."
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model.state_dict()
        )

        torch.save({
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "best_val_rouge_L": self.best_val_rouge_L,
            "best_epoch": self.best_epoch,
            "epoch": epoch
        }, path)
    
    def load_best_model_for_eval(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No best model found at {path}")
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt['model_state']
        
        is_dp_model = isinstance(self.model, torch.nn.DataParallel)
        has_module_prefix = list(state_dict.keys())[0].startswith("module.")
        if is_dp_model and not has_module_prefix:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not is_dp_model and has_module_prefix:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.best_val_rouge_L = ckpt.get('best_val_rouge_L', self.best_val_rouge_L)
        self.best_val_loss = ckpt.get('best_val_loss', self.best_val_loss)
        self.best_epoch = ckpt.get('best_epoch', self.best_epoch)

        return self.best_epoch
    
    def load_checkpoint_for_resume(self, path):
        if not os.path.exists(path):
            self.log(f"No checkpoint found at {path}")
            return 0
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt['model_state']
        
        is_dp_model = isinstance(self.model, torch.nn.DataParallel)
        has_module_prefix = list(state_dict.keys())[0].startswith("module.")
        if is_dp_model and not has_module_prefix:
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        elif not is_dp_model and has_module_prefix:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)

        if 'optimizer_state' in ckpt and ckpt['optimizer_state'] is not None:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception as e:
                self.log(f"Warning: couldn't load optimizer state: {e}")
        
        if self.scheduler and 'scheduler_state' in ckpt and ckpt['scheduler_state'] is not None:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state'])
            except Exception as e:
                self.log(f"Warning: couldn't load scheduler state: {e}")

        self.best_val_rouge_L = ckpt.get('best_val_rouge_L', self.best_val_rouge_L)
        self.best_val_loss = ckpt.get('best_val_loss', self.best_val_loss)
        self.best_epoch = ckpt.get('best_epoch', self.best_epoch)

        last_epoch = ckpt.get('epoch', None)
        if last_epoch is None:
            self.log("Warning: checkpoint has no 'epoch' key; resuming from epoch 0")
            return 0
        
        self.log(f"Checkpoint loaded successfully (last saved epoch {last_epoch})")
        return last_epoch + 1