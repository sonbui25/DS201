# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
import random
from tabulate import tabulate
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report
import numpy as np

"""
This module defines the ClassificationTraining engine responsible for handling
the training, validation, evaluation, and checkpointing logic.
"""

class SeqLabelingTraining():
    """
    A training engine for sequence labeling tasks.
    Manages the training loop, validation loop, metric calculation,
    early stopping, and model checkpointing.
    """
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn, optimizer, device, scheduler=None, logger=None):
        self.model = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_epoch = -1
        
        # Logger placeholder
        self.logger = logger

    def log(self, message):
        """Helper to log to file if logger exists, else print to console"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def train_step(self, epoch_pbar=None) -> Tuple[float, float, float, float, float]:
        """Performs a single training step (one epoch)"""
        self.model.train()
        train_loss, train_acc, train_precision, train_recall, train_f1 = [], [], [], [], []

        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_logits = self.model(X)
            y_logits = y_logits.view(-1, y_logits.shape[-1]) # [bs*seq_len, num_classes]
            y = y.view(-1) # [bs*seq_len]
            loss = self.loss_fn(y_logits, y)
            train_loss.append(loss.item()) # Loss for this batch
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            y_pred_class = torch.argmax(y_logits, dim=1)
            result_report = classification_report(y.cpu(), y_pred_class.cpu(), output_dict=True, zero_division=0)
            train_acc.append(result_report['accuracy'])
            train_precision.append(result_report['macro avg']['precision'])
            train_recall.append(result_report['macro avg']['recall'])
            train_f1.append(result_report['macro avg']['f1-score'])
            
            # update postfix on outer epoch tqdm so loss can be monitored
            if epoch_pbar is not None:
                epoch_pbar.set_postfix({"train_loss":f"{np.mean(train_loss):.3f}"})

        return (np.mean(train_loss), np.mean(train_acc), np.mean(train_precision), 
                np.mean(train_recall), np.mean(train_f1))

    def val_step(self) -> Tuple[float, float, float, float, float]:
        """Performs a single validation step (one epoch)"""
        self.model.eval()
        val_loss = []
        y_true_all, y_pred_all = [], []

        with torch.inference_mode():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                y_logits = y_logits.view(-1, y_logits.shape[-1]) # [bs*seq_len, num_classes]
                y = y.view(-1) # [bs*seq_len]
                loss = self.loss_fn(y_logits, y)
                val_loss.append(loss.item())
                val_pred_label = torch.argmax(y_logits, dim=1)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(val_pred_label.cpu().numpy())

        val_loss = np.mean(val_loss)
        result_report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        return (val_loss, result_report['accuracy'], result_report['macro avg']['precision'], 
                result_report['macro avg']['recall'], result_report['macro avg']['f1-score'])

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
       
        base_name = os.path.splitext(model_name)[0]
        best_model_path = os.path.join(target_dir, f"{base_name}_best_model.pth")
        last_ckpt_path = os.path.join(target_dir, f"{base_name}_last_checkpoint.pth")
        
        results = {
            "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []
        }
        headers = [
            "Epoch", "Train Loss", "Train Acc", "Train Precision", "Train Recall", "Train F1",
            "Val Loss", "Val Acc", "Val Precision", "Val Recall", "Val F1"
        ]
        
        epochs_no_improve = 0
        actual_epochs_ran = start_epoch
        pbar_total = epochs - start_epoch
        pbar = tqdm(range(start_epoch, epochs), desc="Epoch", total=pbar_total, initial=start_epoch)
        
        for epoch in pbar:
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_step(epoch_pbar=pbar)
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.val_step()
            
            row = [
                epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_precision:.4f}", f"{train_recall:.4f}", f"{train_f1:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}", f"{val_f1:.4f}"
            ]
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["train_precision"].append(train_precision)
            results["train_recall"].append(train_recall)
            results["train_f1"].append(train_f1)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["val_precision"].append(val_precision)
            results["val_recall"].append(val_recall)
            results["val_f1"].append(val_f1)
            
            if self.scheduler is not None:
                self.scheduler.step()

            # --- LOGGING TABLE ---
            table_str = tabulate([row], headers=headers, tablefmt="github")
            self.log("\n" + table_str)
            # ---------------------

            actual_epochs_ran = epoch + 1
            
            # Early stopping & checkpoint logic
            if val_f1 > self.best_val_f1:
                self.best_val_loss = val_loss
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.save_best_model(best_model_path, epoch)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve > early_stop_epochs:
                    self.log(f"TRIGGERED EARLY STOP AT {epoch}! (Val F1 not improved for {early_stop_epochs} epochs)")
                    break
            
            # Always save last checkpoint
            self.save_last_checkpoint(last_ckpt_path, epoch)
            
        self.log("\nBest result based on F1 on validation set:")
        self.log(f"Best epoch: {self.best_epoch}, Best Val_Loss: {self.best_val_loss:.4f}, Val_F1: {self.best_val_f1:.4f}")
        
        # # Cleanup handlers
        # if self.logger:
        #     for handler in self.logger.handlers:
        #         handler.close()
        #         self.logger.removeHandler(handler)
        #     self.logger = None

        return results, actual_epochs_ran

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], str]:
        """
        Evaluates the model on a given dataloader (the final test set).
        """
        self.model.eval()
        test_loss = 0
        y_true_all, y_pred_all = [], []

        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Evaluating Test Set"):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                y_logits = y_logits.view(-1, y_logits.shape[-1])  # [bs*seq_len, num_classes]
                y = y.view(-1)  # [bs*seq_len]
                loss = self.loss_fn(y_logits, y)
                test_loss += loss.item()
                test_pred_label = torch.argmax(y_logits, dim=1)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(test_pred_label.cpu().numpy())

        test_loss /= len(dataloader)
        report_dict = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        report_str = classification_report(y_true_all, y_pred_all, zero_division=0)
        test_metrics = {
            "loss": test_loss,
            "acc": report_dict['accuracy'],
            "precision": report_dict['macro avg']['precision'],
            "recall": report_dict['macro avg']['recall'],
            "f1": report_dict['macro avg']['f1-score']
        }

        return test_metrics, report_str

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
            "best_val_f1": self.best_val_f1,
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
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "epoch": epoch
        }, path)
    
    def load_best_model_for_eval(self, path):
        if not os.path.exists(path):
            self.log(f"[ERROR] No best model found at {path}")
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
        self.best_val_f1 = ckpt.get('best_val_f1', self.best_val_f1)
        self.best_val_loss = ckpt.get('best_val_loss', self.best_val_loss)
        self.best_epoch = ckpt.get('best_epoch', self.best_epoch)

        self.log(f"[INFO] Loaded best model from {path} for evaluation.")
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

        self.best_val_f1 = ckpt.get('best_val_f1', self.best_val_f1)
        self.best_val_loss = ckpt.get('best_val_loss', self.best_val_loss)
        self.best_epoch = ckpt.get('best_epoch', self.best_epoch)

        last_epoch = ckpt.get('epoch', None)
        if last_epoch is None:
            self.log("Warning: checkpoint has no 'epoch' key; resuming from epoch 0")
            return 0
        
        self.log(f"Checkpoint loaded successfully (last saved epoch {last_epoch})")
        return last_epoch + 1