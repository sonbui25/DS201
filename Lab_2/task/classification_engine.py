# -*- coding: utf-8 -*-
import os
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

class ClassificationTraining():
    """
    A training engine for image classification tasks.
    Manages the training loop, validation loop, metric calculation,
    early stopping, and model checkpointing.
    """
    def __init__(self, model, train_dataloader, val_dataloader, loss_fn, optimizer, device, scheduler=None):
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

    def train_step(self) -> Tuple[float, float, float, float, float]:
        """Performs a single training step (one epoch)"""
        self.model.train()
        train_loss, train_acc, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            y_pred_class = torch.argmax(y_pred, dim=1)
            result_report = classification_report(y.cpu(), y_pred_class.cpu(), output_dict=True, zero_division=0)
            train_acc += result_report['accuracy']
            train_precision += result_report['macro avg']['precision']
            train_recall += result_report['macro avg']['recall']
            train_f1 += result_report['macro avg']['f1-score']

        len_data = len(self.train_loader)
        train_loss = train_loss / len_data
        train_acc = train_acc / len_data
        train_precision = train_precision / len_data
        train_recall = train_recall / len_data
        train_f1 = train_f1 / len_data

        return train_loss, train_acc, train_precision, train_recall, train_f1

    def val_step(self) -> Tuple[float, float, float, float, float]:
        """Performs a single validation step (one epoch)"""
        self.model.eval()
        val_loss = 0
        y_true_all, y_pred_all = [], []

        with torch.inference_mode():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                loss = self.loss_fn(y_logits, y)
                val_loss += loss.item()
                val_pred_label = torch.argmax(y_logits, dim=1)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(val_pred_label.cpu().numpy())

        val_loss /= len(self.val_loader)
        result_report = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        val_acc = result_report['accuracy']
        val_precision = result_report['macro avg']['precision']
        val_recall = result_report['macro avg']['recall']
        val_f1 = result_report['macro avg']['f1-score']

        return val_loss, val_acc, val_precision, val_recall, val_f1

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
        Saves the best model based on validation loss.
        Implements early stopping.
        """
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

        for epoch in tqdm(
            range(start_epoch, epochs),
            desc="Epoch",
            initial=start_epoch,
            total=epochs
        ):
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_step()
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
            self.scheduler.step()
            print("\n\n", tabulate([row], headers=headers, tablefmt="github"))
            actual_epochs_ran = epoch + 1

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_f1 = val_f1  # <-- lưu lại best val f1
                self.best_epoch = epoch
                self.save_best_model(target_dir, model_name, epoch)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve > early_stop_epochs:
                    print(f"TRIGGERED EARLY STOP AT {epoch}! (Val Loss not improved for {early_stop_epochs} epochs)")
                    break

            # Always save checkpoint for resume
            self.save_checkpoint(os.path.join(target_dir, "last_checkpoint.pth"), epoch)

        print("\nBest result based on Loss on validation set:")
        print(f"Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")
        return results, actual_epochs_ran

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], str]:
        """
        Evaluates the model on a given dataloader (e.g., the final test set).
        Returns a dictionary of metrics and a classification report string.
        """
        self.model.eval()
        test_loss = 0
        y_true_all, y_pred_all = [], []

        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Evaluating Test Set"):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
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

    def save_best_model(self, target_dir: str, model_name: str, epoch: int) -> None:
        """Saves the best model checkpoint (model, optimizer, and scheduler states)."""
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(target_dir, model_name)
        model_state = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        print(f"\n[INFO]: Saving best model at epoch {epoch} to: {save_path}")
        torch.save({
            'best_epoch': epoch,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1
        }, save_path)

    def save_checkpoint(self, path, epoch):
        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return 0  # train from beginning

        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt['model_state']

        # Xử lý DataParallel và non-DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            # Nếu state_dict không có prefix "module.", thêm vào
            if not list(state_dict.keys())[0].startswith("module."):
                new_state_dict = {"module."+k: v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
        else:
            # Nếu state_dict có prefix "module.", bỏ đi
            if list(state_dict.keys())[0].startswith("module."):
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)

        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        if self.scheduler and ckpt['scheduler_state']:
            self.scheduler.load_state_dict(ckpt['scheduler_state'])

        torch.set_rng_state(ckpt['rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
        np.random.set_state(ckpt['numpy_rng_state'])
        random.setstate(ckpt['python_rng_state'])
        self.best_val_f1 = ckpt.get('best_val_f1', 0.0) 
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.best_epoch = ckpt.get('best_epoch', -1)

        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt['epoch']