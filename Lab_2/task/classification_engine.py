# -*- coding: utf-8 -*-
import os
from pathlib import Path
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
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader, # Used for validation during training
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler._LRScheduler # LR scheduler
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_step(self) -> Tuple[float, float, float, float, float]:
        """Performs a single training step (one epoch)"""
        self.model.train()
        train_loss, train_acc, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

        for data in self.train_dataloader:
            X, y = data['image'].to(self.device), data['label'].to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate and accumulate metrics across all batches
            # Note: Averaging batch-wise metrics is less precise but faster than
            # computing on the whole set, which is acceptable for training.
            y_pred_class = torch.argmax(y_pred, dim=1)
            result_report = classification_report(y.cpu(), y_pred_class.cpu(), output_dict=True, zero_division=0)
            train_acc += result_report['accuracy']
            train_precision += result_report['macro avg']['precision']
            train_recall += result_report['macro avg']['recall']
            train_f1 += result_report['macro avg']['f1-score']
        
        # Adjust metrics to get average loss and metrics per batch
        len_data = len(self.train_dataloader)
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

        # Turn on inference mode context manager
        with torch.inference_mode():
            for data in self.val_dataloader:
                X, y = data['image'].to(self.device), data['label'].to(self.device)
                y_logits = self.model(X)
                loss = self.loss_fn(y_logits, y)
                val_loss += loss.item()
                val_pred_label = torch.argmax(y_logits, dim=1)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(val_pred_label.cpu().numpy())

        # Adjust metrics to get average loss
        val_loss /= len(self.val_dataloader)

        # Calculate metrics once for the entire validation set
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
        Saves the best model based on validation F1-score.
        Implements early stopping.
        """
        # Create results dictionary
        results = {
            "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []
        }
        # Headers for logging table
        headers = [
            "Epoch", "Train Loss", "Train Acc", "Train Precision", "Train Recall", "Train F1",
            "Val Loss", "Val Acc", "Val Precision", "Val Recall", "Val F1"
        ]
        # Track best result and early stopping
        best_val_f1 = -float('inf')
        best_row = None
        epochs_no_improve = 0
        actual_epochs_ran = 0

        # Loop through training and validation steps for a number of epochs
        for epoch in tqdm(
            range(start_epoch, epochs),
            desc="Epoch",
            initial=start_epoch,
            total=epochs
        ):
            # Perform training and validation steps
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_step()
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.val_step()
            # Format row for printing
            row = [
                epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_precision:.4f}", f"{train_recall:.4f}", f"{train_f1:.4f}",
                f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_precision:.4f}", f"{val_recall:.4f}", f"{val_f1:.4f}"
            ]
            # Update results dictionary
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
            # Step the scheduler
            self.scheduler.step()
            # Print current epoch results
            print("\n\n", tabulate([row], headers=headers, tablefmt="github"))
            actual_epochs_ran = epoch + 1 # Track total epochs run

            # Check for best model and implement early stopping
            if val_f1 > best_val_f1: # Save best result based on val_f1
                best_val_f1 = val_f1
                best_row = row
                # Save the best model
                self.save_model(
                    target_dir=target_dir,
                    model_name=model_name,
                    epoch=epoch+1
                )
                epochs_no_improve = 0 # Reset counter
            else:
                epochs_no_improve += 1
                if epochs_no_improve > early_stop_epochs: # Triggered early stop
                    print(f"TRIGGERED EARLY STOP AT {epoch}! (Val F1 not improved for {early_stop_epochs} epochs)")
                    break # Exit training loop

        # Print best result on validation set
        print("\nBest result based on F1-score on validation set:")
        if best_row:
            print(tabulate([best_row], headers=headers, tablefmt="github"))
        return results, actual_epochs_ran

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], str]:
        """
        Evaluates the model on a given dataloader (e.g., the final test set).
        Returns a dictionary of metrics and a classification report string.
        """
        self.model.eval()
        test_loss = 0
        y_true_all, y_pred_all = [], []

        # Turn on inference mode context manager
        with torch.inference_mode():
            for data in tqdm(dataloader, desc="Evaluating Test Set"):
                X, y = data['image'].to(self.device), data['label'].to(self.device)
                y_logits = self.model(X)
                loss = self.loss_fn(y_logits, y)
                test_loss += loss.item()
                # Store predictions and true values
                test_pred_label = torch.argmax(y_logits, dim=1)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(test_pred_label.cpu().numpy())

        # Adjust metrics to get average loss
        test_loss /= len(dataloader)

        # Calculate metrics once for the entire set
        report_dict = classification_report(y_true_all, y_pred_all, output_dict=True, zero_division=0)
        report_str = classification_report(y_true_all, y_pred_all, zero_division=0)
        # Store metrics in a dictionary
        test_metrics = {
            "loss": test_loss,
            "acc": report_dict['accuracy'],
            "precision": report_dict['macro avg']['precision'],
            "recall": report_dict['macro avg']['recall'],
            "f1": report_dict['macro avg']['f1-score']
        }
        return test_metrics, report_str

    def save_model(self, target_dir: str, model_name: str, epoch: int) -> None:
        """Saves the model checkpoint (model, optimizer, and scheduler states)."""
        # Create target directory
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True, exist_ok=True)
        # Create model save path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        save_path = os.path.join(target_dir, model_name)
        # Get model state_dict (handle DataParallel)
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        # Save the checkpoint dictionary
        print(f"\n[INFO]: Saving best model at epoch {epoch} to: {save_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Loads a checkpoint from a given path."""
        if os.path.isfile(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                # Handle DataParallel wrapper (if model was saved with it)
                state_dict = checkpoint['model_state_dict']
                if isinstance(self.model, torch.nn.DataParallel):
                    # If current model is DataParallel, just load
                    self.model.load_state_dict(state_dict)
                else:
                    # If current model is not, but checkpoint might be, strip 'module.'
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        name = k.replace("module.", "") if k.startswith("module.") else k
                        new_state_dict[name] = v
                    self.model.load_state_dict(new_state_dict)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Load scheduler state if it exists
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")
                return start_epoch
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Training from scratch.")
                return 0
        else:
            print(f"No checkpoint found at '{checkpoint_path}', training from scratch.")
            return 0