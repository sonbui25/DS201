from pathlib import Path
from tabulate import tabulate
import torch
from  tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report
class ClassificationTraining():
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
    def train_step(self) -> Tuple[float, float, float, float, float]:
        # Put model in train mode
        self.model.train()
        train_loss, train_acc, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

        # Setup train loss and train accuracy values
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)
            # 1. Forward pass
            y_pred = self.model(X)
            # 2. Calculate loss/accuracy
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            #5. Optimizer step
            self.optimizer.step()
            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            result_report = classification_report(y.cpu(), y_pred_class.cpu(), output_dict=True, zero_division=0)
            train_acc += result_report['accuracy']
            train_precision += result_report['macro avg']['precision']
            train_recall += result_report['macro avg']['recall']
            train_f1 += result_report['macro avg']['f1-score']
        # Adjust metrics to get average loss and accuracy per batch
        len_data = len(self.train_dataloader)
        train_loss = train_loss / len_data
        train_acc = train_acc / len_data
        train_precision = train_precision / len_data
        train_recall = train_recall / len_data
        train_f1 = train_f1 / len_data
        return train_loss, train_acc, train_precision, train_recall, train_f1

    def test_step(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
        # Begin test mode
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_dataloader):

                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)

                loss = self.loss_fn(y_logits, y)

                test_pred_label = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
                test_loss += loss.item()
                result_report = classification_report(y.cpu(), test_pred_label.cpu(), output_dict=True, zero_division=0)
                test_acc += result_report['accuracy']
                test_precision += result_report['macro avg']['precision']
                test_recall += result_report['macro avg']['recall']
                test_f1 += result_report['macro avg']['f1-score']
        len_data = len(self.test_dataloader)
        test_loss = test_loss / len_data
        test_acc = test_acc / len_data
        test_precision = test_precision / len_data
        test_recall = test_recall / len_data
        test_f1 = test_f1 / len_data
        return test_loss, test_acc, test_precision, test_recall, test_f1

    def train(self, epochs: int, model_name: str, target_dir: str = "./checkpoints") -> Dict[str, List]:
        results = {"train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
                "test_loss": [], "test_acc": [], "test_precision": [], "test_recall": [], "test_f1": []}
        headers = ["Epoch", "Train Loss", "Train Acc", "Train Precision", "Train Recall", "Train F1",
                "Test Loss", "Test Acc", "Test Precision", "Test Recall", "Test F1"]
        table = []

        # Save best result
        best_test_f1 = 0
        best_row = None

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_step()
            test_loss, test_acc, test_precision, test_recall, test_f1 = self.test_step()
            row = [epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_precision:.4f}", f"{train_recall:.4f}", f"{train_f1:.4f}",
                f"{test_loss:.4f}", f"{test_acc:.4f}", f"{test_precision:.4f}", f"{test_recall:.4f}", f"{test_f1:.4f}"]
            table.append(row)
            #Update results train
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["train_precision"].append(train_precision)
            results["train_recall"].append(train_recall)
            results["train_f1"].append(train_f1)
            #Update results test
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            results["test_precision"].append(test_precision)
            results["test_recall"].append(test_recall)
            results["test_f1"].append(test_f1)

            print("\n\n", tabulate([row], headers=headers, tablefmt="github"))
            # Save best result based on test_f1
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_row = row
                #Save the best model
                self.save_model(target_dir=target_dir,
                                model_name=model_name,
                                epoch=epoch+1)


        # Print best result on test set
        print("\nBest result based on F1-score on test set:")
        print(tabulate([best_row], headers=headers, tablefmt="github"))

        # Predict on entire test set at last epoch
        y_true_all = []
        y_pred_all = []
        self.model.eval()
        with torch.inference_mode():
            for batch, (X, y) in enumerate(self.test_dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_logits = self.model(X)
                test_pred_label = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(test_pred_label.cpu().numpy())

        print("\nClassification report for last epoch on test set:")
        print(classification_report(y_true_all, y_pred_all, zero_division=0))
        
        return results
    def save_model(self,
                target_dir: str,
                model_name: str,
                epoch: int) -> None:
        #Create target directory
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True,
                            exist_ok=True)
        
        #Create model save path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name

        #Save the model state_dict()
        print(f"[INFO] Saving best model at epoch {epoch} to: {model_save_path}")
        torch.save(obj=self.model.state_dict(),
                f=model_save_path)