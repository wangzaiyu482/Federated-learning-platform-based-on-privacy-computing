import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import numpy as np

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('start.log', mode='a',encoding='utf-8'), logging.StreamHandler()])
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        # 计算类别权重
        self.class_weights = self._calculate_class_weights(train_loader)
        self.criterion = nn.BCEWithLogitsLoss()  # 使用 BCEWithLogitsLoss
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 使用AdamW优化器

    def _calculate_class_weights(self, train_loader):
        """计算类别权重"""
        num_pos = 0
        num_neg = 0
        for _, labels in train_loader:
            num_pos += (labels == 1).sum().item()
            num_neg += (labels == 0).sum().item()

        pos_weight = torch.tensor([num_neg / num_pos])  # 少数类的权重
        return pos_weight

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _calculate_f1(self, predicted, labels):
        tp = ((predicted == 1) & (labels == 1)).sum().item()
        fp = ((predicted == 1) & (labels == 0)).sum().item()
        fn = ((predicted == 0) & (labels == 1)).sum().item()
        tn = ((predicted == 0) & (labels == 0)).sum().item()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        confusion_matrix = [[tn, fp], [fn, tp]]
        return f1, precision, recall, confusion_matrix

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)
            num_epochs = config.get("num_epochs", 1)
            self.model.train()
            device = self._move_to_device()

            for epoch in range(num_epochs):
                correct, total, total_loss = 0, 0, 0.0
                for inputs, labels in tqdm(self.train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    total_loss += loss.item()

            # 关键修改：转换为Python原生float
                accuracy = float(correct / total)
                avg_loss = float(total_loss / len(self.train_loader))

                f1, precision, recall, confusion_matrix = self._calculate_f1(predicted, labels)
                return (
                    self.get_parameters(config),
                    len(self.train_loader.dataset),
                    {
                        "accuracy": accuracy,  # 使用Python float
                        "loss": avg_loss,  # 使用Python float
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        # "confusion_matrix": confusion_matrix
                    }
                )
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            self.model.eval()
            # 定义 device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._move_to_device()  # 将模型和数据移动到设备
            correct, total, total_loss = 0, 0, 0.0
            with torch.no_grad(): 
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    total_loss += loss.item()

            accuracy = correct / total
            avg_loss = float(total_loss / len(self.val_loader))
            f1, precision, recall, confusion_matrix = self._calculate_f1(predicted, labels)
            logging.info(f"Validation Loss={total_loss / len(self.val_loader)}, Accuracy={accuracy}, F1={f1}")
            logging.info(f"Validation batch size: {self.val_loader.batch_size}, Num batches: {len(self.val_loader)}")
            return avg_loss, len(self.val_loader.dataset), {
                "accuracy": float(accuracy),
                "loss": avg_loss,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                # "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
            }
        except Exception as e:
            logging.error(f"Error during evaluation: {str(e)}")
            raise

    def _move_to_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        return device

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
