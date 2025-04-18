import os
import torch

import datetime
import matplotlib.pyplot as plt
from config import Config

class EarlyStopping:
    def __init__(self, patience=Config.patience, delta=Config.delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, save_dir):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model, save_dir)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))


def train_model(model, criterion, optimizer, scheduler, early_stopping,
                train_loader, val_loader, device, save_dir):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(1, Config.num_epochs + 1):
        print(f"Epoch {epoch}/{Config.num_epochs}")
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        print(f" Train  Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # 验证
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f" Val    Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 学习率调度（CosineAnnealingLR）
        scheduler.step()

        # 早停
        early_stopping(val_loss, model, save_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 加载最佳模型
    best_path = os.path.join(save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_path, map_location=device))

    # 绘制曲线
    _plot_history(train_losses, val_losses, train_accs, val_accs, save_dir)


def test_model(model, test_loader, device, save_dir):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            corrects += (outputs.argmax(1) == labels).sum().item()

    test_acc = corrects / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")


def _plot_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(); plt.title('Accuracy')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"history_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
    plt.savefig(path)
    print(f"Plot saved to {path}")
