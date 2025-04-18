import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import copy
import datetime
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=10, delta=0.0005):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, save_dir):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        self.val_loss_min = val_loss

def train_model(model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader, num_epochs, device, train_num, save_dir):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu().numpy())  # 将张量移动到 CPU 并转换为 NumPy 数组

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_loss)
        val_accuracies.append(epoch_acc.cpu().numpy())  # 将张量移动到 CPU 并转换为 NumPy 数组

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 学习率调度
        scheduler.step(epoch_loss)

        # 早停检查
        early_stopping(epoch_loss, model, save_dir)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 加载最佳模型权重
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device))
    # 绘制训练和验证曲线
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)

def validate_model(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def test_model(model, test_loader, device, save_dir):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')

    # 保存测试结果
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        f.write(f'Test Accuracy: {test_acc:.4f}')

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_dir):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_plot_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")