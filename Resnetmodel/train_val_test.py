import os
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_model
from data_loader import WheatDiseaseDataset, get_data_loaders
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=200, device='cuda',
                train_num=1, save_dir=None):
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 确保 save_dir 存在
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f'training_log_{timestamp}.txt')
    else:
        log_file = f'training_log_{timestamp}.txt'

    with open(log_file, 'w') as file:
        file.write(f'Training Log - Timestamp: {timestamp}\n')

    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        val_loss, val_acc = validate_model(model, criterion, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 更新学习率
        scheduler.step(val_loss)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth') if save_dir else 'best_model.pth')

        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

    plot_curves(train_losses, train_accuracies, val_losses, val_accuracies, train_num, save_dir=save_dir)


def validate_model(model, criterion, val_loader, device='cuda'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def test_model(model, test_loader, device='cuda', save_dir='./results'):
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device))
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {test_acc:.4f}')


def plot_curves(train_losses, train_accuracies, val_losses, val_accuracies, train_num, save_dir=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f'training_plot_{timestamp}.png')
    else:
        file_name = f'training_plot_{timestamp}.png'

    plt.savefig(file_name)
    print(f"Plot saved to {file_name}")
    plt.close()