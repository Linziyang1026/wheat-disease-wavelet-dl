import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_model
from data_loader import WheatDiseaseDataset, get_data_loaders

def train_model(model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy}%')

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据加载
train_loader, val_loader, test_loader = get_data_loaders(
    train_img_dir=r"F:\wheat\output_threshold_processed_images\train\images",
    train_label_dir=r"F:\wheat\output_threshold_processed_images\train\labels",
    val_img_dir=r"F:\wheat\output_threshold_processed_images\val\images",
    val_label_dir=r"F:\wheat\output_threshold_processed_images\val\labels",
    test_img_dir=r"F:\wheat\output_threshold_processed_images\test\images",
    test_label_dir=r"F:\wheat\output_threshold_processed_images\test\labels",
    batch_size=32
)

# 模型初始化
num_classes = 2  # 根据您的数据集类别数进行调整
model = get_model(num_classes, device)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 模型训练和评估
train_model(model, criterion, optimizer, num_epochs=10)
validate_model(model, val_loader)
test_model(model, test_loader)