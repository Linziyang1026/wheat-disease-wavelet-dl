import torch
import torch.optim as optim

from train_val_test import train_model, validate_model, test_model
from model import get_model
from data_loader import get_data_loaders


def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_loader, val_loader, test_loader = get_data_loaders(
        train_img_dir=r"F:\wheat\output_threshold_processed_images\train\images",
        train_label_dir = r"F:\wheat\output_threshold_processed_images\train\labels",
        val_img_dir = r"F:\wheat\output_threshold_processed_images\val\images",
        val_label_dir = r"F:\wheat\output_threshold_processed_images\val\labels",
        test_img_dir = r"F:\wheat\output_threshold_processed_images\test\images",
        test_label_dir = r"F:\wheat\output_threshold_processed_images\test\labels",
        batch_size=32
    )

    # 模型初始化
    num_classes = 2  # 根据您的数据集类别数进行调整
    model = get_model(num_classes, device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 模型训练和评估
    train_model(model, criterion, optimizer, num_epochs=200)
    validate_model(model, val_loader)
    test_model(model, test_loader)

if __name__ == "__main__":
    main()