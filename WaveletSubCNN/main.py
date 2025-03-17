import os
import torch
from torch import optim
import sys

from data_loader import get_data_loaders
from model import WheatDiseaseClassifier
from train_val_test import train_model, test_model

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义路径
train_img_dir = r"F:\wheat\output\train\images"
train_label_dir = r"F:\wheat\output\train\labels"
val_img_dir = r"F:\wheat\output\val\images"
val_label_dir = r"F:\wheat\output\val\labels"
test_img_dir = r"F:\wheat\output\test\images"
test_label_dir = r"F:\wheat\output\test\labels"


def main(train_num):
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(train_img_dir, train_label_dir, val_img_dir, val_label_dir,
                                                             test_img_dir, test_label_dir)

    # 定义模型
    model = WheatDiseaseClassifier(num_classes=2).to(device)

    # 设置损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型（包含验证）
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=200, device=device,
                train_num=train_num)

    # 测试模型
    test_model(model, test_loader, device=device)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_num = int(sys.argv[1])
    else:
        print("Using default training number as no argument provided.")
        train_num = 5  # 默认值

    main(train_num)