import os
import torch
from torch import optim
import sys

# 引入数据加载器、模型定义和训练验证测试函数
from data_loader import get_data_loaders  # 假设这是你的数据加载器导入路径
from model import WheatDiseaseClassifier  # 导入你的模型定义
from train_val_test import train_model, test_model  # 导入你的训练和测试函数

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义路径
train_img_dir = r"F:\wheat\output_threshold_processed_images\train\images"
train_label_dir = r"F:\wheat\output_threshold_processed_images\train\labels"
val_img_dir = r"F:\wheat\output_threshold_processed_images\val\images"
val_label_dir = r"F:\wheat\output_threshold_processed_images\val\labels"
test_img_dir = r"F:\wheat\output_threshold_processed_images\test\images"
test_label_dir = r"F:\wheat\output_threshold_processed_images\test\labels"


def main(train_num):
    # 加载数据
    try:
        train_loader, val_loader, test_loader = get_data_loaders(train_img_dir, train_label_dir, val_img_dir,
                                                                 val_label_dir,
                                                                 test_img_dir, test_label_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 定义模型
    model = WheatDiseaseClassifier(num_classes=2).to(device)

    # 设置损失函数和优化器，加入L2正则化
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型（包含验证）
    try:
        train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=200, device=device,
                    train_num=train_num)
    except Exception as e:
        print(f"Error during training: {e}")

    # 测试模型
    try:
        test_model(model, test_loader, device=device)
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    # 确保传入一个有效的 train_num 参数值
    if len(sys.argv) > 1:
        train_num = int(sys.argv[1])
    else:
        train_num = 5  # 默认值
    print(f"Using training number: {train_num}")

    main(train_num)