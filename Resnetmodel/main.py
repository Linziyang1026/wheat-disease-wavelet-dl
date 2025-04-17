import os
import datetime
import sys
import torch
import torch.optim as optim
from train_val_test import train_model, validate_model, test_model
from model import get_model
from data_loader import get_data_loaders

def main(train_num):
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建结果存储目录
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\Resnetmodel\results",
                            f"{timestamp}_train{train_num}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据加载
    train_loader, val_loader, test_loader = get_data_loaders(
        train_img_dir=r"F:\wheat\output_threshold_RGB_processed_images\train\images",
        train_label_dir=r"F:\wheat\output_threshold_RGB_processed_images\train\labels",
        val_img_dir=r"F:\wheat\output_threshold_RGB_processed_images\val\images",
        val_label_dir=r"F:\wheat\output_threshold_RGB_processed_images\val\labels",
        test_img_dir=r"F:\wheat\output_threshold_RGB_processed_images\test\images",
        test_label_dir=r"F:\wheat\output_threshold_RGB_processed_images\test\labels",
        batch_size=32
    )

    # 模型初始化
    num_classes = 2  # 根据您的数据集类别数进行调整
    model = get_model(num_classes=num_classes, device=device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # 模型训练和评估
    train_model(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                train_loader=train_loader, val_loader=val_loader, num_epochs=200,
                device=device, train_num=train_num, save_dir=save_dir)

    # 使用相同的save_dir测试模型
    test_model(model=model, test_loader=test_loader, device=device, save_dir=save_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_num = int(sys.argv[1])
    else:
        print("Using default training number as no argument provided.")
        train_num = 5  # 默认值

    main(train_num)