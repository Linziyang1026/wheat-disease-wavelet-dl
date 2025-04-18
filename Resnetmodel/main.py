import os
import datetime
import sys
import torch
import torch.optim as optim
from train_val_test import train_model, test_model, EarlyStopping
from model import get_model
from data_loader import get_data_loaders
from config import Config


def main(train_num):
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建结果存储目录
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(Config.results_dir, f"{timestamp}_train{train_num}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据加载
    train_loader, val_loader, test_loader = get_data_loaders()

    # 模型初始化
    model = get_model(device=device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs, eta_min=1e-6)

    # 初始化早停
    early_stopping = EarlyStopping()

    # 模型训练和评估
    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir
    )

    # 测试模型
    test_model(model=model, test_loader=test_loader, device=device, save_dir=save_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_num = int(sys.argv[1])
    else:
        print("Using default training number as no argument provided.")
        train_num = 5  # 默认值

    main(train_num)