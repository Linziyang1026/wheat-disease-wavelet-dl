import os
import torch
from torch import optim
from datetime import datetime
import sys

from data_loader import get_data_loaders
from model import WheatDiseaseClassifier
from train_val_test import train_model, test_model

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
# 定义路径(原始二值图像）
train_img_dir = "F://wheat//output_threshold_RGB_processed_images//train//images"
train_label_dir = r"F://wheat//output_threshold_RGB_processed_images//train//labels"
val_img_dir = "F://wheat//output_threshold_RGB_processed_images//val//images"
val_label_dir = r"F://wheat//output_threshold_RGB_processed_images//val//labels"
test_img_dir = "F://wheat//output_threshold_RGB_processed_images//test//images"
test_label_dir = r"F://wheat//output_threshold_RGB_processed_images//val//labels"
'''

# 定义路径(小波降噪后图像）
train_img_dir = "F://wheat//output_threshold_RGB_processed_images//train//images"
train_label_dir = "F://wheat//output_threshold_RGB_processed_images//train//labels"
val_img_dir = "F://wheat//output_threshold_RGB_processed_images//val//images"
val_label_dir = "F://wheat//output_threshold_RGB_processed_images//val//labels"
test_img_dir = "F://wheat//output_threshold_RGB_processed_images//test//images"
test_label_dir = "F://wheat//output_threshold_RGB_processed_images//test//labels"


def main(train_num):
    # 创建结果存储目录
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\oricnn\res",
                            f"{timestamp}_train{train_num}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
                train_num=train_num, save_dir=save_dir)

    # 测试模型
    test_model(model, test_loader, device=device)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_num = int(sys.argv[1])
    else:
        print("Using default training number as no argument provided.")
        train_num = 5  # 默认值

    main(train_num)