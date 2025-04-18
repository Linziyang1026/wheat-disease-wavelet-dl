class Config:
    # 训练参数
    num_epochs = 200
    batch_size = 64  # 增大批次大小
    learning_rate = 1e-4  # 提高学习率
    weight_decay = 1e-2  # 减小权重衰减系数
    patience = 10  # 减少早停的耐心周期
    delta = 0.0005

    # 数据集路径
    train_img_dir = r"F:\wheat\output_threshold_RGB_processed_images\train\images"
    train_label_dir = r"F:\wheat\output_threshold_RGB_processed_images\train\labels"
    val_img_dir = r"F:\wheat\output_threshold_RGB_processed_images\val\images"
    val_label_dir = r"F:\wheat\output_threshold_RGB_processed_images\val\labels"
    test_img_dir = r"F:\wheat\output_threshold_RGB_processed_images\test\images"
    test_label_dir = r"F:\wheat\output_threshold_RGB_processed_images\test\labels"

    # 模型参数
    num_classes = 2

    # 结果保存路径
    results_dir = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\Resnetmodel\results"