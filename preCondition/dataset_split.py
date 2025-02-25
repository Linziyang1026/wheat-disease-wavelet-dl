import os
import random
import shutil

def split_dataset(dataset_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    划分数据集为训练集、验证集和测试集。
    """
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(files)

    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    base_dir = os.path.dirname(dataset_dir)
    for set_name, file_list in splits.items():
        set_dir = os.path.join(base_dir, set_name)
        if not os.path.exists(set_dir):
            os.makedirs(set_dir)
        for file in file_list:
            shutil.copy(os.path.join(dataset_dir, file), os.path.join(set_dir, file))

    return splits