import os
import random
import shutil


def match_images_with_labels(image_folder, label_folder):
    """
    匹配原始图像文件和对应的标签文件。

    :param image_folder: 原始图像所在的文件夹路径
    :param label_folder: 标签文件所在的文件夹路径
    :return: 返回一个字典，键为标签文件名（不含扩展名），值为包含标签路径和图像路径的字典
    """
    # 获取标签文件夹中所有的.txt文件名（不包含扩展名）
    labels = {os.path.splitext(f)[0]: os.path.join(label_folder, f) for f in os.listdir(label_folder) if
              f.endswith('.txt')}

    matched_files = {}

    # 遍历图像文件夹中的所有文件
    for image_file in os.listdir(image_folder):
        file_name_without_ext = os.path.splitext(image_file)[0]

        # 提取原图像名称部分（去除 'processed_' 前缀）
        original_image_name = file_name_without_ext.replace('processed_', '', 1)

        # 检查是否存在对应的标签文件
        if original_image_name in labels:
            matched_files[original_image_name] = {
                'label': labels[original_image_name],
                'image': os.path.join(image_folder, image_file)
            }

    return matched_files


def split_dataset(matched_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    划分数据集为训练集、验证集和测试集。

    :param matched_files: 匹配好的文件字典
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :return: 返回三个列表，分别为训练集、验证集和测试集中的文件信息
    """
    file_ids = list(matched_files.keys())
    random.shuffle(file_ids)

    total = len(file_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = [matched_files[id] for id in file_ids[:train_end]]
    val_set = [matched_files[id] for id in file_ids[train_end:val_end]]
    test_set = [matched_files[id] for id in file_ids[val_end:]]

    return train_set, val_set, test_set


def copy_files_to_datasets(dataset, destination_folder):
    """
    将数据集中的文件复制到指定的目标文件夹。

    :param dataset: 数据集中的文件信息列表
    :param destination_folder: 目标文件夹路径
    """
    for item in dataset:
        label_path = item['label']
        label_name = os.path.basename(label_path)
        new_label_path = os.path.join(destination_folder, 'labels', label_name)
        os.makedirs(os.path.dirname(new_label_path), exist_ok=True)
        shutil.copy(label_path, new_label_path)

        image_path = item['image']
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(destination_folder, 'images', image_name)
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        shutil.copy(image_path, new_image_path)


def main():
    image_folder = r"F:\wheat\output_RGB\out_processed_images_RGB"  # 替换为你的原始图像文件夹的实际路径
    label_folder = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\txt_01"  # 替换为你的标签文件夹的实际路径

    matched_files = match_images_with_labels(image_folder, label_folder)
    train_set, val_set, test_set = split_dataset(matched_files)

    # 定义目标文件夹
    output_base_dir = r"F:\wheat\output_threshold_RGB_processed_images"  # 替换为你希望保存划分后数据集的基础路径
    os.makedirs(output_base_dir, exist_ok=True)

    # 复制文件到相应的数据集文件夹
    copy_files_to_datasets(train_set, os.path.join(output_base_dir, 'train'))
    copy_files_to_datasets(val_set, os.path.join(output_base_dir, 'val'))
    copy_files_to_datasets(test_set, os.path.join(output_base_dir, 'test'))


if __name__ == "__main__":
    main()