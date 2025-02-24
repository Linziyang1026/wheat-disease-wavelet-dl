import os
import shutil


def organize_files_by_extension(directory):
    # 遍历给定目录中的所有文件和文件夹
    for filename in os.listdir(directory):
        # 获取文件的完整路径
        file_path = os.path.join(directory, filename)

        # 如果是文件而不是文件夹
        if os.path.isfile(file_path):
            # 获取文件扩展名（不区分大小写）
            file_extension = os.path.splitext(filename)[1][1:].lower()

            # 根据文件扩展名创建目标文件夹的路径
            destination_dir = os.path.join(directory, file_extension)

            # 如果目标文件夹不存在，则创建它
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)

            # 将文件移动到对应类型的文件夹中
            shutil.move(file_path, destination_dir)


# 使用示例
directory_to_organize = "D:\workspace\data\zzy_wheat"  # 替换为你要整理的文件夹路径
organize_files_by_extension(directory_to_organize)