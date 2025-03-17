import os


def modify_labels(file_path):
    """
    修改给定文件中所有非HealthyWheat（健康小麦，假设其标签为1）标签为0。

    :param file_path: 文件路径
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            if len(parts) > 0 and parts[0] != '1':  # 如果不是HealthyWheat的标签
                parts[0] = '0'  # 将其改为0
            file.write(' '.join(parts) + '\n')


def traverse_and_modify(directory):
    """
    遍历目录下的所有txt文件，并调用modify_labels函数修改标签。

    :param directory: 目标文件夹路径
    """
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            modify_labels(os.path.join(directory, filename))


# 使用示例
directory_path = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\txt (withoutHealthyWheat)"  # 替换为您的文件夹路径
traverse_and_modify(directory_path)