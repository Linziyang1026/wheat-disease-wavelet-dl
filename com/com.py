import os


def find_missing_processed_images(folder_a, folder_b):
    """
    查找在folder_a中有但在folder_b中没有对应处理版本（添加了前缀processed_）的图片。

    :param folder_a: 包含原始图片的文件夹路径。
    :param folder_b: 应包含处理后的图片的文件夹路径。
    """
    # 获取文件夹A中的所有.jpg文件名（不包括扩展名）
    images_a = {os.path.splitext(f)[0] for f in os.listdir(folder_a) if f.endswith('.jpg')}

    # 获取文件夹B中所有以processed_开头的.jpg文件，并去除processed_前缀得到原文件名
    processed_prefix = "processed_"
    processed_files_b = {f[len(processed_prefix):len(f) - 4] for f in os.listdir(folder_b) if
                         f.startswith(processed_prefix) and f.endswith('.jpg')}

    # 找出在文件夹A中有但在文件夹B中没有对应处理版本的图片
    missing_images = images_a - processed_files_b

    return missing_images


# 使用函数查找缺失的图片
folder_a_path = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\grayimg"  # 替换为你的文件夹A的实际路径
folder_b_path = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\threshold_processed_images"  # 替换为你的文件夹B的实际路径

missing_images = find_missing_processed_images(folder_a_path, folder_b_path)

print("Missing processed images:")
for img in missing_images:
    print(f"{img}.jpg")

