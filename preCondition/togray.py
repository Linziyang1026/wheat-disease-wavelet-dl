from PIL import Image
import os
import numpy as np

def rgb_to_grayscale(image_path, output_dir):
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    img_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, img_name)
    img.save(output_path)
    return output_path

# 示例调用
image_dir = "D:\workspace\data\wheat\jpg"
grayscale_dir = "D:\workspace\data\wheat\grayjpg"
if not os.path.exists(grayscale_dir):
    os.makedirs(grayscale_dir)

for image_file in os.listdir(image_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        rgb_to_grayscale(os.path.join(image_dir, image_file), grayscale_dir)