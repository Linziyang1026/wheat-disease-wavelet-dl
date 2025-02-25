from PIL import Image
import os

def rgb_to_grayscale(image_path, output_dir):
    """
        将原始图像转为灰度图像
    """
    img = Image.open(image_path).convert('L')  # 转换为灰度图
    img_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, img_name)
    img.save(output_path)
    return output_path


def process_images(input_dir, output_dir):
    """
        处理目录下的所有图像并转换为灰度图。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(input_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            rgb_to_grayscale(os.path.join(input_dir, image_file), output_dir)