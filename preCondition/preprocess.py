from PIL import Image
import os


def process_images(input_dir, output_dir):
    """
        处理目录下的所有彩色图像并转换为灰度图并输出到要求的路径中。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):

        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):

            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('L')
            output_path = os.path.join(output_dir, filename)
            img.save(output_path)
