import os

from optimize_wavelet import optimize_wavelet_for_dataset
from wavelet_transform import process_images_with_optimized_wavelets

"""
def main():
    
    #主函数，用于执行图像处理和特征提取的整个流程。
   
    # 原始图像路径
    input_dir = r"F:\wheat\wheatData"
    # 灰度图像输出路径
    grayscale_dir = r"F:\wheat\wheatData\grayimg"
    # 输出处理后的图像的目录
    processed_output_dir = r"F:\wheat\wheatData\threshold_processed_images"
    # 输出子波段的目录
    subbands_output_dir = r"F:\wheat\wheatData\subbands"


    # 创建输出目录
    os.makedirs(grayscale_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)
    os.makedirs(subbands_output_dir, exist_ok=True)

    # 图像预处理：将彩色图像转换为灰度图像
    process_images(input_dir, grayscale_dir)

    # 获取所有灰度图像的路径
    all_gray_images = [os.path.join(grayscale_dir, f) for f in os.listdir(grayscale_dir) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG')]

    # 定义输出文件路径
    output_file_dir = r"D:\workspace\data\wheat\results.txt"

    # 使用所有灰度图像来优化小波变换参数
    best_wavelet_results, best_wavelet, best_level = optimize_wavelet_for_dataset(all_gray_images, output_file_dir)

    print(f"Best Wavelet for the entire dataset: {best_wavelet}, Best Level: {best_level}")

    # 使用找到的最佳小波基及其尺度对整个数据集进行处理
    process_images_with_optimized_wavelets(all_gray_images, grayscale_dir, processed_output_dir, subbands_output_dir,
                                           best_wavelet_results=best_wavelet_results)

"""


def main(input_dir=None, output_dir=None, subbands_dir=None):
    # 主函数，用于执行图像处理和特征提取的整个流程。

    # 设置默认目录路径
    if input_dir is None:
        input_dir = r"F:\wheat\wheatData"
    if output_dir is None:
        output_dir = r"F:\wheat\output_RGB\out_processed_images_RGB"
    if subbands_dir is None:
        subbands_dir = r"F:\wheat\output_RGB\out_subbands_RGB"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(subbands_dir, exist_ok=True)

    # 获取所有图像的路径
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                  f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    # 定义输出文件路径
    output_file_dir = r"F:\wheat\output_RGB\results.txt"

    # 使用所有图像来优化小波变换参数
    best_wavelet_results, best_wavelet, best_level = optimize_wavelet_for_dataset(all_images, output_file_dir)

    print(f"Best Wavelet for the entire dataset: {best_wavelet}, Best Level: {best_level}")

    # 使用找到的最佳小波基及其尺度对整个数据集进行处理
    process_images_with_optimized_wavelets(all_images, best_wavelet, best_level, output_dir, subbands_dir)




if __name__ == "__main__":
    main()