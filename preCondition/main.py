import os
from preprocess import process_images  # 确保这是正确的预处理模块导入
from optimize_wavelet import optimize_wavelet_for_dataset
from wavelet_transform import process_images_with_optimized_wavelets

def main():
    """
    主函数，用于执行图像处理和特征提取的整个流程。
    """
    # 原始图像路径
    input_dir = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\img"
    # 灰度图像输出路径
    grayscale_dir = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\grayimg"
    # 输出处理后的图像的目录
    processed_output_dir = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\threshold_processed_images"
    # 输出子波段的目录
    subbands_output_dir = r"D:\workspace\pyspace\code\wheat-disease-wavelet-dl\wheatdata\subbands"

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


if __name__ == "__main__":
    main()