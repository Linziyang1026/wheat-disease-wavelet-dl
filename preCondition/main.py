import os
import numpy as np
from PIL import Image

from feature_extraction import find_representative_samples
import preprocess  # 图像灰度化处理
import optimize_wavelet  # 寻找最佳小波基和尺度
import wavelet_transform  # 小波变换与阈值处理
import dataset_split  # 数据集划分


def main():
    # 配置路径
    image_dir = "D:/workspace/data/wheat/jpg"  # 原始彩色图像目录
    grayscale_dir = "D:/workspace/data/wheat/grayjpg"  # 灰度图像输出目录

    print("Step 1: Processing images to grayscale...")
    # 数据预处理：将彩色图像转换为灰度图像
    preprocess.process_images(image_dir, grayscale_dir)

    print("\nStep 2: Finding representative samples using clustering...")
    # 使用聚类算法找到代表性样本
    num_clusters = 5  # 聚类数量可以根据实际情况调整
    representative_samples = find_representative_samples(grayscale_dir, num_clusters=num_clusters)

    print("\nStep 3: Finding the best wavelet and level for the dataset...")
    # 在代表性样本中寻找最优的小波基和尺度
    best_wavelet, best_level = optimize_wavelet.find_best_wavelet_for_dataset(representative_samples)

    print(f"\nBest Wavelet for the dataset: {best_wavelet}, Best Level: {best_level}")

    print("\nStep 4: Applying wavelet transform and thresholding to the entire dataset...")
    # 使用找到的最佳小波基和尺度对整个数据集进行处理
    for image_file in os.listdir(grayscale_dir):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            sample_image_path = os.path.join(grayscale_dir, image_file)
            sample_image = np.array(Image.open(sample_image_path))

            # 应用小波变换
            coeffs = wavelet_transform.apply_wavelet_transform(sample_image, wavelet=best_wavelet, level=best_level)

            # 阈值处理
            coeffs_thresholded = wavelet_transform.threshold_coeffs(coeffs, thresholds=(0.1, 0.1))

            # 重构图像
            reconstructed_image = wavelet_transform.reconstruct_image(coeffs_thresholded, wavelet=best_wavelet)

            # 保存或进一步处理重构后的图像
            output_path = os.path.join(grayscale_dir, 'processed_' + image_file)
            Image.fromarray(reconstructed_image).convert('L').save(output_path)

    print("\nStep 5: Splitting the dataset into training, validation, and test sets...")
    # 划分数据集
    dataset_splits = dataset_split.split_dataset(grayscale_dir)
    print("Dataset split into:", list(dataset_splits.keys()))


if __name__ == "__main__":
    main()