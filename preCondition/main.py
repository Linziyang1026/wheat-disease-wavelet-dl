import os
from preprocess import process_images  # 确保这是正确的预处理模块导入
from feature_extraction import FeatureExtractor, find_representative_samples
from optimize_wavelet import optimize_wavelet_for_representatives
from wavelet_transform import process_images_with_optimized_wavelets

def main():
    """
    主函数，用于执行图像处理和特征提取的整个流程。
    """
    # 原始图像路径
    input_dir = r"D:\workspace\data\wheat\img"
    # 灰度图像输出路径
    grayscale_dir = r"D:\workspace\data\wheat\grayimg"
    # 输出处理后的图像的目录
    processed_output_dir = r"D:\workspace\data\wheat\threshold_processed_images"
    # 输出子波段的目录
    subbands_output_dir = r"D:\workspace\data\wheat\subbands"

    # 创建输出目录
    os.makedirs(grayscale_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)
    os.makedirs(subbands_output_dir, exist_ok=True)

    # 图像预处理：将彩色图像转换为灰度图像
    process_images(input_dir, grayscale_dir)

    # 特征提取及确定最佳聚类数量
    extractor = FeatureExtractor()
    # 获取所有原始图像的路径
    all_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                  f.endswith('.jpg') or f.endswith('.png')]
    # 提取所有图像的特征
    features = [extractor.extract_features(img) for img in all_images]

    # 自动确定最佳聚类数量并找到代表性样本
    representative_samples = find_representative_samples(all_images, features)

    # 在代表性样本上优化小波变换参数
    best_wavelet_results = optimize_wavelet_for_representatives(representative_samples)

    # 使用找到的最佳小波基及其尺度对整个数据集进行处理
    process_images_with_optimized_wavelets(all_images, grayscale_dir, processed_output_dir, subbands_output_dir,
                                           best_wavelet_results)


if __name__ == "__main__":
    main()