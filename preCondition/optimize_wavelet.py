import pywt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
import numpy as np
import feature_extraction

def evaluate_wavelet(image, wavelet, level):
    """
    对给定的小波基和分解层次进行评估。
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    # 进行阈值处理（这里我们简单地保留系数）
    coeffs_thresholded = [coeffs[0]] + [tuple(pywt.threshold(j, 0.1) for j in c) for c in coeffs[1:]]
    reconstructed_image = pywt.waverec2(coeffs_thresholded, wavelet)

    # 确保重建图像的尺寸与原图相同
    if reconstructed_image.shape != image.shape:
        reconstructed_image = reconstructed_image[:image.shape[0], :image.shape[1]]

    error = mse(image, reconstructed_image)
    quality = psnr(image, reconstructed_image)

    return error, quality


def find_best_wavelet(image, wavelets=('haar', 'db1', 'db2', 'coif1', 'sym2'), max_level=3):
    """
    寻找最优的小波基和分解层次。
    """
    best_wavelet = None
    best_level = 0
    best_quality = -float('inf')

    for wavelet in wavelets:
        for level in range(1, max_level + 1):
            try:
                error, quality = evaluate_wavelet(image, wavelet, level)
                print(f"Wavelet: {wavelet}, Level: {level}, MSE: {error:.4f}, PSNR: {quality:.4f}")
                if quality > best_quality:
                    best_quality = quality
                    best_wavelet = wavelet
                    best_level = level
            except Exception as e:
                print(f"Failed to process Wavelet: {wavelet}, Level: {level} due to {e}")

    print(f"\nBest Wavelet: {best_wavelet}, Best Level: {best_level}, Best PSNR: {best_quality:.4f}")
    return best_wavelet, best_level


def optimize_wavelet_for_representatives(grayscale_dir, num_clusters=5):
    """
    在代表性样本上优化小波变换参数。
    """
    extractor = FeatureExtractor()
    representative_samples = find_representative_samples(grayscale_dir, num_clusters=num_clusters)

    results = {}
    for sample_path in representative_samples:
        img = Image.open(sample_path).convert('L')  # 转为灰度图像
        img_array = np.array(img, dtype=np.float64) / 255.0  # 归一化
        best_wavelet, best_level = find_best_wavelet(img_array)
        results[sample_path] = {'best_wavelet': best_wavelet, 'best_level': best_level}

    return results