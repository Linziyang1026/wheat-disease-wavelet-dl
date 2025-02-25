import pywt
from PIL import Image  # 使用新版Pillow中的Resampling
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
import numpy as np

def adjust_image_size(sample_path):
    """
    调整图像大小以确保宽度和高度都是偶数。
    """
    img = Image.open(sample_path)
    width, height = img.size
    new_width = ((width + 1) // 2) * 2  # 确保宽度是偶数
    new_height = ((height + 1) // 2) * 2  # 确保高度是偶数
    if width != new_width or height != new_height:
        try:
            # 调整图像尺寸
            img = img.resize((new_width, new_height), resample=1)
        except Exception as e:
            print(f"Failed to process image {sample_path} due to {e}")
        img.save(sample_path)

def evaluate_wavelet(image, wavelet, level):
    """
    对给定的小波基和分解层数评估小波变换性能。
    """
    # 检查图像数据类型和值范围
    if image.dtype != np.float64:
        print(f"Warning: Image data type is {image.dtype}, expected np.float64")
    if not (0 <= image.min() and image.max() <= 1):
        print(f"Warning: Image value range is [{image.min():.4f}, {image.max():.4f}], expected [0, 1]")

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresholded = [coeffs[0]] + [tuple(pywt.threshold(j, 0.1) for j in c) for c in coeffs[1:]]
    reconstructed_image = pywt.waverec2(coeffs_thresholded, wavelet)

    # 如果形状不匹配，则裁剪或填充
    if reconstructed_image.shape != image.shape:
        print(f"Warning: Shape mismatch between original and reconstructed images: {image.shape} vs {reconstructed_image.shape}")
        if reconstructed_image.shape[0] > image.shape[0] or reconstructed_image.shape[1] > image.shape[1]:
            # 裁剪
            reconstructed_image = reconstructed_image[:image.shape[0], :image.shape[1]]
        else:
            # 填充（如果需要）
            pad_height = image.shape[0] - reconstructed_image.shape[0]
            pad_width = image.shape[1] - reconstructed_image.shape[1]
            reconstructed_image = np.pad(reconstructed_image,
                                         ((0, pad_height), (0, pad_width)),
                                         mode='constant', constant_values=0)

    error = mse(image, reconstructed_image)
    quality = psnr(image, reconstructed_image)

    return error, quality

def find_best_wavelet(image, wavelets=('haar', 'db1', 'db2', 'coif1', 'sym2'), max_level=3):
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

def optimize_wavelet_for_representatives(representative_samples):
    results = {}
    for sample_path in representative_samples:
        adjust_image_size(sample_path)

        img = Image.open(sample_path).convert('L')
        img_array = np.array(img, dtype=np.float64) / 255.0
        best_wavelet, best_level = find_best_wavelet(img_array)
        results[sample_path] = {'best_wavelet': best_wavelet, 'best_level': best_level}

        print(f"For image {sample_path}, Best Wavelet is {best_wavelet} with level {best_level}")

    return results