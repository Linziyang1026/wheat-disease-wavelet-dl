import pywt
from PIL import Image  # 使用新版Pillow中的Resampling
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
import numpy as np
from collections import Counter

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
    original_shape = image.shape  # 记录原始尺寸
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    coeffs_thresholded = [coeffs[0]] + [tuple(pywt.threshold(j, 0.1) for j in c) for c in coeffs[1:]]
    reconstructed_image = pywt.waverec2(coeffs_thresholded, wavelet)

    # 确保重构后的图像与原始图像尺寸相同
    if reconstructed_image.shape[0] > original_shape[0] or reconstructed_image.shape[1] > original_shape[1]:
        # 裁剪以匹配原始尺寸
        reconstructed_image = reconstructed_image[:original_shape[0], :original_shape[1]]
    else:
        # 填充以匹配原始尺寸
        pad_height = original_shape[0] - reconstructed_image.shape[0]
        pad_width = original_shape[1] - reconstructed_image.shape[1]
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


def optimize_wavelet_for_dataset(image_paths, output_file):
    results = {}
    wavelet_counter = {}
    level_counter = {}

    with open(output_file, 'w') as f:
        for sample_path in image_paths:
            try:
                # 调整图像大小以确保宽度和高度都是偶数
                img = Image.open(sample_path).convert('L')
                img_array = np.array(img, dtype=np.float64) / 255.0

                # 找到当前图像的最佳小波基和分解尺度
                best_wavelet, best_level = find_best_wavelet(img_array)

                # 记录并打印每个图像的最佳配置
                result_line = f"For image {sample_path}, Best Wavelet is {best_wavelet} with level {best_level}\n"
                print(result_line.strip())
                f.write(result_line)

                # 统计每个小波基和分解尺度的使用次数
                if best_wavelet not in wavelet_counter:
                    wavelet_counter[best_wavelet] = 0
                wavelet_counter[best_wavelet] += 1

                if best_level not in level_counter:
                    level_counter[best_level] = 0
                level_counter[best_level] += 1

                results[sample_path] = {'best_wavelet': best_wavelet, 'best_level': best_level}

            except Exception as e:
                error_line = f"Failed to process image {sample_path} due to {e}\n"
                print(error_line.strip())
                f.write(error_line)

        # 确定整个数据集的最佳小波基和分解尺度
        most_common_wavelet = Counter(wavelet_counter).most_common(1)[0]
        most_common_level = Counter(level_counter).most_common(1)[0]

        overall_result_line = f"\nOverall Best Wavelet: {most_common_wavelet[0]}, Level: {most_common_level[0]}\n"
        print(overall_result_line.strip())
        f.write(overall_result_line)

    return results, most_common_wavelet[0], most_common_level[0]