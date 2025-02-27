import os
import numpy as np
from PIL import Image
import pywt


def apply_wavelet_transform(image, wavelet='haar', level=1):
    """
    Apply wavelet transform on the input image with specified wavelet and decomposition level.
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs


def threshold_coeffs(coeffs, thresholds=(0.1,)):
    """
    Apply thresholding to the wavelet coefficients.
    """
    coeffs_thresholded = [coeffs[0]] + [tuple(pywt.threshold(j, thresholds[0]) for j in c) for c in coeffs[1:]]
    return coeffs_thresholded


def reconstruct_image(coeffs, wavelet='haar'):
    """
    Reconstruct the image from wavelet coefficients.
    """
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    original_shape = coeffs[0].shape

    if reconstructed_image.shape[0] > original_shape[0] or reconstructed_image.shape[1] > original_shape[1]:
        reconstructed_image = reconstructed_image[:original_shape[0], :original_shape[1]]
    else:
        pad_height = original_shape[0] - reconstructed_image.shape[0]
        pad_width = original_shape[1] - reconstructed_image.shape[1]
        reconstructed_image = np.pad(reconstructed_image,
                                     ((0, pad_height), (0, pad_width)),
                                     mode='constant', constant_values=0)
    return reconstructed_image


def save_subbands(coeffs, output_dir, filename_prefix):
    """Save each subband of the wavelet coefficients to a separate file."""
    approx = coeffs[0]
    output_path = os.path.join(output_dir, f"{filename_prefix}_approx.png")
    Image.fromarray((np.abs(approx) * 255 / np.max(np.abs(approx))).astype(np.uint8)).save(output_path)

    for i, detail_level in enumerate(coeffs[1:], start=1):
        for orientation, detail in zip(['h', 'v', 'd'], detail_level):
            output_path = os.path.join(output_dir, f"{filename_prefix}_level{i}_{orientation}.png")
            Image.fromarray((np.abs(detail) * 255 / np.max(np.abs(detail))).astype(np.uint8)).save(output_path)


def process_images_with_optimized_wavelets(image_paths, grayscale_dir, processed_output_dir, subbands_output_dir,
                                           best_wavelet_results):
    """
    Process images using optimized wavelet parameters obtained from main.py.
    """
    # 确保输出目录存在
    if not os.path.exists(processed_output_dir):
        os.makedirs(processed_output_dir)
    if not os.path.exists(subbands_output_dir):
        os.makedirs(subbands_output_dir)

    for sample_image_path in image_paths:
        filename = os.path.basename(sample_image_path)

        if filename.endswith('.jpg') or filename.endswith('.png'):
            sample_image = np.array(Image.open(sample_image_path).convert('L'), dtype=np.float64) / 255.0

            # 获取针对当前图片的最佳小波基及分解层次
            best_wavelet_info = best_wavelet_results.get(filename, list(best_wavelet_results.values())[0])
            wavelet = best_wavelet_info['best_wavelet']
            level = best_wavelet_info['best_level']

            # 应用小波变换
            coeffs = apply_wavelet_transform(sample_image, wavelet=wavelet, level=level)

            # 保存子带图像
            save_subbands(coeffs, subbands_output_dir, os.path.splitext(filename)[0])

            # 阈值处理
            coeffs_thresholded = threshold_coeffs(coeffs, thresholds=(0.1,))

            # 图像重构
            reconstructed_image = reconstruct_image(coeffs_thresholded, wavelet=wavelet)

            # 保存处理后的图像
            output_filename = 'processed_' + filename
            output_path = os.path.join(processed_output_dir, output_filename)
            Image.fromarray((np.clip(reconstructed_image, 0, 1) * 255).astype(np.uint8)).save(output_path)