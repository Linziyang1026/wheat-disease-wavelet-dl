import os
import numpy as np
from PIL import Image
import pywt


def preprocess_image(image):
    """Preprocess the image to ensure its dimensions are suitable for wavelet transform."""
    original_shape = image.shape  # 保存原始尺寸
    h, w = original_shape
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    if new_h != h or new_w != w:
        image = np.pad(image, ((0, new_h - h), (0, new_w - w)), mode='symmetric')
    return image, original_shape  # 返回处理后的图像和原始尺寸


def apply_wavelet_transform(image, wavelet='haar', level=1):
    """
    Apply wavelet transform on the input image with specified wavelet and decomposition level.
    """
    processed_image, original_shape = preprocess_image(image)
    coeffs = pywt.wavedec2(processed_image, wavelet, level=level)
    return coeffs, original_shape  # 返回小波系数和原始尺寸


def threshold_coeffs(coeffs, thresholds=(0.1,)):
    """
    Apply thresholding to the wavelet coefficients.
    """
    coeffs_thresholded = [coeffs[0]] + [tuple(pywt.threshold(j, thresholds[0]) for j in c) for c in coeffs[1:]]
    return coeffs_thresholded


def reconstruct_image(coeffs, wavelet='haar', original_shape=None):
    """
    Reconstruct the image from wavelet coefficients ensuring the size matches the original.
    """
    reconstructed_image = pywt.waverec2(coeffs, wavelet)

    if original_shape is not None:
        target_height, target_width = original_shape
        # 如果重构后的图像大于原始尺寸，则裁剪
        if reconstructed_image.shape[0] > target_height or reconstructed_image.shape[1] > target_width:
            reconstructed_image = reconstructed_image[:target_height, :target_width]
        # 如果小于原始尺寸，则填充
        elif reconstructed_image.shape[0] < target_height or reconstructed_image.shape[1] < target_width:
            pad_height = target_height - reconstructed_image.shape[0]
            pad_width = target_width - reconstructed_image.shape[1]
            reconstructed_image = np.pad(reconstructed_image,
                                         ((0, pad_height), (0, pad_width)),
                                         mode='symmetric')

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
    if not os.path.exists(processed_output_dir):
        os.makedirs(processed_output_dir)
    if not os.path.exists(subbands_output_dir):
        os.makedirs(subbands_output_dir)

    for sample_image_path in image_paths:
        filename = os.path.basename(sample_image_path)

        if filename.endswith('.jpg') or filename.endswith('.png'):
            sample_image = np.array(Image.open(os.path.join(grayscale_dir, filename)).convert('L'),
                                    dtype=np.float64) / 255.0

            best_wavelet_info = best_wavelet_results.get(filename, list(best_wavelet_results.values())[0])
            wavelet = best_wavelet_info['best_wavelet']
            level = best_wavelet_info['best_level']

            coeffs, original_shape = apply_wavelet_transform(sample_image, wavelet=wavelet, level=level)
            save_subbands(coeffs, subbands_output_dir, os.path.splitext(filename)[0])

            coeffs_thresholded = threshold_coeffs(coeffs, thresholds=(0.1,))
            reconstructed_image = reconstruct_image(coeffs_thresholded, wavelet=wavelet, original_shape=original_shape)

            output_filename = 'processed_' + filename
            output_path = os.path.join(processed_output_dir, output_filename)
            Image.fromarray((np.clip(reconstructed_image, 0, 1) * 255).astype(np.uint8)).save(output_path)