import os
import numpy as np
from PIL import Image
import pywt


def apply_wavelet_transform(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs


def threshold_coeffs(coeffs, thresholds=(0.1,)):
    coeffs_thresholded = [coeffs[0]] + [tuple(pywt.threshold(j, thresholds[0]) for j in c) for c in coeffs[1:]]
    return coeffs_thresholded


def reconstruct_image(coeffs, wavelet='haar'):
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    # 如果重构图像的尺寸与原始LL系数不匹配，则裁剪重构图像
    if reconstructed_image.shape != coeffs[0].shape:
        reconstructed_image = reconstructed_image[:coeffs[0].shape[0], :coeffs[0].shape[1]]
    return reconstructed_image


def save_subbands(coeffs, output_dir, image_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(os.path.join(output_dir, f"{image_filename}_approx.npy"), coeffs[0])
    for i, detail_coeff in enumerate(coeffs[1:], start=1):
        horizontal, vertical, diagonal = detail_coeff
        np.save(os.path.join(output_dir, f"{image_filename}_h{i}.npy"), horizontal)
        np.save(os.path.join(output_dir, f"{image_filename}_v{i}.npy"), vertical)
        np.save(os.path.join(output_dir, f"{image_filename}_d{i}.npy"), diagonal)


def process_images_with_optimized_wavelets(image_paths, grayscale_dir, processed_output_dir, subbands_output_dir,
                                           best_wavelet_results):
    """
    根据最佳小波基及其尺度处理图像，并保存结果。
    """
    for sample_image_path in image_paths:
        filename = os.path.basename(sample_image_path)  # 获取文件名

        # 确保只处理.jpg或.png文件
        if filename.endswith('.jpg') or filename.endswith('.png'):
            sample_image = np.array(Image.open(sample_image_path).convert('L'), dtype=np.float64) / 255.0

            # 获取该图像对应的最佳小波基和尺度
            best_wavelet_info = best_wavelet_results.get(filename, list(best_wavelet_results.values())[0])
            wavelet = best_wavelet_info['best_wavelet']
            level = best_wavelet_info['best_level']

            # 应用小波变换
            coeffs = apply_wavelet_transform(sample_image, wavelet=wavelet, level=level)

            # 保存子带系数
            save_subbands(coeffs, subbands_output_dir, os.path.splitext(filename)[0])

            # 阈值处理
            coeffs_thresholded = threshold_coeffs(coeffs, thresholds=(0.1,))

            # 重构图像
            reconstructed_image = reconstruct_image(coeffs_thresholded, wavelet=wavelet)

            # 构造正确的输出路径
            output_filename = 'processed_' + filename
            output_path = os.path.join(processed_output_dir, output_filename)

            # 保存或进一步处理重构后的图像
            Image.fromarray((reconstructed_image * 255).astype(np.uint8)).save(output_path)
            #print(f"Processed {filename} using {wavelet} at level {level} and saved to {output_path}")