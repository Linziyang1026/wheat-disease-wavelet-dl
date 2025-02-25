import pywt
import numpy as np

def apply_wavelet_transform(image, wavelet='db1', level=1):
    """
    对输入的灰度图像进行离散小波变换。
    """
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def threshold_coeffs(coeffs, thresholds=(0.1, 0.1)):
    """
    对小波系数应用阈值处理。
    """
    new_coeffs = [coeffs[0]]  # LL 子带不进行阈值处理
    for c in coeffs[1:]:
        thr = thresholds[0] if len(new_coeffs) == 1 else thresholds[1]
        new_coeffs.append(tuple(pywt.threshold(j, thr) for j in c))
    return new_coeffs

def reconstruct_image(coeffs, wavelet='db1'):
    """
    根据处理后的小波系数重构图像。
    """
    reconstructed_image = pywt.waverec2(coeffs, wavelet)
    return reconstructed_image