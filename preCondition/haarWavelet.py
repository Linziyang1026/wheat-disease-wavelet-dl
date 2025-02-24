import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse


def evaluate_wavelet(image_path, wavelet='haar', level=1):
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)

    coeffs = pywt.wavedec2(img_array, wavelet, level=level)
    reconstructed_img = pywt.waverec2(coeffs, wavelet)

    # 计算PSNR和MSE
    original_psnr = psnr(img_array, img_array)
    reconstructed_psnr = psnr(img_array, reconstructed_img[:img_array.shape[0], :img_array.shape[1]])
    original_mse = mse(img_array, img_array)
    reconstructed_mse = mse(img_array, reconstructed_img[:img_array.shape[0], :img_array.shape[1]])

    return {'wavelet': wavelet, 'level': level, 'psnr': reconstructed_psnr, 'mse': reconstructed_mse}


# 示例调用
wavelets = ['haar', 'db1', 'coif1', 'sym2']
results = []

for wavelet in wavelets:
    for level in range(1, 4):  # 多尺度分解
        result = evaluate_wavelet(os.path.join(grayscale_dir, os.listdir(grayscale_dir)[0]), wavelet, level)
        results.append(result)

# 输出结果
for res in results:
    print(res)