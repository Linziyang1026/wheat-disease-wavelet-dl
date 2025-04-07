import os
import numpy as np
from PIL import Image
import pywt


def preprocess_image(image):
    """Preprocess the image to ensure its dimensions are suitable for wavelet transform."""
    original_shape = image.shape  # Save original size
    h, w = original_shape  # For single channel, shape is 2D
    new_h = h if h % 2 == 0 else h + 1
    new_w = w if w % 2 == 0 else w + 1
    if new_h != h or new_w != w:
        # Adjust padding for 2D images (single channel)
        image = np.pad(image, ((0, new_h - h), (0, new_w - w)), mode='symmetric')
    return image, original_shape  # Return processed image and original shape


def apply_wavelet_transform(image, wavelet='coif1', level=1):
    """
    Apply wavelet transform on each channel of the input image with specified wavelet and decomposition level.
    """
    processed_images = []
    original_shape = image.shape
    for i in range(3):
        channel_image = image[:, :, i]
        processed_channel, _ = preprocess_image(channel_image)
        coeffs = pywt.wavedec2(processed_channel, wavelet, level=level)
        processed_images.append(coeffs)
    return processed_images, original_shape

def reconstruct_image(coeffs_list, wavelet='coif1', original_shape=None):
    """
    Reconstruct each channel from wavelet coefficients ensuring the size matches the original.
    """
    reconstructed_channels = []
    for coeffs in coeffs_list:
        reconstructed_channel = pywt.waverec2(coeffs, wavelet)
        if original_shape is not None:
            target_height, target_width = original_shape[:2]
            if reconstructed_channel.shape[0] > target_height or reconstructed_channel.shape[1] > target_width:
                reconstructed_channel = reconstructed_channel[:target_height, :target_width]
            elif reconstructed_channel.shape[0] < target_height or reconstructed_channel.shape[1] < target_width:
                pad_height = target_height - reconstructed_channel.shape[0]
                pad_width = target_width - reconstructed_channel.shape[1]
                reconstructed_channel = np.pad(reconstructed_channel,
                                               ((0, pad_height), (0, pad_width)),
                                               mode='symmetric')
        reconstructed_channels.append(reconstructed_channel)
    return np.stack(reconstructed_channels, axis=-1)  # Stack channels to form a color image


def save_subbands(coeffs, output_dir, filename_prefix):
    """Save each subband of the wavelet coefficients to a separate file."""
    approx = coeffs[0]
    output_path = os.path.join(output_dir, f"{filename_prefix}_approx.png")
    Image.fromarray((np.abs(approx) * 255 / np.max(np.abs(approx))).astype(np.uint8)).save(output_path)

    for i, detail_level in enumerate(coeffs[1:], start=1):
        for orientation, detail in zip(['h', 'v', 'd'], detail_level):
            output_path = os.path.join(output_dir, f"{filename_prefix}_level{i}_{orientation}.png")
            Image.fromarray((np.abs(detail) * 255 / np.max(np.abs(detail))).astype(np.uint8)).save(output_path)


def process_and_save_image(input_path, output_dir, subbands_dir):
    """Process an image using optimal wavelet parameters and save the result."""
    image = np.array(Image.open(input_path).convert('RGB'), dtype=np.float64) / 255.0
    coeffs_list, original_shape = apply_wavelet_transform(image, wavelet='coif1', level=1)

    # Save subbands for the first channel as an example
    save_subbands(coeffs_list[0], subbands_dir, os.path.splitext(os.path.basename(input_path))[0])

    reconstructed_image = reconstruct_image(coeffs_list, wavelet='coif1', original_shape=original_shape)

    output_filename = 'processed_' + os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_filename)
    Image.fromarray((np.clip(reconstructed_image, 0, 1) * 255).astype(np.uint8)).save(output_path)


def main(input_dir, output_dir, subbands_dir):
    """Process all images in the input directory and save the results to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(subbands_dir):
        os.makedirs(subbands_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.JPG')):
            input_path = os.path.join(input_dir, filename)
            process_and_save_image(input_path, output_dir, subbands_dir)


if __name__ == "__main__":
    input_dir = r'F:\wheat\wheatData'
    output_dir = r'F:\wheat\output_RGB\out_processed_images_RGB'
    subbands_dir = r'F:\wheat\output_RGB\out_subbands_RGB'
    main(input_dir, output_dir, subbands_dir)