import os
import numpy as np
from PIL import Image

POST_ALGORITHMS = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

# GUI label -> actual extension (None = keep the original one)
EXTENSION_CHOICES = {
    "keep original": None,
    ".png": ".png",
    ".jpg": ".jpg",
}


def is_supported_image(img_path):
    """True if Pillow can open the file as an image."""
    if not os.path.isfile(img_path):
        return False
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def post_process(image_np, algorithm, factor): #will still process when factor = 1
    """Resize the AI-upscaled image by `factor` using the chosen algorithm.
    Returns the image unchanged if algorithm is off/None"""
    if not algorithm or algorithm == "off":
        return image_np

    resample = POST_ALGORITHMS[algorithm]
    img = Image.fromarray(image_np)
    new_size = (img.width * factor, img.height * factor)
    return np.array(img.resize(new_size, resample))


def build_output_path(input_file, output_dir, prefix, extension):
    """prefix + original name + chosen extension, inside output_dir.
    `extension` is None to keep the original extension."""
    stem, orig_ext = os.path.splitext(os.path.basename(input_file))
    ext = extension if extension else orig_ext
    return os.path.join(output_dir, f"{prefix}{stem}{ext}")


def save_np_image(image_np, save_path):
    img = Image.fromarray(image_np)
    # JPEG has no alpha channel
    if save_path.lower().endswith((".jpg", ".jpeg")) and img.mode != "RGB":
        img = img.convert("RGB")
    img.save(save_path)