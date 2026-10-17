import torch
import numpy as np
from PIL import Image
import re
from upcunet_v3 import RealWaifuUpScaler
import gc
import os

from image_utils import (
    is_supported_image,
    post_process,
    build_output_path,
    save_np_image,
)

def create_cugan_upscaler(model_path, scale):
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    use_half = (device.type != "cpu")

    return RealWaifuUpScaler(scale=scale, weight_path=model_path, half=use_half, device=device)

def get_device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(0)
    else:
        return "CPU"

def inference(img_path, model_path, scale, tile_mode, cache_mode, alpha, upscaler): #inference
    image = Image.open(img_path)
    if not image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_np = np.array(image)

    try:
        output_np = upscaler(image_np, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)
    except torch.OutOfMemoryError as e: #out of vram, fall back to cpu
        del e
        gc.collect()
        torch.cuda.empty_cache()

        upscalerCPU = RealWaifuUpScaler(scale=scale, weight_path=model_path, half=False, device='cpu')
        output_np = upscalerCPU(image_np, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)

    return output_np

def cugan_start(input_path, model_path, scale, tile_mode, cache_mode, alpha, output_path,
                prefix="", extension=None, post_algorithm=None, post_factor=1,
                progress_callback=None):
    """
    Upscale a single image or every image in a folder and save the results.
 
    extension: ".png", ".jpg", or None to keep each file's original extension.
    post_algorithm: "bilinear" / "bicubic" / "lanczos" / None(off); post_factor multiplies the
                    width and height of the AI-upscaled image.
    progress_callback(done, total): optional, called before each file starts.
 
    Returns -1 for an invalid input/output path, otherwise a dict:
        {"processed": int, "skipped": int, "errors": [(filename, message), ...], "outputs": [paths]}
    """
    if not input_path or not os.path.exists(input_path):  # invalid input path
        return -1
    if not output_path:
        return -1
 
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in sorted(os.listdir(input_path))]
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = [input_path]
 
    os.makedirs(output_path, exist_ok=True)
 
    result = {"processed": 0, "skipped": 0, "errors": []}
    upsca = create_cugan_upscaler(model_path, scale)
 
    for i, full_path in enumerate(files):
        if progress_callback:
            progress_callback(i, len(files))
 
        name = os.path.basename(full_path)
 
        if not is_supported_image(full_path):
            result["skipped"] += 1
            continue

        filename = f"{prefix}{i}"
        save_path = build_output_path(full_path, output_path, filename, extension)
        # if (os.path.abspath(save_path) == os.path.abspath(full_path)):  #never overwrite the source, skip instead
            # result["skipped"] += 1
            # result["errors"].append((name, "output would overwrite the source file (change prefix, extension or output folder)"))
            # continue
        while (os.path.abspath(save_path) == os.path.abspath(full_path)): #alt: rename until no duplicate
            counter = 1
            filename = f"{prefix}{i}{_}{counter}"
            save_path = build_output_path(full_path, output_path, filename, extension)
            counter += 1
 
        try:
            img_out = inference(full_path, model_path, scale, tile_mode, cache_mode, alpha, upsca)
            img_out = post_process(img_out, post_algorithm, post_factor)
            save_np_image(img_out, save_path)
            result["processed"] += 1
        except Exception as e:
            result["errors"].append((name, str(e)))
 
    if progress_callback:
        progress_callback(len(files), len(files))
 
    return result



#tile_mode: 0-5, 0=no tiling, process whole image, use high vram, fast, 1=half crop, 2=recommened, 5=very low vram, slow

#cache_mode: 0=standard, fast if within vram, 1=compress cache data to 9 bit quant, can lead to rounding error, 2=no cache, slow but low vram

#alpha: start at 1, lower if output is too sharp or removing fine detail, increase to apply enhancement more heavy