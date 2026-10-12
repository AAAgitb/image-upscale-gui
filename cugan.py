import torch
import numpy as np
from PIL import Image
import re
from upcunet_v3 import RealWaifuUpScaler
import gc

def create_cugan_upscaler(model_path, scale):
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

    return RealWaifuUpScaler(scale=scale, weight_path=model_path, half=True if device != 'cpu' else False, device=device)

def get_device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(0)
    else:
        return "CPU"

def save_np_image(image_np, save_path):
    Image.fromarray(image_np).save(save_path)

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
        output_np = upscalerCPU(image_np, tile_mode=tile_mode, cache_mode=2, alpha=1)

    return output_np



#tile_mode: 0-5, 0=no tiling, process whole image, use high vram, fast, 1=half crop, 2=recommened, 5=very low vram, slow

#cache_mode: 0=standard, fast if within vram, 1=compress cache data to 9 bit quant, can lead to rounding error, 2=no cache, slow but low vram

#alpha: start at 1, lower if output is too sharp or removing fine detail, increase to apply enhancement more heavy