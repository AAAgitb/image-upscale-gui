from PIL import Image

def resample_filter(pil_input, new_size, filter_type = "bilinear"):
    filter_map = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS
    }
    resample_filter = filter_map.get(filter_type.lower(), Image.Resampling.BILINEAR)
    
    #Check for alpha channel and apply premultiplied alpha ("RGBa")
    pil_img = pil_input
    if (pil_input.mode == "RGBA"):
        pil_img = pil_input.convert("RGBa")
        
    resized_pil_img = pil_img.resize(new_size, resample=resample_filter)
    
    #Convert back to standard RGBA if it was originally transparent
    if (pil_input.mode == "RGBA"):
        resized_pil_img = resized_pil_img.convert("RGBA")

    return resized_pil_img