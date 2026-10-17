# image-upscale-gui
Image upscaling GUI supporting AI and traditional resizing algorithms


## Requirements (Windows)
- Python 3.10+
- Pillow 9.1+:
```bash
pip install pillow
```

- Pytorch (install the version corresponding to your hardware):
-- CPU only: `pip install torch torchvision`
-- Nvidia GPU CUDA 13.2: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132`
-- Nvidia GPU CUDA 12.6: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
-- Intel XPU: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/xpu`


## Getting a model
This repository does not include model weights. User need to load model manually

### CUGAN
Download Real-CUGAN `.pth` weights from the [Real-CUGAN project](https://github.com/bilibili/ailab/blob/main/Real-CUGAN/README_EN.md#executable-file)

Please keep the original file name `up2x`, `up3x` or `up4x` in the models (as in the original downloads) so the scale is
detected automatically. If the name has been changed, the program will ask you for the scale.

## Acknowledgements

- [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) by bilibili / ailab, for the CUGAN models and `upcunet_v3.py`
- [PyTorch](https://pytorch.org/) and [Pillow](https://python-pillow.org/)