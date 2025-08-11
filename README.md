# WinImage Stock Pro — v2.1

Windows GUI to prep images for **Adobe Stock** with optional **AI upscaling** (Real‑ESRGAN).  
Scrollable UI, pinned Run controls, clean .gitignore.

## Features
- Batch convert: PNG/JPEG/WEBP/BMP/TIFF (GIF→first frame)
- **Adobe Stock mode**: JPEG+sRGB, ≥4 MP, ≤100 MP, ≤45 MB; upscale off by default
- Optional AI Upscaler (Real‑ESRGAN): 2×/4× with model/tile/GPU options
- Progress bar, per‑file log, CSV summary

## Quick start
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python winimage_stock_pro.py
```

## Build EXE
```
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --windowed --onefile winimage_stock_pro.py
```

## Real‑ESRGAN setup
1. Download Windows zip: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases
2. Extract: keep `realesrgan-ncnn-vulkan.exe` and `models/` together.
3. In the app: enable AI upscaler → Browse to the `.exe` → pick 2×/4×.

## Git
This repo includes a `.gitignore` that skips venv/build caches and (optionally) Real‑ESRGAN binaries and `models/`.
