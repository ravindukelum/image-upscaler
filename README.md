# WinImage Stock Pro — v2 (Scrollable + Polished UI)

Changes in v2:
- **Scrollable content** for sections 1–5 so you can always see the **Run** controls.
- **Pinned Run bar** (Start/Cancel + progress) always visible at the bottom.
- Refreshed **ttk styling** (larger fonts, spacing, clean vista/clam fallback).
- Removed SciPy dependency; quality checks now use a fast NumPy Laplacian.
- Same Adobe Stock mode guardrails and Real-ESRGAN integration.

## Quick start (dev)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python winimage_stock_pro.py
```

## Build a single-file EXE
```
.venv\Scripts\pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --windowed --onefile winimage_stock_pro.py
```
