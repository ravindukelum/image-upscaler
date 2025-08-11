@echo off
REM Build single-file EXE with PyInstaller
IF NOT EXIST .venv (
  python -m venv .venv
)
CALL .venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --noconfirm --windowed --onefile winimage_stock_pro.py
echo Build complete. Dist\winimage_stock_pro.exe
