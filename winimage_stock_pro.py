"""
WinImage Stock Pro — Convert & Upscale (Adobe Stock Mode) — v2
Scrollable UI + improved styling; pinned Run controls.
"""
import os, sys, io, csv, subprocess, queue, threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image, ImageOps, ImageStat
try:
    from PIL import ImageCms
    HAS_IMGCMS = True
except Exception:
    HAS_IMGCMS = False
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

SUPPORTED_INPUT_EXTS = {".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff",".gif"}
SUPPORTED_OUTPUT_FORMATS = ["auto","png","jpeg","webp","bmp","tiff"]
RESAMPLING_OPTIONS = {
    "Lanczos (high quality)": Image.LANCZOS,
    "Bicubic": Image.BICUBIC,
    "Bilinear": Image.BILINEAR,
    "Nearest (pixel art)": Image.NEAREST,
}
UPSCALE_FACTORS = ["1x (no upscale)","2x","4x"]
ADOBE_MIN_MP = 4.0
ADOBE_MAX_MP = 100.0
ADOBE_MAX_MB = 45

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def is_gif(p): return os.path.splitext(p)[1].lower() == ".gif"
def has_alpha(img):
    return img.mode in ("RGBA","LA") or (img.mode == "P" and "transparency" in img.info)
def mp_from_size(size): return (size[0]*size[1])/1_000_000.0
def exif_bytes_or_none(img):
    try: return img.info.get("exif")
    except Exception: return None

def to_srgb(img: Image.Image) -> Image.Image:
    if not HAS_IMGCMS:
        return img.convert("RGB") if img.mode not in ("RGB","RGBA") else img
    try:
        srgb_profile = ImageCms.createProfile("sRGB")
        if "icc_profile" in img.info:
            src = ImageCms.ImageCmsProfile(io.BytesIO(img.info["icc_profile"]))
            return ImageCms.profileToProfile(img, src, srgb_profile, outputMode="RGB")
        return img.convert("RGB")
    except Exception:
        return img.convert("RGB")

def convert_mode_for_format(img: Image.Image, fmt: str, background: Optional[Tuple[int,int,int]]):
    if fmt.lower() in ("jpeg","jpg"):
        if has_alpha(img):
            bg = background or (255,255,255)
            base = Image.new("RGB", img.size, bg)
            rgba = img.convert("RGBA")
            base.paste(rgba, mask=rgba.split()[-1])
            return base
        return img.convert("RGB")
    return img

def infer_out_ext(fmt: str, src_path: str) -> str:
    if fmt == "auto":
        return os.path.splitext(src_path)[1]
    mapping = {"jpeg":".jpg","jpg":".jpg","png":".png","webp":".webp","bmp":".bmp","tiff":".tif"}
    return mapping.get(fmt.lower(), f".{fmt.lower()}")

def load_first_frame(path: str) -> Image.Image:
    img = Image.open(path)
    if is_gif(path):
        try: img.seek(0)
        except Exception: pass
        return img.convert("RGBA")
    return img

def upscale_pillow(img: Image.Image, scale: int, resample_label: str) -> Image.Image:
    if scale == 1: return img
    resample = RESAMPLING_OPTIONS.get(resample_label, Image.LANCZOS)
    new_size = (img.width*scale, img.height*scale)
    return img.resize(new_size, resample=resample)

def run_realesrgan(exe_path: str, in_path: str, out_path: str, scale: int,
                   model: Optional[str]=None, tile: Optional[int]=None, gpu: Optional[int]=None) -> Tuple[bool,str]:
    try:
        cmd = [exe_path, "-i", in_path, "-o", out_path, "-s", str(scale)]
        if model: cmd += ["-n", model]
        if tile is not None: cmd += ["-t", str(tile)]
        if gpu is not None: cmd += ["-g", str(gpu)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        ok = (proc.returncode == 0) and os.path.exists(out_path)
        if ok: return True, "AI upscale ok"
        return False, f"AI upscale failed (code {proc.returncode})\n{proc.stderr.decode(errors='ignore')}"
    except FileNotFoundError:
        return False, "AI upscaler not found"
    except Exception as e:
        return False, f"AI upscaler error: {e}"

# --- lightweight QC heuristics using NumPy ---
def estimate_banding(img: Image.Image) -> float:
    g = img.convert("L").resize((256,256), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    grad = np.abs(a[:,1:] - a[:,:-1])
    v = float(grad.var())
    return max(0.0, min(1.0, 1.0 - (v / 200.0)))

def estimate_oversharp(img: Image.Image) -> float:
    g = img.convert("L").resize((256,256), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    # Laplacian: 4C - N - S - E - W
    pad = np.pad(a, 1, mode="edge")
    center = pad[1:-1,1:-1]
    north  = pad[:-2,1:-1]
    south  = pad[2:,1:-1]
    west   = pad[1:-1,:-2]
    east   = pad[1:-1,2:]
    lap = 4*center - north - south - east - west
    energy = float((lap**2).mean())
    var = float(a.var()) + 1e-6
    score = max(0.0, min(1.0, (energy/var)/50.0))
    return score

def jpeg_bytes_under(img: Image.Image, max_bytes: int, exif: Optional[bytes]=None) -> Tuple[bytes,int]:
    lo, hi = 75, 95
    best = None; best_q = 95
    while lo <= hi:
        q = (lo + hi)//2
        bio = io.BytesIO()
        params = {"format":"JPEG","quality": q, "subsampling": 0, "optimize": True}
        if exif: params["exif"] = exif
        img.save(bio, **params)
        b = bio.getvalue()
        if len(b) <= max_bytes:
            best = b; best_q = q; lo = q + 1
        else:
            hi = q - 1
    if best is None:
        bio = io.BytesIO()
        params = {"format":"JPEG","quality": 75, "subsampling": 0, "optimize": True}
        if exif: params["exif"] = exif
        img.save(bio, **params)
        best = bio.getvalue(); best_q = 75
    return best, best_q

@dataclass
class Config:
    out_dir: str
    out_format: str
    scale: int
    resample_label: str
    use_ai: bool
    ai_exe: Optional[str]
    ai_model: Optional[str]
    ai_tile: Optional[int]
    ai_gpu: Optional[int]
    preserve_exif: bool
    add_suffix: bool
    background_rgb: Optional[Tuple[int,int,int]]
    adobe_stock_mode: bool
    ai_content_checkbox: bool

# --- Scrollable Frame helper ---
class ScrollableSection(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0,0), window=self.inner, anchor="nw")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)      # Windows
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)  # Linux up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)  # Linux down

    def _on_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfig(self.inner_id, width=self.canvas.winfo_width())

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WinImage Stock Pro — Convert & Upscale (Adobe Stock Mode)")
        self.geometry("980x780")
        self.minsize(900, 720)
        self.queue = queue.Queue()
        self.worker = None
        self.cancel_flag = threading.Event()
        self._style()
        self._build_ui()

    def _style(self):
        style = ttk.Style()
        for preferred in ("vista", "xpnative", "clam", "default"):
            try:
                style.theme_use(preferred)
                break
            except Exception:
                continue
        style.configure("TLabel", padding=3)
        style.configure("TButton", padding=6)
        style.configure("TLabelframe", padding=8)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=(12,8))

    def _build_ui(self):
        pad = {"padx":10, "pady":8}

        # Top: scrollable sections 1-5
        self.scroll = ScrollableSection(self)
        self.scroll.pack(side="top", fill="both", expand=True)

        container = self.scroll.inner

        # 1) Files
        f_in = ttk.LabelFrame(container, text="1) Choose images")
        f_in.pack(fill="x", **pad)
        self.lst_files = tk.Listbox(f_in, height=8, selectmode=tk.EXTENDED)
        self.lst_files.pack(side="left", fill="both", expand=True, padx=(10,4), pady=10)
        right = ttk.Frame(f_in); right.pack(side="right", fill="y", padx=(4,10), pady=10)
        ttk.Button(right, text="Add Files", command=self.add_files).pack(fill="x", pady=4)
        ttk.Button(right, text="Add Folder", command=self.add_folder).pack(fill="x", pady=4)
        ttk.Button(right, text="Remove Selected", command=self.remove_selected).pack(fill="x", pady=4)
        ttk.Button(right, text="Clear", command=self.clear_files).pack(fill="x", pady=4)

        # 2) Output options
        f_out = ttk.LabelFrame(container, text="2) Output options")
        f_out.pack(fill="x", **pad)

        row = ttk.Frame(f_out); row.pack(fill="x", padx=10, pady=6)
        ttk.Label(row, text="Output folder:").pack(side="left")
        self.out_dir_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Pictures", "WinImage_Output"))
        ttk.Entry(row, textvariable=self.out_dir_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse", command=self.choose_out_dir).pack(side="left")

        row2 = ttk.Frame(f_out); row2.pack(fill="x", padx=10, pady=6)
        ttk.Label(row2, text="Output format:").pack(side="left")
        self.format_var = tk.StringVar(value="auto")
        ttk.Combobox(row2, textvariable=self.format_var, values=SUPPORTED_OUTPUT_FORMATS, state="readonly", width=10).pack(side="left", padx=(6,20))

        ttk.Label(row2, text="Upscale:").pack(side="left")
        self.scale_var = tk.StringVar(value=UPSCALE_FACTORS[0])
        ttk.Combobox(row2, textvariable=self.scale_var, values=UPSCALE_FACTORS, state="readonly", width=16).pack(side="left", padx=(6,20))

        ttk.Label(row2, text="Resampling:").pack(side="left")
        self.resample_var = tk.StringVar(value="Lanczos (high quality)")
        ttk.Combobox(row2, textvariable=self.resample_var, values=list(RESAMPLING_OPTIONS.keys()), state="readonly", width=24).pack(side="left")

        # 3) Adobe Stock mode
        f_stock = ttk.LabelFrame(container, text="3) Adobe Stock mode")
        f_stock.pack(fill="x", **pad)
        self.adobe_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_stock, text="Enable Adobe Stock mode (recommended)", variable=self.adobe_mode_var).pack(anchor="w", padx=10, pady=(8,2))
        self.ai_content_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_stock, text="This content is Generative AI (reminder for upload step)", variable=self.ai_content_var).pack(anchor="w", padx=10, pady=(0,8))
        ttk.Label(f_stock, text="Guardrails: JPEG + sRGB, ≥4 MP, ≤100 MP, ≤45 MB; upscale OFF by default.", style="Header.TLabel").pack(anchor="w", padx=10, pady=(4,8))

        # 4) AI upscaler
        f_ai = ttk.LabelFrame(container, text="4) AI upscaler (optional)")
        f_ai.pack(fill="x", **pad)
        self.use_ai_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f_ai, text="Use external AI upscaler (Real-ESRGAN ncnn Vulkan)", variable=self.use_ai_var).pack(anchor="w", padx=10, pady=(8,2))

        row_ai1 = ttk.Frame(f_ai); row_ai1.pack(fill="x", padx=10, pady=6)
        ttk.Label(row_ai1, text="Path to realesrgan-ncnn-vulkan.exe:").pack(side="left")
        self.ai_path_var = tk.StringVar()
        ttk.Entry(row_ai1, textvariable=self.ai_path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row_ai1, text="Browse", command=self.choose_ai_exe).pack(side="left")

        row_ai2 = ttk.Frame(f_ai); row_ai2.pack(fill="x", padx=10, pady=6)
        ttk.Label(row_ai2, text="Model:").pack(side="left")
        self.ai_model_var = tk.StringVar(value="")
        ttk.Entry(row_ai2, textvariable=self.ai_model_var, width=20).pack(side="left", padx=(6,20))
        ttk.Label(row_ai2, text="Tile:").pack(side="left")
        self.ai_tile_var = tk.StringVar(value="0")
        ttk.Entry(row_ai2, textvariable=self.ai_tile_var, width=6).pack(side="left", padx=(6,20))
        ttk.Label(row_ai2, text="GPU:").pack(side="left")
        self.ai_gpu_var = tk.StringVar(value="0")
        ttk.Entry(row_ai2, textvariable=self.ai_gpu_var, width=6).pack(side="left")

        # 5) Advanced
        f_adv = ttk.LabelFrame(container, text="5) Advanced")
        f_adv.pack(fill="x", **pad)
        row_a1 = ttk.Frame(f_adv); row_a1.pack(fill="x", padx=10, pady=6)
        self.preserve_exif_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_a1, text="Preserve JPEG EXIF when possible", variable=self.preserve_exif_var).pack(side="left", padx=(0,20))
        self.suffix_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row_a1, text="Add _{scale}x suffix when upscaling", variable=self.suffix_var).pack(side="left")

        row_a2 = ttk.Frame(f_adv); row_a2.pack(fill="x", padx=10, pady=6)
        ttk.Label(row_a2, text="Background for transparency → JPEG (R,G,B):").pack(side="left")
        self.bg_r = tk.StringVar(value="255"); self.bg_g = tk.StringVar(value="255"); self.bg_b = tk.StringVar(value="255")
        for var in (self.bg_r, self.bg_g, self.bg_b):
            ttk.Entry(row_a2, textvariable=var, width=4).pack(side="left", padx=3)

        # Bottom: pinned Run area
        run = ttk.Frame(self, padding=(10,10))
        run.pack(side="bottom", fill="x")
        ttk.Separator(run).pack(fill="x", pady=(0,8))
        self.prog = ttk.Progressbar(run, mode="determinate")
        self.prog.pack(fill="x", padx=2, pady=(0,8))
        btns = ttk.Frame(run)
        btns.pack(fill="x")
        self.btn_start = ttk.Button(btns, text="Start", style="Big.TButton", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(btns, text="Cancel", style="Big.TButton", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=10)

        # Log
        f_log = ttk.LabelFrame(container, text="Log")
        f_log.pack(fill="both", expand=True, **pad)
        self.txt_log = tk.Text(f_log, height=10, wrap="word")
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=10)

    # --- UI actions ---
    def add_files(self):
        files = filedialog.askopenfilenames(title="Select images",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff;*.gif"),("All files","*.*")])
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_INPUT_EXTS:
                self.lst_files.insert(tk.END, f)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder: return
        for root, _, files in os.walk(folder):
            for name in files:
                p = os.path.join(root, name)
                if os.path.splitext(name)[1].lower() in SUPPORTED_INPUT_EXTS:
                    self.lst_files.insert(tk.END, p)

    def remove_selected(self):
        sel = list(self.lst_files.curselection())
        for idx in reversed(sel):
            self.lst_files.delete(idx)

    def clear_files(self):
        self.lst_files.delete(0, tk.END)

    def choose_out_dir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.out_dir_var.set(d)

    def choose_ai_exe(self):
        p = filedialog.askopenfilename(title="Select realesrgan-ncnn-vulkan.exe",
                                       filetypes=[("Executable","*.exe"),("All files","*.*")])
        if p: self.ai_path_var.set(p)

    def log(self, msg: str):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.update_idletasks()

    def set_running(self, running: bool):
        self.btn_start.config(state="disabled" if running else "normal")
        self.btn_cancel.config(state="normal" if running else "disabled")

    def start(self):
        files = list(self.lst_files.get(0, tk.END))
        if not files:
            messagebox.showwarning("No files", "Please add images to process."); return
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output folder", "Please choose an output folder."); return

        scale_label = self.scale_var.get(); scale = 1
        for s in (1,2,4):
            if scale_label.startswith(f"{s}x"): scale = s; break

        try:
            bg = (int(self.bg_r.get()), int(self.bg_g.get()), int(self.bg_b.get()))
        except ValueError:
            bg = (255,255,255)

        cfg = Config(
            out_dir=out_dir,
            out_format=self.format_var.get(),
            scale=scale,
            resample_label=self.resample_var.get(),
            use_ai=self.use_ai_var.get(),
            ai_exe=self.ai_path_var.get().strip() or None,
            ai_model=(self.ai_model_var.get().strip() or None),
            ai_tile=(int(self.ai_tile_var.get()) if self.ai_tile_var.get().strip().isdigit() else None),
            ai_gpu=(int(self.ai_gpu_var.get()) if self.ai_gpu_var.get().strip().isdigit() else None),
            preserve_exif=True,
            add_suffix=True,
            background_rgb=bg,
            adobe_stock_mode=True if hasattr(self, "adobe_mode_var") and self.adobe_mode_var.get() else False,
            ai_content_checkbox=True if hasattr(self, "ai_content_var") and self.ai_content_var.get() else False,
        )

        if cfg.adobe_stock_mode:
            self.format_var.set("jpeg")
            if scale != 1 and not messagebox.askyesno("Upscale in Adobe Stock mode?",
                "Adobe Stock discourages enlarging. Continue with AI upscale anyway?"):
                self.scale_var.set("1x (no upscale)")
                cfg.scale = 1

        self.txt_log.delete("1.0", tk.END)
        self.cancel_flag.clear()
        self.set_running(True)
        self.prog.configure(maximum=len(files), value=0)

        self.worker = threading.Thread(target=self._worker_run, args=(cfg, files), daemon=True)
        self.worker.start()
        self.after(120, self._poll_queue)

    def cancel(self):
        if self.worker and self.worker.is_alive():
            self.cancel_flag.set()
            self.log("Cancelling…")

    def _worker_run(self, cfg: 'Config', files: List[str]):
        processed = 0
        ensure_dir(cfg.out_dir)
        csv_path = os.path.join(cfg.out_dir, "WinImageStockPro_log.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["source","output","w","h","megapixels","size_MB","jpeg_quality","warnings"])
            for path in files:
                if self.cancel_flag.is_set():
                    self.queue.put(("log","Cancelled by user.")); break
                ok, msg, out_path, meta = process_one(path, cfg)
                self.queue.put(("log", ("✔ " if ok else "✖ ") + msg))
                processed += 1
                self.queue.put(("progress", processed))
                if meta:
                    writer.writerow([path, out_path or "", meta.get("w"), meta.get("h"),
                                     f"{meta.get('mp',0):.2f}", f"{meta.get('mb',0):.2f}",
                                     meta.get("jpeg_q",""), "; ".join(meta.get("warnings",[]))])
        self.queue.put(("done", None))

    def _poll_queue(self):
        try:
            while True:
                typ, payload = self.queue.get_nowait()
                if typ == "log": self.log(payload)
                elif typ == "progress": self.prog.configure(value=payload)
                elif typ == "done": self.set_running(False)
        except queue.Empty:
            pass
        finally:
            if self.worker and self.worker.is_alive():
                self.after(120, self._poll_queue)

def process_one(path: str, cfg: 'Config'):
    meta = {"w":0,"h":0,"mp":0.0,"mb":0.0,"jpeg_q": "", "warnings":[]}
    try:
        img = load_first_frame(path)
        if cfg.adobe_stock_mode:
            img = to_srgb(img)

        src_ext = os.path.splitext(path)[1]
        scale = cfg.scale
        temp_in = None; temp_out = None

        mp0 = mp_from_size(img.size)
        meta["w"], meta["h"], meta["mp"] = img.width, img.height, mp0

        if cfg.adobe_stock_mode:
            if mp0 < ADOBE_MIN_MP and scale == 1:
                meta["warnings"].append(f"Below 4 MP ({mp0:.2f} MP)")
            if mp0 > ADOBE_MAX_MP:
                return False, f"Too large (>100 MP): {os.path.basename(path)}", "", meta

        if cfg.use_ai and cfg.ai_exe and scale > 1:
            base_noext = os.path.splitext(os.path.basename(path))[0]
            temp_in = os.path.join(cfg.out_dir, f".__temp_in__{base_noext}.png")
            img.save(temp_in, format="PNG")
            temp_out = os.path.join(cfg.out_dir, f".__temp_out__{base_noext}.png")
            ok_ai, msg = run_realesrgan(cfg.ai_exe, temp_in, temp_out, scale, cfg.ai_model, cfg.ai_tile, cfg.ai_gpu)
            if ok_ai:
                img = Image.open(temp_out)
            else:
                meta["warnings"].append("AI upscale failed; used Lanczos")
                img = upscale_pillow(img, scale, cfg.resample_label)
        else:
            img = upscale_pillow(img, scale, cfg.resample_label)

        target_fmt = cfg.out_format
        if cfg.adobe_stock_mode:
            target_fmt = "jpeg"
        elif target_fmt == "auto":
            target_fmt = src_ext.lstrip(".").lower()
            if target_fmt == "jpg": target_fmt = "jpeg"

        img = convert_mode_for_format(img, target_fmt, cfg.background_rgb)

        base = os.path.splitext(os.path.basename(path))[0]
        suffix = f"_{cfg.scale}x" if (cfg.add_suffix and cfg.scale>1) else ""
        out_ext = infer_out_ext(target_fmt, path)
        out_name = f"{base}{suffix}{out_ext}"
        out_path = os.path.join(cfg.out_dir, out_name)

        try:
            bscore = estimate_banding(img); oscore = estimate_oversharp(img)
            if bscore > 0.7: meta["warnings"].append("Possible banding")
            if oscore > 0.7: meta["warnings"].append("Possible oversharpening")
        except Exception:
            pass

        if target_fmt in ("jpeg","jpg"):
            exifb = exif_bytes_or_none(Image.open(path)) if cfg.preserve_exif else None
            if cfg.adobe_stock_mode:
                jpeg_bytes, q_used = jpeg_bytes_under(img, ADOBE_MAX_MB*1024*1024, exifb)
                with open(out_path, "wb") as f: f.write(jpeg_bytes)
                meta["jpeg_q"] = q_used
                meta["mb"] = os.path.getsize(out_path)/1024/1024
            else:
                img.save(out_path, format="JPEG", quality=95, subsampling=0, optimize=True, exif=exifb if exifb else None)
                meta["mb"] = os.path.getsize(out_path)/1024/1024
        elif target_fmt == "png":
            img.save(out_path, format="PNG", optimize=True, compress_level=6)
            meta["mb"] = os.path.getsize(out_path)/1024/1024
        elif target_fmt == "webp":
            img.save(out_path, format="WEBP", quality=95, method=6, lossless=False)
            meta["mb"] = os.path.getsize(out_path)/1024/1024
        elif target_fmt == "tiff":
            img.save(out_path, format="TIFF", compression="tiff_lzw")
            meta["mb"] = os.path.getsize(out_path)/1024/1024
        elif target_fmt == "bmp":
            img.save(out_path, format="BMP")
            meta["mb"] = os.path.getsize(out_path)/1024/1024
        else:
            img.save(out_path)
            meta["mb"] = os.path.getsize(out_path)/1024/1024

        for p in (temp_in, temp_out):
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

        return True, f"Saved → {out_path}", out_path, meta
    except Exception as e:
        return False, f"Error: {os.path.basename(path)} — {e}", "", meta

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass
    app = App()
    app.mainloop()
