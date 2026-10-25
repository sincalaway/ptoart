"""Desktop window for ptoart (Windows EXE double-click entry).

Pure-Python Tk UI via CustomTkinter. Model discovery and option mapping live in
plain functions so they are testable without a display.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

STYLE_NAMES = {
    "hayao": "宫崎骏",
    "shinkai": "新海诚",
    "paprika": "今敏 · 红辣椒",
    "celeba_distill": "CelebA 清新",
    "face_paint_512_v1": "人像油彩 v1",
    "face_paint_512_v2": "人像油彩 v2",
}
UPSCALE_NAMES = {"anime": "动漫（快）", "general": "通用（细节多）"}
MODES = {"face": "卡通头像", "photo": "整张照片", "style": "动漫风格"}
IMAGE_TYPES = (("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*"))


@dataclass
class Models:
    root: Path
    face: bool = False          # photo2cartoon_weights.pt + seg_model_384.pb
    styles: dict = field(default_factory=dict)    # key -> path
    upscalers: dict = field(default_factory=dict)  # key -> path

    @property
    def any(self):
        return self.face or bool(self.styles)


def find_model_dir():
    """weights/ beside the EXE (or CWD), else PTOART_MODEL_DIR, else ~/.cache/ptoart/models."""
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / "weights")
    if os.environ.get("PTOART_MODEL_DIR"):
        candidates.append(Path(os.environ["PTOART_MODEL_DIR"]).expanduser())
    candidates += [Path.cwd() / "weights", Path("~/.cache/ptoart/models").expanduser()]
    for folder in candidates:
        if folder.is_dir():
            return folder.resolve()
    return candidates[0].resolve()


def scan_models(root):
    root = Path(root)
    models = Models(root=root)
    models.face = (root / "photo2cartoon_weights.pt").is_file() and (root / "seg_model_384.pb").is_file()
    styles = root / "styles"
    if styles.is_dir():
        found = {p.stem: p for p in styles.glob("*.onnx")}
        order = list(STYLE_NAMES) + sorted(k for k in found if k not in STYLE_NAMES)
        models.styles = {k: found[k] for k in order if k in found}
    upscale = root / "upscale"
    if upscale.is_dir():
        models.upscalers = {p.stem: p for p in sorted(upscale.glob("*.onnx"))}
    return models


def style_label(key):
    return STYLE_NAMES.get(key, key)


def build_config(models, *, mode, style=None, size=1024, strength=1.0, transparent=False, upscale=None):
    """Map UI choices to an ArtConfig (Windows build: CPU + SFD detector)."""
    from ptoart.config import ArtConfig

    return ArtConfig(
        backend="torch", device="cpu", detector="sfd", model_dir=models.root,
        output=mode, size=size if mode == "face" else 256, strength=strength,
        background=None if (transparent and mode == "face") else (255, 255, 255),
        style=str(models.styles[style]) if mode == "style" else None,
        upscale=str(models.upscalers[upscale]) if upscale else None,
    )


def output_name(source, mode, style=None, transparent=False):
    stem = Path(source).stem
    if mode == "face":
        return f"{stem}-头像.png"
    if mode == "photo":
        return f"{stem}-卡通.jpg"
    return f"{stem}-{style_label(style)}.jpg"


def friendly_error(exc):
    from ptoart.errors import ImageError, ModelError, NoFaceError

    if isinstance(exc, NoFaceError):
        return "没有检测到人脸。请换一张正脸、清晰、光线好的照片；只想要动漫效果可选「动漫风格」。"
    if isinstance(exc, ModelError):
        return f"模型加载失败：{exc}\n\n请确认模型文件放在 weights 文件夹里且文件名没有被改动。"
    if isinstance(exc, ImageError):
        return f"图片无法处理：{exc}"
    text = str(exc)
    if "urlopen" in text or "HTTP" in text or "Connection" in text:
        return "首次使用需要联网下载人脸检测数据（只下载一次）。请连接网络后重试。"
    return f"处理失败：{type(exc).__name__}: {exc}"


def hide_own_console():
    """When double-clicked, the console belongs only to us: hide it."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        processes = (ctypes.c_uint * 4)()
        if kernel.GetConsoleProcessList(processes, 4) <= 2:
            ctypes.windll.user32.ShowWindow(kernel.GetConsoleWindow(), 0)
    except Exception:
        pass


# --------------------------------------------------------------------------- UI


def run_gui(screenshot=None):
    from tkinter import filedialog, messagebox

    import customtkinter as ctk
    import numpy as np
    from PIL import Image

    from ptoart.io import read_image, write_image

    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    FONT = "Microsoft YaHei UI" if os.name == "nt" else None
    ACCENT = ("#5B5BD6", "#7C7CF0")
    ACCENT_HOVER = ("#4A4AC4", "#6A6AE0")
    CARD = ("#FFFFFF", "#23232B")
    BG = ("#F3F3F8", "#17171C")
    MUTED = ("#6B6B7B", "#9A9AAA")

    def font(size=13, weight="normal"):
        return ctk.CTkFont(family=FONT, size=size, weight=weight)

    class App(ctk.CTk):
        def __init__(self):
            super().__init__(fg_color=BG)
            self.title("ptoart · 照片变卡通")
            self.geometry("1180x760")
            self.minsize(980, 640)
            self.models = scan_models(find_model_dir())
            self.sources: list[Path] = []
            self.result = None
            self.result_mode = None
            self.pipelines = {}
            self.events = queue.Queue()
            self.working = False
            self._ui_build()
            self._ui_refresh_models()
            self.after(100, self._ui_poll)
            self.protocol("WM_DELETE_WINDOW", self._ui_close)

        # ---- layout
        def _ui_build(self):
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)

            side = ctk.CTkFrame(self, width=320, corner_radius=0, fg_color=CARD)
            side.grid(row=0, column=0, sticky="nsw")
            side.grid_propagate(False)
            side.grid_columnconfigure(0, weight=1)
            r = 0
            ctk.CTkLabel(side, text="ptoart", font=font(28, "bold"), text_color=ACCENT).grid(
                row=r, column=0, padx=24, pady=(26, 0), sticky="w")
            r += 1
            ctk.CTkLabel(side, text="照片一键变卡通 · 全部本地离线处理", font=font(12), text_color=MUTED).grid(
                row=r, column=0, padx=24, pady=(0, 18), sticky="w")
            r += 1

            self.pick_btn = ctk.CTkButton(side, text="＋  选择照片", height=44, font=font(15, "bold"),
                                          fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._ui_pick)
            self.pick_btn.grid(row=r, column=0, padx=24, pady=(0, 18), sticky="ew")
            r += 1

            self._ui_section(side, "效果", r)
            r += 1
            self.mode = ctk.CTkSegmentedButton(side, values=list(MODES.values()), font=font(13),
                                               selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
                                               command=lambda _v: self._ui_update_options())
            self.mode.grid(row=r, column=0, padx=24, pady=(4, 6), sticky="ew")
            r += 1
            self.mode_hint = ctk.CTkLabel(side, text="", font=font(12), text_color=MUTED, wraplength=270,
                                          justify="left")
            self.mode_hint.grid(row=r, column=0, padx=24, pady=(0, 10), sticky="w")
            r += 1

            self.opts = ctk.CTkFrame(side, fg_color="transparent")
            self.opts.grid(row=r, column=0, padx=24, sticky="ew")
            r += 1
            self.opts.grid_columnconfigure(0, weight=1)
            self.style_menu = ctk.CTkOptionMenu(self.opts, values=["—"], font=font(13), height=34,
                                                fg_color=ACCENT, button_color=ACCENT_HOVER,
                                                button_hover_color=ACCENT_HOVER, dropdown_font=font(13))
            self.size_menu = ctk.CTkSegmentedButton(self.opts, values=["512", "1024", "2048"], font=font(13),
                                                    selected_color=ACCENT, selected_hover_color=ACCENT_HOVER)
            self.size_menu.set("1024")
            self.size_label = ctk.CTkLabel(self.opts, text="头像尺寸（像素）", font=font(12), text_color=MUTED)
            self.style_label = ctk.CTkLabel(self.opts, text="风格", font=font(12), text_color=MUTED)
            self.strength_label = ctk.CTkLabel(self.opts, text="卡通强度 100%", font=font(12), text_color=MUTED)
            self.strength = ctk.CTkSlider(self.opts, from_=0, to=100, number_of_steps=20, button_color=ACCENT,
                                          progress_color=ACCENT, command=self._ui_strength_changed)
            self.strength.set(100)
            self.transparent = ctk.CTkSwitch(self.opts, text="透明背景（PNG）", font=font(13),
                                             progress_color=ACCENT)

            self._ui_section(side, "清晰度", r)
            r += 1
            self.upscale = ctk.CTkSwitch(side, text="超分辨率增强", font=font(13), progress_color=ACCENT,
                                         command=self._ui_update_options)
            self.upscale.grid(row=r, column=0, padx=24, pady=(4, 4), sticky="w")
            r += 1
            self.upscale_menu = ctk.CTkOptionMenu(side, values=["—"], font=font(13), height=32,
                                                  fg_color=ACCENT, button_color=ACCENT_HOVER,
                                                  button_hover_color=ACCENT_HOVER, dropdown_font=font(13))
            self.upscale_menu.grid(row=r, column=0, padx=24, pady=(0, 10), sticky="ew")
            r += 1

            side.grid_rowconfigure(r, weight=1)
            r += 1
            self.model_status = ctk.CTkLabel(side, text="", font=font(12), text_color=MUTED, justify="left",
                                             wraplength=270)
            self.model_status.grid(row=r, column=0, padx=24, pady=(0, 6), sticky="w")
            r += 1
            links = ctk.CTkFrame(side, fg_color="transparent")
            links.grid(row=r, column=0, padx=18, pady=(0, 20), sticky="ew")
            r += 1
            for text, cmd in (("打开模型文件夹", self._ui_open_models), ("重新检测模型", self._ui_rescan)):
                ctk.CTkButton(links, text=text, font=font(12), height=28, fg_color="transparent",
                              text_color=ACCENT, hover_color=BG, command=cmd).pack(side="left", padx=4)

            # main area
            main = ctk.CTkFrame(self, fg_color="transparent")
            main.grid(row=0, column=1, sticky="nsew", padx=22, pady=22)
            main.grid_columnconfigure((0, 1), weight=1, uniform="p")
            main.grid_rowconfigure(0, weight=1)
            self.before = self._ui_panel(main, "原图", 0)
            self.after_panel = self._ui_panel(main, "效果", 1)

            bar = ctk.CTkFrame(main, fg_color="transparent")
            bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(16, 0))
            bar.grid_columnconfigure(0, weight=1)
            self.status = ctk.CTkLabel(bar, text="选择一张照片开始", font=font(13), text_color=MUTED, anchor="w")
            self.status.grid(row=0, column=0, sticky="ew")
            self.progress = ctk.CTkProgressBar(bar, mode="indeterminate", progress_color=ACCENT, height=6)
            self.save_btn = ctk.CTkButton(bar, text="保存结果", width=120, height=42, font=font(14),
                                          fg_color="transparent", border_width=2, border_color=ACCENT,
                                          text_color=ACCENT, hover_color=CARD, command=self._ui_save,
                                          state="disabled")
            self.save_btn.grid(row=0, column=1, padx=(10, 0))
            self.go_btn = ctk.CTkButton(bar, text="开始转换  →", width=170, height=42, font=font(15, "bold"),
                                        fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._ui_start,
                                        state="disabled")
            self.go_btn.grid(row=0, column=2, padx=(10, 0))

            self.mode.set(MODES["face"])

        def _ui_section(self, parent, text, row):
            ctk.CTkLabel(parent, text=text, font=font(13, "bold")).grid(row=row, column=0, padx=24,
                                                                        pady=(8, 0), sticky="w")

        def _ui_panel(self, parent, title, col):
            card = ctk.CTkFrame(parent, corner_radius=16, fg_color=CARD)
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col else 0, 10 if col == 0 else 0))
            card.grid_rowconfigure(1, weight=1)
            card.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(card, text=title, font=font(13, "bold"), text_color=MUTED).grid(
                row=0, column=0, padx=18, pady=(14, 0), sticky="w")
            label = ctk.CTkLabel(card, text="", font=font(13), text_color=MUTED)
            label.grid(row=1, column=0, sticky="nsew", padx=14, pady=14)
            label.bind("<Configure>", lambda _e, lb=label: self._ui_redraw(lb))
            label.image_data = None
            return label

        # ---- images
        def _ui_show(self, label, image, placeholder=""):
            label.image_data = image
            if image is None:
                label.configure(image=None, text=placeholder)
                label._ctk_img = None
            self._ui_redraw(label)

        def _ui_redraw(self, label):
            image = getattr(label, "image_data", None)
            if image is None:
                return
            w, h = max(label.winfo_width() - 8, 50), max(label.winfo_height() - 8, 50)
            scale = min(w / image.width, h / image.height, 1.0)
            size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=size)
            label._ctk_img = ctk_img
            label.configure(image=ctk_img, text="")

        # ---- options
        def _ui_mode_key(self):
            return next(k for k, v in MODES.items() if v == self.mode.get())

        def _ui_update_options(self):
            for w in self.opts.winfo_children():
                w.grid_forget()
            mode = self._ui_mode_key()
            hints = {
                "face": "最大的一张脸生成卡通头像，适合做头像。",
                "photo": "照片里每张脸都卡通化并贴回原图，身体和背景不变。",
                "style": "整张图变成动漫画风，风景、全身照都可以。",
            }
            self.mode_hint.configure(text=hints[mode])
            row = 0
            if mode == "style":
                self.style_label.grid(row=row, column=0, sticky="w")
                row += 1
                self.style_menu.grid(row=row, column=0, sticky="ew", pady=(2, 10))
                row += 1
            if mode == "face":
                self.size_label.grid(row=row, column=0, sticky="w")
                row += 1
                self.size_menu.grid(row=row, column=0, sticky="ew", pady=(2, 10))
                row += 1
            if mode in ("face", "photo"):
                self.strength_label.grid(row=row, column=0, sticky="w")
                row += 1
                self.strength.grid(row=row, column=0, sticky="ew", pady=(2, 10))
                row += 1
            if mode == "face":
                self.transparent.grid(row=row, column=0, sticky="w", pady=(0, 10))
                row += 1
            if self.models.upscalers and self.upscale.get():
                self.upscale_menu.grid()
            else:
                self.upscale_menu.grid_remove()
            self._ui_update_ready()

        def _ui_strength_changed(self, value):
            self.strength_label.configure(text=f"卡通强度 {int(value)}%")

        def _ui_refresh_models(self):
            m = self.models
            styles = [style_label(k) for k in m.styles] or ["未找到风格模型"]
            self.style_menu.configure(values=styles)
            self.style_menu.set(styles[0])
            ups = [UPSCALE_NAMES.get(k, k) for k in m.upscalers] or ["未找到超分模型"]
            self.upscale_menu.configure(values=ups)
            self.upscale_menu.set(ups[0])
            if m.upscalers:
                self.upscale.configure(state="normal")
                if not self.upscale.get():
                    self.upscale.select()
            else:
                self.upscale.deselect()
                self.upscale.configure(state="disabled")
            ok, no = "✓", "✗"
            self.model_status.configure(text=(
                f"{ok if m.face else no} 卡通头像模型    {ok if m.styles else no} 风格 {len(m.styles)} 种    "
                f"{ok if m.upscalers else no} 超分\n模型文件夹：{m.root}"))
            self._ui_update_options()

        def _ui_rescan(self):
            self.models = scan_models(find_model_dir())
            self._ui_close_pipelines()
            self._ui_refresh_models()

        def _ui_open_models(self):
            self.models.root.mkdir(parents=True, exist_ok=True)
            _open_path(self.models.root)

        def _ui_update_ready(self):
            mode = self._ui_mode_key()
            have = self.models.face if mode != "style" else bool(self.models.styles)
            ready = bool(self.sources) and have and not self.working
            self.go_btn.configure(state="normal" if ready else "disabled")
            if not have and not self.working:
                need = ("photo2cartoon_weights.pt 和 seg_model_384.pb" if mode != "style"
                        else "styles 文件夹中的风格模型（如 hayao.onnx）")
                self.status.configure(text=f"缺少模型：请把 {need} 放进模型文件夹，再点「重新检测模型」")
            elif not self.working and not self.sources:
                self.status.configure(text="选择一张照片开始")

        # ---- actions
        def _ui_pick(self):
            paths = filedialog.askopenfilenames(title="选择照片（可多选）", filetypes=IMAGE_TYPES)
            if not paths:
                return
            self.sources = [Path(p) for p in paths]
            try:
                self._ui_show(self.before, Image.fromarray(read_image(self.sources[0])))
            except Exception as exc:
                messagebox.showerror("ptoart", friendly_error(exc))
                self.sources = []
                return
            self._ui_show(self.after_panel, None, "")
            self.result = None
            self.save_btn.configure(state="disabled")
            extra = f"（共 {len(self.sources)} 张，结果自动保存到照片旁的 ptoart输出 文件夹）" if len(self.sources) > 1 else ""
            self.status.configure(text=f"{self.sources[0].name}{extra}")
            self._ui_update_ready()

        def _ui_options(self):
            mode = self._ui_mode_key()
            style = None
            if mode == "style":
                style = next((k for k in self.models.styles if style_label(k) == self.style_menu.get()), None)
            upscale = None
            if self.upscale.get() and self.models.upscalers:
                upscale = next((k for k in self.models.upscalers
                                if UPSCALE_NAMES.get(k, k) == self.upscale_menu.get()), None)
            return dict(mode=mode, style=style, size=int(self.size_menu.get()),
                        strength=round(self.strength.get() / 100, 2),
                        transparent=bool(self.transparent.get()) and mode == "face", upscale=upscale)

        def _ui_start(self):
            if self.working or not self.sources:
                return
            opts = self._ui_options()
            self.working = True
            self.go_btn.configure(state="disabled", text="处理中…")
            self.pick_btn.configure(state="disabled")
            self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
            self.progress.start()
            threading.Thread(target=self._ui_work, args=(list(self.sources), opts), daemon=True).start()

        def _ui_pipeline(self, opts):
            from ptoart.pipeline import ArtPipeline

            key = tuple(sorted(opts.items()))
            if key not in self.pipelines:
                if opts["mode"] != "style" and not any(k for k in self.pipelines if dict(k)["mode"] != "style"):
                    self.events.put(("status", "正在加载模型（首次使用需联网下载人脸检测数据，请稍候）…"))
                else:
                    self.events.put(("status", "正在加载模型…"))
                self.pipelines[key] = ArtPipeline(build_config(self.models, **opts))
            return self.pipelines[key]

        def _ui_work(self, sources, opts):
            try:
                pipeline = self._ui_pipeline(opts)
                many = len(sources) > 1
                failures = []
                last = None
                for i, src in enumerate(sources, 1):
                    self.events.put(("status", f"正在处理 {src.name}（{i}/{len(sources)}）…"))
                    try:
                        image = read_image(src)
                        self.events.put(("before", image))
                        last = pipeline.convert(image)
                        self.events.put(("result", (last, opts, src)))
                        if many:
                            folder = src.parent / "ptoart输出"
                            folder.mkdir(exist_ok=True)
                            write_image(folder / output_name(src, opts["mode"], opts["style"]), last,
                                        overwrite=True)
                    except Exception as exc:  # keep going in batch mode
                        if not many:
                            raise
                        failures.append(f"{src.name}：{friendly_error(exc)}")
                if many:
                    self.events.put(("batch_done", (sources[0].parent / "ptoart输出", len(sources), failures)))
                else:
                    self.events.put(("done", None))
            except Exception as exc:
                self.events.put(("error", friendly_error(exc)))

        def _ui_poll(self):
            try:
                while True:
                    kind, data = self.events.get_nowait()
                    if kind == "status":
                        self.status.configure(text=data)
                    elif kind == "before":
                        self._ui_show(self.before, Image.fromarray(data))
                    elif kind == "result":
                        array, opts, src = data
                        self.result, self.result_opts, self.result_src = array, opts, src
                        self._ui_show(self.after_panel, Image.fromarray(array))
                    elif kind == "done":
                        self._ui_finish("完成！点「保存结果」保存图片。")
                        self.save_btn.configure(state="normal")
                    elif kind == "batch_done":
                        folder, total, failures = data
                        self._ui_finish(f"完成 {total - len(failures)}/{total} 张，已保存到 {folder}")
                        self.save_btn.configure(state="normal")
                        if failures:
                            messagebox.showwarning("部分照片未处理", "\n\n".join(failures[:8]))
                        _open_path(folder)
                    elif kind == "error":
                        self._ui_finish("处理失败")
                        messagebox.showerror("ptoart", data)
            except queue.Empty:
                pass
            self.after(100, self._ui_poll)

        def _ui_finish(self, text):
            self.working = False
            self.progress.stop()
            self.progress.grid_forget()
            self.go_btn.configure(text="开始转换  →")
            self.pick_btn.configure(state="normal")
            self.status.configure(text=text)
            self._ui_update_ready()

        def _ui_save(self):
            if self.result is None:
                return
            opts = self.result_opts
            name = output_name(self.result_src, opts["mode"], opts["style"])
            ext = Path(name).suffix
            path = filedialog.asksaveasfilename(
                title="保存结果", initialfile=name, defaultextension=ext,
                filetypes=(("PNG 图片", "*.png"), ("JPEG 图片", "*.jpg")) if ext == ".png"
                else (("JPEG 图片", "*.jpg"), ("PNG 图片", "*.png")))
            if not path:
                return
            try:
                result = self.result
                if Path(path).suffix.lower() in (".jpg", ".jpeg") and result.shape[2] == 4:
                    alpha = result[..., 3:4].astype(np.float32) / 255
                    result = (result[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
                write_image(path, result, overwrite=True)
                self.status.configure(text=f"已保存：{path}")
            except Exception as exc:
                messagebox.showerror("ptoart", friendly_error(exc))

        def _ui_close_pipelines(self):
            for p in self.pipelines.values():
                try:
                    p.close()
                except Exception:
                    pass
            self.pipelines.clear()

        def _ui_close(self):
            self._ui_close_pipelines()
            self.destroy()

    app = App()
    if screenshot:
        _demo_and_capture(app, screenshot, Image)
        return 0
    app.mainloop()
    return 0


def _open_path(path):
    path = str(path)
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 - opens Explorer on a local folder
    else:
        import subprocess

        subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", path])


def _demo_and_capture(app, target, Image):
    """CI helper: render the window with sample images and save a screenshot."""
    import numpy as np

    y, x = np.mgrid[0:600, 0:480]
    sample = np.dstack([(x * 255 // 480), (y * 255 // 600), np.full_like(x, 180)]).astype(np.uint8)
    app.geometry("1180x760+20+20")
    app.update()
    app._ui_show(app.before, Image.fromarray(sample))
    app._ui_show(app.after_panel, Image.fromarray(255 - sample))
    app.status.configure(text="预览截图（CI）")
    app.lift()
    app.attributes("-topmost", True)
    for _ in range(20):
        app.update()
    from PIL import ImageGrab

    x0, y0 = app.winfo_rootx(), app.winfo_rooty()
    shot = ImageGrab.grab(bbox=(x0, y0, x0 + app.winfo_width(), y0 + app.winfo_height()))
    shot.save(target)
    app._ui_close()
