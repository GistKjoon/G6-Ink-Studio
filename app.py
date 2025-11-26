import os
import sys
import json
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("TkAgg")

CONFIG_DIR = Path.home() / ".g6_ink_studio"
CONFIG_PATH = CONFIG_DIR / "config.json"

from ai_enhance import refine_channels
from color_model import (
    DEFAULT_ABSORBANCE,
    DEFAULT_PINV,
    compute_pinv,
    channels_from_rgb_linear,
    linear_to_srgb,
    rgb_linear_from_channels,
    srgb_to_linear,
)

# 정의: Canon G695의 6색(예시) 구성. 필요시 라벨/색상 수정.
INK_CHANNELS = [
    {"key": "cyan", "label": "Cyan", "hex": "#00bcd4"},
    {"key": "magenta", "label": "Magenta", "hex": "#e91e63"},
    {"key": "yellow", "label": "Yellow", "hex": "#ffeb3b"},
    {"key": "black", "label": "Black", "hex": "#222222"},
    {"key": "red", "label": "Red", "hex": "#ff1744"},
    {"key": "gray", "label": "Gray", "hex": "#9e9e9e"},
]

CALIB_PRESETS = {
    "Orthogonal CMY (default)": [
        [0.95, 0.00, 0.00, 0.70, 0.30, 0.33],
        [0.00, 0.95, 0.00, 0.70, 0.55, 0.33],
        [0.00, 0.00, 0.95, 0.70, 0.55, 0.33],
    ],
    "Neutral Gray Boost": [
        [0.95, 0.00, 0.00, 0.70, 0.30, 0.45],
        [0.00, 0.95, 0.00, 0.70, 0.55, 0.45],
        [0.00, 0.00, 0.95, 0.70, 0.55, 0.45],
    ],
    "Soft Black": [
        [0.90, 0.00, 0.00, 0.55, 0.25, 0.30],
        [0.00, 0.90, 0.00, 0.55, 0.50, 0.30],
        [0.00, 0.00, 0.90, 0.55, 0.50, 0.30],
    ],
}

PRIMARY = "#4caf50"
BG_MAIN = "#1c1c1c"
CARD = "#252525"
TEXT = "#f3f3f3"
MUTED = "#cfcfcf"
BORDER = "#2f2f2f"


def _hex_to_rgb01(code: str) -> tuple[float, float, float]:
    code = code.lstrip("#")
    r = int(code[0:2], 16) / 255.0
    g = int(code[2:4], 16) / 255.0
    b = int(code[4:6], 16) / 255.0
    return r, g, b


def _hex_to_rgb255(code: str) -> tuple[int, int, int]:
    code = code.lstrip("#")
    return int(code[0:2], 16), int(code[2:4], 16), int(code[4:6], 16)


class InkSeparationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("G6 Ink Studio")
        self.root.geometry("1380x880")
        self.root.configure(bg=BG_MAIN)

        # 상태
        self.original_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.channel_arrays: dict[str, np.ndarray] = {}
        self.channel_images: dict[str, Image.Image] = {}
        self.tk_channel_previews: dict[str, ImageTk.PhotoImage] = {}
        self.tk_composite: ImageTk.PhotoImage | None = None
        self.tk_original_preview: ImageTk.PhotoImage | None = None
        self.channel_enabled: dict[str, tk.BooleanVar] = {
            ch["key"]: tk.BooleanVar(value=True) for ch in INK_CHANNELS
        }
        self.preview_size: tuple[int, int] = (880, 600)

        # 파라미터
        self.red_boost = tk.DoubleVar(value=1.0)
        self.gray_neutral = tk.DoubleVar(value=0.8)
        self.composite_mode = tk.StringVar(value="match")  # match: 원본과 동일한 색, tint: 채널 틴트
        self.absorb_matrix = DEFAULT_ABSORBANCE
        self.absorb_pinv = DEFAULT_PINV
        self.calib_entries: list[list[tk.Entry]] = []
        self.sim_window: tk.Toplevel | None = None
        self.sim_intensity = tk.DoubleVar(value=1.0)
        self.sim_spacing = tk.DoubleVar(value=0.0)
        self.sim_light_x = tk.DoubleVar(value=0.0)
        self.sim_light_y = tk.DoubleVar(value=0.0)
        self.sim_light_z = tk.DoubleVar(value=100.0)
        self.sim_depth_factor = tk.DoubleVar(value=0.0)
        self.sim_light_color = "#ffffff"
        self.sim_preview_label: ttk.Label | None = None
        self.sim_fig: Figure | None = None
        self.sim_ax = None
        self.sim_canvas: FigureCanvasTkAgg | None = None
        self.auto_render = tk.BooleanVar(value=True)
        self.sim_info: tk.StringVar | None = None
        self.preview_quality = tk.StringVar(value="Medium")
        self.calib_preset = tk.StringVar(value="Orthogonal CMY (default)")
        self.precision_mode = tk.StringVar(value="Standard")
        self.contrast_boost = tk.DoubleVar(value=1.0)
        self.preview_scale = tk.StringVar(value="1.0")
        # 튜닝: 화이트밸런스/게인/오프셋/감마
        self.wb_r = tk.DoubleVar(value=1.0)
        self.wb_g = tk.DoubleVar(value=1.0)
        self.wb_b = tk.DoubleVar(value=1.0)
        self.global_gain = tk.DoubleVar(value=1.0)
        self.global_offset = tk.DoubleVar(value=0.0)
        self.view_gamma = tk.DoubleVar(value=1.0)
        self.ai_enabled = tk.BooleanVar(value=False)
        self.ai_strength = tk.DoubleVar(value=0.2)
        self.image_info = tk.StringVar(value="No image")
        self._load_settings()

        # UI 생성
        self._build_layout()
        self._set_dark_style()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _set_dark_style(self) -> None:
        style = ttk.Style(self.root)
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            style.theme_use("clam")

        style.configure("TFrame", background=BG_MAIN)
        style.configure("TLabel", background=BG_MAIN, foreground=TEXT)
        style.configure("TCheckbutton", background=BG_MAIN, foreground=TEXT)
        style.configure("TButton", padding=8, background=CARD, foreground=TEXT)
        style.map(
            "TButton",
            background=[("active", "#3a3a3a")],
            foreground=[("active", "#ffffff")],
        )
        style.configure("Big.TLabel", font=("Helvetica", 17, "bold"), foreground=TEXT, background=BG_MAIN)
        style.configure("Small.TLabel", font=("Helvetica", 10), foreground=TEXT, background=BG_MAIN)
        style.configure("Card.TFrame", background=CARD, relief="ridge", borderwidth=1)
        style.configure("Card.TLabelframe", background=CARD, relief="ridge", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=CARD, foreground=TEXT, font=("Helvetica", 11, "bold"))
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"), foreground=TEXT, background=BG_MAIN)
        style.configure("Muted.TLabel", font=("Helvetica", 10), foreground=MUTED, background=BG_MAIN)
        style.configure("Badge.TLabel", font=("Helvetica", 9, "bold"), foreground="#ffffff", background=PRIMARY)
        style.configure("Status.TFrame", background=CARD, relief="ridge", borderwidth=1)
        style.configure("Status.TLabel", background=CARD, foreground=MUTED, font=("Helvetica", 10))

    def _build_layout(self) -> None:
        # 메뉴바
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image", command=self.open_image, accelerator="Cmd/Ctrl+O")
        filemenu.add_command(label="Export All", command=self.export_all, accelerator="Cmd/Ctrl+E")
        filemenu.add_command(label="Open Exports", command=self._open_exports_folder)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self._on_close, accelerator="Cmd/Ctrl+Q")
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Simulator", command=self.open_simulator, accelerator="Cmd/Ctrl+S")
        menubar.add_cascade(label="Tools", menu=helpmenu)
        self.root.config(menu=menubar)
        # 단축키
        self.root.bind_all("<Command-o>" if sys.platform == "darwin" else "<Control-o>", lambda e: self.open_image())
        self.root.bind_all("<Command-e>" if sys.platform == "darwin" else "<Control-e>", lambda e: self.export_all())
        self.root.bind_all("<Command-s>" if sys.platform == "darwin" else "<Control-s>", lambda e: self.open_simulator())
        self.root.bind_all("<Command-q>" if sys.platform == "darwin" else "<Control-q>", lambda e: self._on_close())

        # 좌측 제어 패널 (고정 폭)
        control = ttk.Frame(self.root, padding=16, style="TFrame")
        control.pack(side=tk.LEFT, fill=tk.Y, anchor=tk.N)

        # 헤더
        header = ttk.Frame(control, style="TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="G6 Ink Studio", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="6채널 분리 · 확인 · 내보내기",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        # 버튼 행
        btn_row = ttk.Frame(control, style="TFrame")
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="Open Image", command=self.open_image).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row, text="Export All", command=self.export_all).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_row, text="Simulator", command=self.open_simulator).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(btn_row, text="Open Exports", command=self._open_exports_folder).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        btn_row2 = ttk.Frame(control, style="TFrame")
        btn_row2.pack(fill=tk.X, pady=(4, 12))
        ttk.Button(btn_row2, text="Reset Channels", command=self._enable_all_channels).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row2, text="Reset Params", command=self._reset_params).pack(
            side=tk.LEFT
        )
        ttk.Button(btn_row2, text="Reset Tuning", command=self._reset_tuning).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        ttk.Separator(control, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(control, text="Red Boost", style="Small.TLabel").pack(
            anchor=tk.W, pady=(12, 2)
        )
        ttk.Scale(
            control,
            from_=0.5,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.red_boost,
            command=self._on_params_change,
        ).pack(fill=tk.X)

        ttk.Label(control, text="Gray Neutral", style="Small.TLabel").pack(
            anchor=tk.W, pady=(12, 2)
        )
        ttk.Scale(
            control,
            from_=0.2,
            to=1.5,
            orient=tk.HORIZONTAL,
            variable=self.gray_neutral,
            command=self._on_params_change,
        ).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            control,
            text="채도를 낮게 잡을수록 Gray 채널에 더 많이 분배됩니다.",
            style="Muted.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(control, text="Preview Quality", style="Small.TLabel").pack(anchor=tk.W, pady=(6, 2))
        quality_row = ttk.Frame(control, style="TFrame")
        quality_row.pack(fill=tk.X, pady=(0, 8))
        ttk.OptionMenu(
            quality_row,
            self.preview_quality,
            self.preview_quality.get(),
            "Low",
            "Medium",
            "High",
        ).pack(side=tk.LEFT, anchor=tk.W)

        ttk.Label(control, text="Preview Scale (performance)", style="Small.TLabel").pack(anchor=tk.W, pady=(6, 2))
        scale_row = ttk.Frame(control, style="TFrame")
        scale_row.pack(fill=tk.X, pady=(0, 8))
        ttk.OptionMenu(
            scale_row,
            self.preview_scale,
            self.preview_scale.get(),
            "1.0",
            "0.75",
            "0.5",
            command=lambda _=None: self._process_image(),
        ).pack(side=tk.LEFT, anchor=tk.W)

        ttk.Label(control, text="Separation Quality", style="Small.TLabel").pack(anchor=tk.W, pady=(6, 2))
        pq_row = ttk.Frame(control, style="TFrame")
        pq_row.pack(fill=tk.X, pady=(0, 4))
        ttk.OptionMenu(
            pq_row,
            self.precision_mode,
            self.precision_mode.get(),
            "Standard",
            "High",
            command=lambda _=None: self._process_image(),
        ).pack(side=tk.LEFT, anchor=tk.W)
        ttk.Label(control, text="Channel Contrast", style="Muted.TLabel").pack(anchor=tk.W, pady=(4, 2))
        ttk.Scale(
            control,
            from_=1.0,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.contrast_boost,
            command=lambda _=None: self._process_image(),
        ).pack(fill=tk.X, pady=(0, 6))

        # 이미지 튜닝 (화이트밸런스/게인/오프셋/감마)
        tuning = ttk.Labelframe(control, text="Image Tuning", padding=8, style="Card.TLabelframe")
        tuning.pack(fill=tk.X, pady=(6, 8))
        for lbl, var in (("WB R", self.wb_r), ("WB G", self.wb_g), ("WB B", self.wb_b)):
            row = ttk.Frame(tuning, style="TFrame")
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=lbl, style="Muted.TLabel", width=6).pack(side=tk.LEFT)
            ttk.Scale(
                row,
                from_=0.5,
                to=1.5,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda _=None: self._process_image(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        row_gain = ttk.Frame(tuning, style="TFrame")
        row_gain.pack(fill=tk.X, pady=2)
        ttk.Label(row_gain, text="Gain", style="Muted.TLabel", width=6).pack(side=tk.LEFT)
        ttk.Scale(
            row_gain,
            from_=0.5,
            to=1.5,
            orient=tk.HORIZONTAL,
            variable=self.global_gain,
            command=lambda _=None: self._process_image(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        row_off = ttk.Frame(tuning, style="TFrame")
        row_off.pack(fill=tk.X, pady=2)
        ttk.Label(row_off, text="Offset", style="Muted.TLabel", width=6).pack(side=tk.LEFT)
        ttk.Scale(
            row_off,
            from_=-0.1,
            to=0.1,
            orient=tk.HORIZONTAL,
            variable=self.global_offset,
            command=lambda _=None: self._process_image(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        row_gamma = ttk.Frame(tuning, style="TFrame")
        row_gamma.pack(fill=tk.X, pady=2)
        ttk.Label(row_gamma, text="Gamma", style="Muted.TLabel", width=6).pack(side=tk.LEFT)
        ttk.Scale(
            row_gamma,
            from_=0.8,
            to=2.2,
            orient=tk.HORIZONTAL,
            variable=self.view_gamma,
            command=lambda _=None: self._process_image(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ai_frame = ttk.Labelframe(control, text="AI Enhance (exp.)", padding=8, style="Card.TLabelframe")
        ai_frame.pack(fill=tk.X, pady=(6, 8))
        ttk.Checkbutton(
            ai_frame, text="Enable AI edge-guided smoothing", variable=self.ai_enabled, command=self._process_image
        ).pack(anchor=tk.W)
        ttk.Label(ai_frame, text="Strength", style="Muted.TLabel").pack(anchor=tk.W, pady=(4, 2))
        ttk.Scale(
            ai_frame,
            from_=0.0,
            to=0.5,
            orient=tk.HORIZONTAL,
            variable=self.ai_strength,
            command=lambda _=None: self._process_image(),
        ).pack(fill=tk.X)

        ttk.Label(control, text="Composite Preview", style="Small.TLabel").pack(
            anchor=tk.W, pady=(8, 4)
        )
        mode_row = ttk.Frame(control, style="TFrame")
        mode_row.pack(anchor=tk.W, pady=(0, 6))
        ttk.Radiobutton(
            mode_row,
            text="원본과 동일 (기본값)",
            value="match",
            variable=self.composite_mode,
            command=self.update_composite,
            style="TRadiobutton",
        ).pack(anchor=tk.W, pady=1)
        ttk.Radiobutton(
            mode_row,
            text="채널 틴트 미리보기",
            value="tint",
            variable=self.composite_mode,
            command=self.update_composite,
            style="TRadiobutton",
        ).pack(anchor=tk.W, pady=1)

        ttk.Label(control, text="Channel Enable", style="Small.TLabel").pack(
            anchor=tk.W, pady=(10, 4)
        )
        for ch in INK_CHANNELS:
            row = ttk.Frame(control, style="TFrame")
            row.pack(fill=tk.X, anchor=tk.W)
            color_chip = tk.Label(row, width=2, height=1, bg=ch["hex"])
            color_chip.pack(side=tk.LEFT, padx=(0, 8), pady=2)
            ttk.Checkbutton(
                row,
                text=ch["label"],
                variable=self.channel_enabled[ch["key"]],
                command=self.update_composite,
            ).pack(side=tk.LEFT, anchor=tk.W)

        ttk.Separator(control, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        info_card = ttk.Frame(control, padding=10, style="Card.TFrame")
        info_card.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(info_card, text="현재 이미지", style="Small.TLabel").pack(anchor=tk.W)
        self.info_label = ttk.Label(info_card, text="No image loaded", wraplength=240)
        self.info_label.pack(anchor=tk.W, pady=(4, 4))
        self.status_label = ttk.Label(
            info_card,
            text="Load an image to start",
            style="Muted.TLabel",
            wraplength=240,
        )
        self.status_label.pack(anchor=tk.W)

        tips = ttk.Frame(control, padding=10, style="Card.TFrame")
        tips.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(tips, text="Quick Tips", style="Small.TLabel").pack(anchor=tk.W)
        ttk.Label(
            tips,
            text="1) Red Boost로 붉은 색 강조\n"
            "2) Gray Neutral로 무채색 감도 조절\n"
            "3) 체크박스로 합성에 포함할 채널 선택\n"
            "4) Export All로 채널/합성 PNG 저장",
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=240,
        ).pack(anchor=tk.W, pady=(4, 0))

        # Calibration section
        calib = ttk.Labelframe(control, text="Calibration (Absorbance)", padding=10, style="Card.TLabelframe")
        calib.pack(fill=tk.X, pady=(8, 8))
        ttk.Label(calib, text="R/G/B 흡수율 (0~1, 3x6)", style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 6))
        grid = ttk.Frame(calib, style="TFrame")
        grid.pack(fill=tk.X)
        channels = ["C", "M", "Y", "K", "Red", "Gray"]
        rows = ["R", "G", "B"]
        header = ttk.Frame(grid, style="TFrame")
        header.grid(row=0, column=1, columnspan=6, sticky="w")
        for idx, ch in enumerate(channels):
            ttk.Label(header, text=ch, style="Muted.TLabel", width=6, anchor=tk.CENTER).grid(row=0, column=idx, padx=1)
        self.calib_entries = []
        for r_idx, row_name in enumerate(rows):
            ttk.Label(grid, text=row_name, style="Muted.TLabel", width=2).grid(row=r_idx + 1, column=0, padx=2, sticky="e")
            row_entries = []
            for c_idx in range(6):
                ent = ttk.Entry(grid, width=6)
                ent.grid(row=r_idx + 1, column=c_idx + 1, padx=1, pady=1)
                row_entries.append(ent)
            self.calib_entries.append(row_entries)
        self._load_matrix_to_entries(self.absorb_matrix)

        calib_btns = ttk.Frame(calib, style="TFrame")
        calib_btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(calib_btns, text="Apply Matrix", command=self._apply_calibration).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(calib_btns, text="Reset Default", command=self._reset_calibration).pack(side=tk.LEFT)
        preset_row = ttk.Frame(calib, style="TFrame")
        preset_row.pack(fill=tk.X, pady=(6, 2))
        ttk.Label(preset_row, text="Preset", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.OptionMenu(
            preset_row,
            self.calib_preset,
            self.calib_preset.get(),
            *CALIB_PRESETS.keys(),
            command=lambda _=None: self._apply_preset(),
        ).pack(side=tk.LEFT)
        save_load = ttk.Frame(calib, style="TFrame")
        save_load.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(save_load, text="Load JSON", command=self._load_calibration_file).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(save_load, text="Save JSON", command=self._save_calibration_file).pack(side=tk.LEFT)

        # 우측: 미리보기 및 채널 그리드
        preview_area = ttk.Frame(self.root, padding=10, style="TFrame")
        preview_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        composite_frame = ttk.Frame(preview_area, padding=8, style="Card.TFrame")
        composite_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        composite_frame.bind("<Configure>", self._on_composite_resize)

        header_bar = ttk.Frame(composite_frame, style="Card.TFrame")
        header_bar.pack(fill=tk.X)
        ttk.Label(
            header_bar, text="Composite Preview", style="Small.TLabel"
        ).pack(side=tk.LEFT, padx=(2, 8))
        self.badge_label = ttk.Label(
            header_bar, text="Ready", style="Badge.TLabel", anchor=tk.CENTER
        )
        self.badge_label.pack(side=tk.LEFT, padx=(4, 0))
        info = ttk.Label(header_bar, textvariable=self.image_info, style="Muted.TLabel")
        info.pack(side=tk.RIGHT, padx=6)

        preview_row = ttk.Frame(composite_frame, style="Card.TFrame")
        preview_row.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left_panel = ttk.Frame(preview_row, padding=8, style="TFrame")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        ttk.Label(left_panel, text="Original", style="Small.TLabel").pack(anchor=tk.W)
        self.original_label = ttk.Label(left_panel, text="이미지를 불러오세요", anchor=tk.CENTER)
        self.original_label.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        right_panel = ttk.Frame(preview_row, padding=8, style="TFrame")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        ttk.Label(right_panel, text="Composite (채널 적용)", style="Small.TLabel").pack(anchor=tk.W)
        self.composite_label = ttk.Label(right_panel, text="", anchor=tk.CENTER)
        self.composite_label.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        channels_frame = ttk.Frame(preview_area, style="TFrame")
        channels_frame.pack(fill=tk.BOTH, expand=False)

        self.channel_labels: dict[str, ttk.Label] = {}
        rows = 2
        cols = 3
        for idx, ch in enumerate(INK_CHANNELS):
            r = idx // cols
            c = idx % cols
            cell = ttk.Frame(channels_frame, padding=6, style="Card.TFrame")
            cell.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            channels_frame.columnconfigure(c, weight=1)
            ttk.Label(cell, text=ch["label"]).pack(anchor=tk.W)
            lbl = ttk.Label(cell, text="")
            lbl.pack(fill=tk.BOTH, expand=True)
            self.channel_labels[ch["key"]] = lbl

        # 상태바
        status_bar = ttk.Frame(self.root, padding=6, style="Status.TFrame")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar_label = ttk.Label(
            status_bar, text="Waiting for image...", style="Status.TLabel"
        )
        self.status_bar_label.pack(side=tk.LEFT)

    def open_image(self) -> None:
        filetypes = [
            ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
        if not path:
            return
        self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"이미지를 열 수 없습니다.\n{exc}")
            return

        self.original_path = path
        self.original_image = img
        self.image_info.set(f"{path.name} / {img.size[0]}x{img.size[1]}")
        self.info_label.config(text=f"{path.name}\n{img.size[0]} x {img.size[1]}")
        self.status_label.config(text="Splitting channels...")
        self._set_status(f"Loaded {path.name} ({img.size[0]}x{img.size[1]})")
        self.root.after(20, self._process_image)

    def _on_params_change(self, _evt: str | None = None) -> None:
        if self.original_image is None:
            return
        self._process_image()

    def _enable_all_channels(self) -> None:
        for var in self.channel_enabled.values():
            var.set(True)
        self.update_composite()
        self._set_status("All channels enabled")

    def _reset_params(self) -> None:
        self.red_boost.set(1.0)
        self.gray_neutral.set(0.8)
        self._process_image()
        self._set_status("Parameters reset")

    def _process_image(self, full: bool = False, silent: bool = False) -> None:
        if self.original_image is None:
            return
        scale = 1.0 if full else float(self.preview_scale.get())
        if scale < 1.0:
            w, h = self.original_image.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            work_img = self.original_image.resize(new_size, Image.LANCZOS)
        else:
            work_img = self.original_image
        self.channel_arrays = self.compute_channels(work_img)
        self.channel_images = {
            key: self._array_to_image(arr) for key, arr in self.channel_arrays.items()
        }
        self._update_channel_previews()
        self._update_original_preview()
        self.update_composite()
        if not silent:
            self.status_label.config(text="Ready")
        self.badge_label.config(text="Ready", style="Badge.TLabel")
        self._set_status("Channel split complete")

    def compute_channels(self, img: Image.Image) -> dict[str, np.ndarray]:
        """RGB -> 6채널 잉크 분리 (흡수 매트릭스 기반 역변환)."""
        dtype = np.float64 if self.precision_mode.get() == "High" else np.float32
        rgb = np.asarray(img.convert("RGB"), dtype=dtype) / 255.0
        # 화이트밸런스, 게인/오프셋 적용
        wb = np.array([self.wb_r.get(), self.wb_g.get(), self.wb_b.get()], dtype=dtype)
        rgb = rgb * wb
        rgb = rgb * float(self.global_gain.get()) + float(self.global_offset.get())
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_lin = srgb_to_linear(rgb)
        channels = channels_from_rgb_linear(
            rgb_lin, absorb=self.absorb_matrix, pinv=self.absorb_pinv
        )

        # 선택적 튜닝
        red_idx = 4  # INK_CHANNELS에서 red의 인덱스
        gray_idx = 5  # INK_CHANNELS에서 gray의 인덱스
        channels[..., red_idx] *= float(self.red_boost.get())
        channels[..., gray_idx] *= float(self.gray_neutral.get())
        channels = np.clip(channels, 0.0, 1.0)

        # 고급 대비 보정
        channels = self._enhance_channel_maps(channels)

        # AI 실험적 스무딩
        if self.ai_enabled.get():
            channels = refine_channels(rgb, channels, strength=float(self.ai_strength.get()), blur_k=5)

        return {
            "cyan": channels[..., 0],
            "magenta": channels[..., 1],
            "yellow": channels[..., 2],
            "black": channels[..., 3],
            "red": channels[..., 4],
            "gray": channels[..., 5],
        }

    def _array_to_image(self, channel: np.ndarray) -> Image.Image:
        channel_u8 = np.clip(channel * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(channel_u8, mode="L")

    def _update_channel_previews(self) -> None:
        if not self.channel_images:
            return
        # 작은 썸네일 생성
        thumb_max = 260
        for ch in INK_CHANNELS:
            key = ch["key"]
            img = self.channel_images[key]
            preview = self._colorize_preview(img, ch["hex"])
            preview.thumbnail((thumb_max, thumb_max))
            tk_img = ImageTk.PhotoImage(preview)
            self.tk_channel_previews[key] = tk_img
            self.channel_labels[key].config(image=tk_img)

    def _colorize_preview(self, gray_img: Image.Image, hex_color: str) -> Image.Image:
        # 회색을 해당 채널 색으로 틴트
        return ImageOps.colorize(gray_img, black="#101010", white=hex_color).convert("RGB")

    def update_composite(self) -> None:
        if not self.channel_arrays:
            return
        if self.original_image is None:
            return

        def _resize_channel(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
            h, w = arr.shape
            if (w, h) == size:
                return arr
            img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
            img = img.resize(size, Image.LANCZOS)
            return np.asarray(img, dtype=np.float32) / 255.0

        # 선택된 채널만 남긴다
        selected_channels = []
        target_size = self.original_image.size
        for ch in INK_CHANNELS:
            key = ch["key"]
            arr = self.channel_arrays[key]
            if not self.channel_enabled[key].get():
                arr = np.zeros_like(arr)
            arr = _resize_channel(arr, target_size)
            selected_channels.append(arr)
        ch_stack = np.stack(selected_channels, axis=-1)

        # composite 계산: match 모드는 재구성(흡수 매트릭스 역변환)으로 원본 근사
        if self.composite_mode.get() == "match":
            rgb_lin = rgb_linear_from_channels(ch_stack, absorb=self.absorb_matrix)
            srgb = linear_to_srgb(rgb_lin)
            comp_img = Image.fromarray((srgb * 255.0).astype(np.uint8), mode="RGB")
        else:
            base = np.zeros(
                (
                    self.original_image.height,
                    self.original_image.width,
                    3,
                ),
                dtype=np.float32,
            )
            for idx, ch in enumerate(INK_CHANNELS):
                tint = _hex_to_rgb01(ch["hex"])
                val = ch_stack[..., idx][..., None]
                base += val * np.array(tint, dtype=np.float32)
            base = np.clip(base, 0.0, 1.0)
            comp_img = Image.fromarray((base * 255.0).astype(np.uint8), mode="RGB")

        disp = comp_img.copy()
        if float(self.view_gamma.get()) != 1.0:
            gamma = float(self.view_gamma.get())
            disp = disp.convert("RGB")
            arr = np.asarray(disp, dtype=np.float32) / 255.0
            arr = np.clip(arr ** (1.0 / gamma), 0.0, 1.0)
            disp = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
        disp.thumbnail(self.preview_size)
        self.tk_composite = ImageTk.PhotoImage(disp)
        self.composite_label.config(image=self.tk_composite, text="")

    def _update_original_preview(self) -> None:
        if self.original_image is None:
            return
        disp = self.original_image.copy()
        disp.thumbnail(self.preview_size)
        self.tk_original_preview = ImageTk.PhotoImage(disp)
        self.original_label.config(image=self.tk_original_preview, text="")

    def export_all(self) -> None:
        if not self.channel_images:
            messagebox.showinfo("Info", "먼저 이미지를 로드하세요.")
            return
        export_dir = Path("exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export 시 풀 해상도로 재계산
        self._process_image(full=True, silent=True)
        stem = self.original_path.stem if self.original_path else "output"
        for ch in INK_CHANNELS:
            key = ch["key"]
            out_path = export_dir / f"{stem}_{key}.png"
            plate = self._render_plate_rgba(key, ch["hex"])
            plate.save(out_path)

        # 선택된 채널 합성도 저장
        comp = self._composite_full_res()
        comp_path = export_dir / f"{stem}_composite.png"
        comp.save(comp_path)
        messagebox.showinfo("Exported", f"저장됨: {export_dir}")
        self._set_status(f"Exported to {export_dir}")
        # 프리뷰 상태 복원
        self._process_image(full=False, silent=True)

    def _composite_full_res(self) -> Image.Image:
        """내보내기용 전체 해상도 합성."""
        # Export 합성본은 분리 채널의 물리적 재구성(흡수 매트릭스 기반)을 사용한다.
        selected_channels = []
        for ch in INK_CHANNELS:
            key = ch["key"]
            arr = self.channel_arrays[key]
            if not self.channel_enabled[key].get():
                arr = np.zeros_like(arr)
            selected_channels.append(arr)
        ch_stack = np.stack(selected_channels, axis=-1)
        rgb_lin = rgb_linear_from_channels(ch_stack, absorb=self.absorb_matrix)
        srgb = linear_to_srgb(rgb_lin)
        return Image.fromarray((srgb * 255.0).astype(np.uint8), mode="RGB")

    def _render_plate_rgba(self, key: str, hex_color: str) -> Image.Image:
        """단일 채널을 투명 배경 + 채널 색상으로 출력."""
        gray = self.channel_images[key]
        rgb = _hex_to_rgb255(hex_color)
        color = Image.new("RGB", gray.size, rgb)
        plate = color.convert("RGBA")
        plate.putalpha(gray)
        return plate

    # --- Calibration helpers ---

    def _load_matrix_to_entries(self, matrix: np.ndarray) -> None:
        for r in range(3):
            for c in range(6):
                self.calib_entries[r][c].delete(0, tk.END)
                self.calib_entries[r][c].insert(0, f"{matrix[r, c]:.3f}")

    def _apply_calibration(self) -> None:
        try:
            vals = np.zeros((3, 6), dtype=np.float32)
            for r in range(3):
                for c in range(6):
                    vals[r, c] = float(self.calib_entries[r][c].get())
            vals = np.clip(vals, 0.0, 1.2)  # 약간의 여유 허용
        except ValueError:
            messagebox.showerror("Error", "숫자만 입력하세요 (0~1).")
            return
        self.absorb_matrix = vals
        self.absorb_pinv = compute_pinv(vals)
        self._process_image()
        self._set_status("Calibration matrix applied")

    def _reset_calibration(self) -> None:
        self.absorb_matrix = DEFAULT_ABSORBANCE
        self.absorb_pinv = DEFAULT_PINV
        self._load_matrix_to_entries(self.absorb_matrix)
        self._process_image()
        self._set_status("Calibration reset to default")

    def _load_calibration_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Load calibration JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        import json  # local import

        try:
            data = json.load(open(path, "r"))
            mat = np.array(data["absorb"], dtype=np.float32)
            if mat.shape != (3, 6):
                raise ValueError("matrix must be 3x6")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"불러오기 실패: {exc}")
            return
        self.absorb_matrix = mat
        self.absorb_pinv = compute_pinv(mat)
        self._load_matrix_to_entries(mat)
        self._process_image()
        self._set_status(f"Loaded calibration: {Path(path).name}")

    def _save_calibration_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save calibration JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile="calibration.json",
        )
        if not path:
            return
        import json  # local import

        vals = np.zeros((3, 6), dtype=np.float32)
        for r in range(3):
            for c in range(6):
                vals[r, c] = float(self.calib_entries[r][c].get() or 0)
        data = {"absorb": vals.tolist()}
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"저장 실패: {exc}")
            return
        self._set_status(f"Saved calibration: {Path(path).name}")

    def _apply_preset(self) -> None:
        name = self.calib_preset.get()
        if name not in CALIB_PRESETS:
            return
        mat = np.array(CALIB_PRESETS[name], dtype=np.float32)
        self.absorb_matrix = mat
        self.absorb_pinv = compute_pinv(mat)
        self._load_matrix_to_entries(mat)
        self._process_image()
        self._set_status(f"Preset applied: {name}")

    # --- Simulator ---
    def open_simulator(self) -> None:
        if not self.channel_images:
            messagebox.showinfo("Info", "먼저 이미지를 로드하세요.")
            return
        if self.sim_window and tk.Toplevel.winfo_exists(self.sim_window):
            self.sim_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Light/Film Simulator")
        win.geometry("1400x900")
        win.configure(bg="#1c1c1c")
        win.protocol("WM_DELETE_WINDOW", self._close_sim_window)
        self.sim_window = win

        panel = ttk.Frame(win, padding=12, style="TFrame")
        panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(panel, text="Light Settings", style="Small.TLabel").pack(anchor=tk.W)
        ttk.Label(panel, text="Intensity", style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Scale(
            panel,
            from_=0.1,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.sim_intensity,
            command=lambda _=None: self._update_simulator(),
        ).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(panel, text="Light Color", style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Button(panel, text=self.sim_light_color, command=self._pick_light_color).pack(
            anchor=tk.W, pady=(0, 6)
        )

        ttk.Label(panel, text="Position (x,y,z)", style="Muted.TLabel").pack(anchor=tk.W)
        for lbl, var in (("X", self.sim_light_x), ("Y", self.sim_light_y), ("Z", self.sim_light_z)):
            row = ttk.Frame(panel, style="TFrame")
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=lbl, width=2, style="Muted.TLabel").pack(side=tk.LEFT)
            ttk.Scale(
                row,
                from_=-300.0 if lbl != "Z" else 10.0,
                to=300.0 if lbl != "Z" else 500.0,
                orient=tk.HORIZONTAL,
                variable=var,
                command=lambda _=None: self._update_simulator(),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(panel, text="Plate Spacing (mm eq.)", style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        ttk.Scale(
            panel,
            from_=0.0,
            to=30.0,
            orient=tk.HORIZONTAL,
            variable=self.sim_spacing,
            command=lambda _=None: self._update_simulator(),
        ).pack(fill=tk.X, pady=(0, 6))

        ttk.Label(panel, text="Depth Parallax", style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Scale(
            panel,
            from_=0.0,
            to=10.0,
            orient=tk.HORIZONTAL,
            variable=self.sim_depth_factor,
            command=lambda _=None: self._update_simulator(auto=True),
        ).pack(fill=tk.X, pady=(0, 6))

        controls_row = ttk.Frame(panel, style="TFrame")
        controls_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(controls_row, text="Render", command=self._update_simulator).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Checkbutton(
            controls_row,
            text="Live",
            variable=self.auto_render,
            command=lambda: self._update_simulator(auto=False),
        ).pack(side=tk.LEFT, padx=(6, 0))

        right = ttk.Frame(win, padding=10, style="TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 3D 뷰
        fig = Figure(figsize=(8, 6), dpi=100, facecolor="#2a2a2a")
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.sim_fig = fig
        self.sim_ax = ax
        self.sim_canvas = canvas

        # 2D 결과 프리뷰
        self.sim_preview_label = ttk.Label(right, text="시뮬레이션 결과", anchor=tk.CENTER)
        self.sim_preview_label.pack(fill=tk.BOTH, expand=True)

        self._update_simulator()

    def _pick_light_color(self) -> None:
        color = colorchooser.askcolor(color=self.sim_light_color, title="Choose light color")
        if not color or not color[1]:
            return
        self.sim_light_color = color[1]
        self._update_simulator()

    def _update_simulator(self, auto: bool = False) -> None:
        if not self.sim_preview_label:
            return
        if auto and not self.auto_render.get():
            return
        if not self.channel_images:
            if not auto:
                messagebox.showinfo("Info", "이미지를 로드하고 채널을 만든 뒤 시뮬레이터를 사용하세요.")
            return
        light_rgb = np.array(_hex_to_rgb01(self.sim_light_color), dtype=np.float32)
        intensity = float(self.sim_intensity.get())
        spacing = float(self.sim_spacing.get())
        depth = float(self.sim_depth_factor.get())
        lx = float(self.sim_light_x.get())
        ly = float(self.sim_light_y.get())
        lz = max(float(self.sim_light_z.get()), 10.0)

        # 준비: 각 채널 RGBA
        plates: list[dict[str, object]] = []
        for idx, ch in enumerate(INK_CHANNELS):
            key = ch["key"]
            if key not in self.channel_images:
                continue
            plate = self._render_plate_rgba(key, ch["hex"])
            plates.append({"img": plate, "hex": ch["hex"], "x": idx})
        if not plates:
            messagebox.showinfo("Info", "채널 이미지가 비어 있습니다. 이미지를 다시 로드하세요.")
            return

        z_spacing = spacing if spacing > 0 else 5.0
        canvas_x = len(plates) * z_spacing + 40

        # 시뮬레이션 크기(프리뷰 축소)
        w, h = self.original_image.size
        quality_map = {"Low": 400, "Medium": 700, "High": 1200}
        max_w = quality_map.get(self.preview_quality.get(), 700)
        if w > max_w:
            scale = max_w / w
            w_new = int(w * scale)
            h_new = int(h * scale)
        else:
            w_new, h_new = w, h

        def shift_image(img_arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
            return np.roll(np.roll(img_arr, dy, axis=0), dx, axis=1)

        canvas = np.ones((h, w, 3), dtype=np.float32) * light_rgb * intensity
        canvas_side = 1 if canvas_x >= lx else -1
        total_count = max(len(plates) - 1, 1)
        for idx, plate_data in enumerate(plates):
            plate_x = plate_data["x"] * z_spacing  # type: ignore[operator]
            # 광원-캔버스 선분 밖에 있으면 스킵
            if (plate_x - lx) * canvas_side < 0 or (canvas_x - plate_x) * canvas_side < 0:
                continue
            plate = plate_data["img"].copy()  # type: ignore[operator]
            hex_color = plate_data["hex"]  # type: ignore[assignment]
            # 패럴럭스: 광원 위치에 따른 약간의 평행 이동
            offset_scale = depth * (idx / total_count)
            dx = int(round((lx / lz) * offset_scale))
            dy = int(round((ly / lz) * offset_scale))
            rgba = np.array(plate.convert("RGBA"), dtype=np.float32) / 255.0
            if dx or dy:
                rgba = shift_image(rgba, dx, dy)
            alpha = rgba[..., 3:4]
            # 거리 감쇠: spacing으로 알파 감소
            atten = np.exp(-spacing * 0.05)
            alpha_eff = np.clip(alpha * atten, 0.0, 1.0)
            tint = np.array(_hex_to_rgb01(str(hex_color)), dtype=np.float32)
            absorb = alpha_eff * tint
            canvas *= (1.0 - absorb)

        canvas = np.clip(canvas, 0.0, 1.0)
        sim_img = Image.fromarray((canvas * 255.0).astype(np.uint8), mode="RGB")
        sim_img = sim_img.resize((w_new, h_new), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(sim_img)
        self.sim_preview_label.config(image=tk_img, text="")
        self.sim_preview_label.image = tk_img
        self._update_3d_view(
            plate_count=len(plates),
            spacing=spacing,
            lx=lx,
            ly=ly,
            lz=lz,
            width=w,
            height=h,
        )
        if self.sim_info is None:
            self.sim_info = tk.StringVar()
            self.sim_info.set(
                f"Light: ({lx:.1f}, {ly:.1f}, {lz:.1f})  Canvas X: {canvas_x:.1f}  Plates: {len(plates)}"
            )
        if self.sim_preview_label.master:
            info_label = getattr(self, "_sim_status_label", None)
            if info_label is None or not info_label.winfo_exists():
                status_row = ttk.Frame(self.sim_preview_label.master, style="TFrame")
                status_row.pack(fill=tk.X, pady=(6, 0))
                info_lbl = ttk.Label(status_row, textvariable=self.sim_info, style="Muted.TLabel")
                info_lbl.pack(anchor=tk.W)
                self._sim_status_label = info_lbl
            else:
                info_label.configure(textvariable=self.sim_info)

    def _close_sim_window(self) -> None:
        if self.sim_window and tk.Toplevel.winfo_exists(self.sim_window):
            try:
                self.sim_window.destroy()
            except Exception:
                pass
        self.sim_window = None
        if hasattr(self, "_sim_status_label"):
            delattr(self, "_sim_status_label")

    def _enhance_channel_maps(self, channels: np.ndarray) -> np.ndarray:
        """채널 대비 보정: 퍼센타일 스트레치 + 선택적 감마."""
        # channels shape: (H, W, 6)
        boost = float(self.contrast_boost.get())
        out = np.empty_like(channels)
        for idx in range(channels.shape[-1]):
            ch = channels[..., idx]
            lo, hi = np.percentile(ch, [0.5, 99.5])
            if hi - lo < 1e-5:
                out[..., idx] = ch
                continue
            ch = (ch - lo) / (hi - lo)
            ch = np.clip(ch, 0.0, 1.0)
            if boost > 1.0:
                ch = np.clip(ch ** (1.0 / boost), 0.0, 1.0)
            out[..., idx] = ch
        return out

    def _update_3d_view(
        self, plate_count: int, spacing: float, lx: float, ly: float, lz: float, width: int, height: int
    ) -> None:
        if not self.sim_ax or not self.sim_canvas:
            return
        ax = self.sim_ax
        ax.clear()
        # 배경/격자: 밝은 회색 배경, 중간 톤 격자
        ax.set_facecolor("#e6e6e6")
        # 최근 Matplotlib에서는 pane 설정이 변경되어 오류가 발생할 수 있으므로 단순화
        try:
            ax.xaxis.pane.set_facecolor((0.9, 0.9, 0.9, 1.0))
            ax.yaxis.pane.set_facecolor((0.9, 0.9, 0.9, 1.0))
            ax.zaxis.pane.set_facecolor((0.9, 0.9, 0.9, 1.0))
        except Exception:
            pass
        ax.grid(True, color="#aaaaaa", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors="#222222")
        ax.xaxis.label.set_color("#222222")
        ax.yaxis.label.set_color("#222222")
        ax.zaxis.label.set_color("#222222")

        # 정규화된 크기 (필름을 YZ 평면에 세움)
        h_norm = 100.0
        w_norm = 100.0 * (width / max(height, 1))
        z_spacing = spacing if spacing > 0 else 5.0

        # 플레이트들 (세로: YZ, X는 깊이)
        for idx, ch in enumerate(INK_CHANNELS):
            x = idx * z_spacing
            color = _hex_to_rgb01(ch["hex"])
            verts = [
                (x, -w_norm / 2, -h_norm / 2),
                (x, w_norm / 2, -h_norm / 2),
                (x, w_norm / 2, h_norm / 2),
                (x, -w_norm / 2, h_norm / 2),
            ]
            poly = Poly3DCollection(
                [verts],
                alpha=0.6,
                facecolor=color,
                edgecolor="#111111",
                linewidths=0.8,
            )
            ax.add_collection3d(poly)

        # 캔버스(출력면): 맨 뒤쪽 X 위치
        canvas_x = plate_count * z_spacing + 40
        canvas_verts = [
            (canvas_x, -w_norm / 2, -h_norm / 2),
            (canvas_x, w_norm / 2, -h_norm / 2),
            (canvas_x, w_norm / 2, h_norm / 2),
            (canvas_x, -w_norm / 2, h_norm / 2),
        ]
        canvas_poly = Poly3DCollection(
            [canvas_verts], alpha=0.4, facecolor="#cccccc", edgecolor="#111111", linewidths=1.0
        )
        ax.add_collection3d(canvas_poly)

        # 광원
        ax.scatter([lx], [ly], [lz], color=self.sim_light_color, s=120, marker="o", label="Light")

        ax.set_xlim(min(lx, -20), canvas_x + 80)
        ax.set_ylim(-w_norm, w_norm)
        ax.set_zlim(-h_norm, h_norm)
        ax.set_xlabel("X (depth)")
        ax.set_ylabel("Y (width)")
        ax.set_zlabel("Z (height)")
        ax.view_init(elev=20, azim=-45)
        ax.set_box_aspect((canvas_x + 80, 2 * w_norm, 2 * h_norm))
        ax.legend(loc="upper right", labelcolor="#222222", facecolor="#eeeeee", edgecolor="#555555")
        self.sim_canvas.draw_idle()

    def _on_composite_resize(self, event: tk.Event) -> None:  # type: ignore[override]
        # 라벨 크기에 맞춰 미리보기 최대 크기 갱신
        usable_w = max(event.width - 40, 120)
        each_w = max(usable_w // 2, 120)
        new_size = (each_w, max(event.height - 120, 120))
        if new_size != self.preview_size:
            self.preview_size = new_size
            self._update_original_preview()
            self.update_composite()

    def _set_status(self, text: str) -> None:
        self.status_bar_label.config(text=text)
        self.root.update_idletasks()

    def _open_exports_folder(self) -> None:
        export_dir = Path("exports").resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "darwin":
                import subprocess

                subprocess.Popen(["open", str(export_dir)])
            elif sys.platform.startswith("win"):
                os.startfile(str(export_dir))  # type: ignore[attr-defined]
            else:
                import subprocess

                subprocess.Popen(["xdg-open", str(export_dir)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"폴더를 열 수 없습니다: {exc}")

    def _load_settings(self) -> None:
        try:
            if CONFIG_PATH.exists():
                data = json.loads(CONFIG_PATH.read_text())
                self.preview_quality.set(data.get("preview_quality", "Medium"))
                self.calib_preset.set(data.get("calib_preset", "Orthogonal CMY (default)"))
                self.preview_scale.set(data.get("preview_scale", "1.0"))
                self.precision_mode.set(data.get("precision_mode", "Standard"))
                self.contrast_boost.set(float(data.get("contrast_boost", 1.0)))
                self.wb_r.set(float(data.get("wb_r", 1.0)))
                self.wb_g.set(float(data.get("wb_g", 1.0)))
                self.wb_b.set(float(data.get("wb_b", 1.0)))
                self.global_gain.set(float(data.get("global_gain", 1.0)))
                self.global_offset.set(float(data.get("global_offset", 0.0)))
                self.view_gamma.set(float(data.get("view_gamma", 1.0)))
                self.ai_enabled.set(bool(data.get("ai_enabled", False)))
                self.ai_strength.set(float(data.get("ai_strength", 0.2)))
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "preview_quality": self.preview_quality.get(),
                "calib_preset": self.calib_preset.get(),
                "preview_scale": self.preview_scale.get(),
                "precision_mode": self.precision_mode.get(),
                "contrast_boost": float(self.contrast_boost.get()),
                "wb_r": float(self.wb_r.get()),
                "wb_g": float(self.wb_g.get()),
                "wb_b": float(self.wb_b.get()),
                "global_gain": float(self.global_gain.get()),
                "global_offset": float(self.global_offset.get()),
                "view_gamma": float(self.view_gamma.get()),
                "ai_enabled": bool(self.ai_enabled.get()),
                "ai_strength": float(self.ai_strength.get()),
            }
            CONFIG_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _reset_tuning(self) -> None:
        self.preview_quality.set("Medium")
        self.preview_scale.set("1.0")
        self.precision_mode.set("Standard")
        self.contrast_boost.set(1.0)
        self.wb_r.set(1.0)
        self.wb_g.set(1.0)
        self.wb_b.set(1.0)
        self.global_gain.set(1.0)
        self.global_offset.set(0.0)
        self.view_gamma.set(1.0)
        self.ai_enabled.set(False)
        self.ai_strength.set(0.2)
        self._process_image()
        self._set_status("Tuning reset to defaults")

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = InkSeparationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
