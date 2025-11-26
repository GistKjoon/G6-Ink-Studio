import sys
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
        self.root.title("G695 Ink Separation Lab")
        self.root.geometry("1380x880")
        self.root.configure(bg="#1c1c1c")

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

        # UI 생성
        self._build_layout()
        self._set_dark_style()

    def _set_dark_style(self) -> None:
        style = ttk.Style(self.root)
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            style.theme_use("clam")
        style.configure("TFrame", background="#1c1c1c")
        style.configure("TLabel", background="#1c1c1c", foreground="#f3f3f3")
        style.configure("TCheckbutton", background="#1c1c1c", foreground="#f3f3f3")
        style.configure("TButton", padding=8)
        style.configure("Big.TLabel", font=("Helvetica", 17, "bold"))
        style.configure("Small.TLabel", font=("Helvetica", 10))
        style.configure("Card.TFrame", background="#252525", relief="ridge", borderwidth=1)
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Muted.TLabel", font=("Helvetica", 10), foreground="#cfcfcf")
        style.configure("Badge.TLabel", font=("Helvetica", 9, "bold"), foreground="#111", background="#4caf50")
        style.configure("Card.TLabelframe", background="#252525", relief="ridge", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background="#252525", foreground="#f3f3f3", font=("Helvetica", 11, "bold"))

    def _build_layout(self) -> None:
        # 좌측 제어 패널 (고정 폭)
        control = ttk.Frame(self.root, padding=16, style="TFrame")
        control.pack(side=tk.LEFT, fill=tk.Y, anchor=tk.N)

        # 헤더
        header = ttk.Frame(control, style="TFrame")
        header.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(header, text="G695 Ink Lab", style="Title.TLabel").pack(anchor=tk.W)
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

        btn_row2 = ttk.Frame(control, style="TFrame")
        btn_row2.pack(fill=tk.X, pady=(4, 12))
        ttk.Button(btn_row2, text="Reset Channels", command=self._enable_all_channels).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(btn_row2, text="Reset Params", command=self._reset_params).pack(
            side=tk.LEFT
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
        status_bar = ttk.Frame(self.root, padding=6, style="Card.TFrame")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar_label = ttk.Label(
            status_bar, text="Waiting for image...", style="Muted.TLabel"
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

    def _process_image(self) -> None:
        if self.original_image is None:
            return
        self.channel_arrays = self.compute_channels(self.original_image)
        self.channel_images = {
            key: self._array_to_image(arr) for key, arr in self.channel_arrays.items()
        }
        self._update_channel_previews()
        self._update_original_preview()
        self.update_composite()
        self.status_label.config(text="Ready")
        self.badge_label.config(text="Ready", style="Badge.TLabel")
        self._set_status("Channel split complete")

    def compute_channels(self, img: Image.Image) -> dict[str, np.ndarray]:
        """RGB -> 6채널 잉크 분리 (흡수 매트릭스 기반 역변환)."""
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
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

        # 선택된 채널만 남긴다
        selected_channels = []
        for ch in INK_CHANNELS:
            key = ch["key"]
            arr = self.channel_arrays[key]
            if not self.channel_enabled[key].get():
                arr = np.zeros_like(arr)
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
        max_w = 700
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
            if info_label is None:
                status_row = ttk.Frame(self.sim_preview_label.master, style="TFrame")
                status_row.pack(fill=tk.X, pady=(6, 0))
                info_lbl = ttk.Label(status_row, textvariable=self.sim_info, style="Muted.TLabel")
                info_lbl.pack(anchor=tk.W)
                self._sim_status_label = info_lbl
            else:
                info_label.configure(textvariable=self.sim_info)

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


def main() -> None:
    root = tk.Tk()
    app = InkSeparationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
