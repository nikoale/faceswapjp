"""Optional visible watermark (burned into the main output only, never into the matte)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..compositing.blend import from_float, to_float

POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left")


@dataclass
class Watermark:
    text: str = "AI face-swapped"
    position: str = "bottom-right"
    opacity: float = 0.6
    scale: float = 1.0  # relative to a 1080p frame
    font_path: Path | None = None  # TrueType font for non-ASCII text (needs Pillow)

    def __post_init__(self):
        if self.position not in POSITIONS:
            raise ValueError(f"watermark position must be one of {', '.join(POSITIONS)}")
        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int, int]] = {}

    def _render(self, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (color float BGR, alpha float) patches for the text."""
        px = max(12, int(round(36 * self.scale * height / 1080)))
        if self.text.isascii() and self.font_path is None:
            font, fs = cv2.FONT_HERSHEY_SIMPLEX, px / 30.0
            thick = max(1, px // 12)
            (tw, th), base = cv2.getTextSize(self.text, font, fs, thick)
            pad = thick * 2
            alpha = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
            fill = alpha.copy()
            org = (pad, pad + th)
            cv2.putText(alpha, self.text, org, font, fs, 255, thick * 3, cv2.LINE_AA)  # outline
            cv2.putText(fill, self.text, org, font, fs, 255, thick, cv2.LINE_AA)
        else:
            from PIL import Image, ImageDraw, ImageFont

            if self.font_path is None:
                raise ValueError("non-ASCII watermark text needs --watermark-font (a .ttf/.otf file)")
            font = ImageFont.truetype(str(self.font_path), px)
            x0, y0, x1, y1 = font.getbbox(self.text, stroke_width=max(1, px // 12))
            size = (x1 - x0 + 8, y1 - y0 + 8)
            a_img, f_img = Image.new("L", size, 0), Image.new("L", size, 0)
            ImageDraw.Draw(a_img).text((4 - x0, 4 - y0), self.text, font=font, fill=255, stroke_width=max(1, px // 12), stroke_fill=255)
            ImageDraw.Draw(f_img).text((4 - x0, 4 - y0), self.text, font=font, fill=255)
            alpha, fill = np.array(a_img), np.array(f_img)
        color = np.repeat((fill.astype(np.float32) / 255.0)[..., None], 3, axis=2)  # white text, black outline
        return color, alpha.astype(np.float32) / 255.0 * self.opacity

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        key = (h, w)
        if key not in self._cache:
            color, alpha = self._render(h)
            ph, pw = alpha.shape
            margin = int(0.03 * h)
            x = w - pw - margin if "right" in self.position else margin
            y = h - ph - margin if "bottom" in self.position else margin
            self._cache[key] = (color, alpha, max(0, x), max(0, y))
        color, alpha, x, y = self._cache[key]
        ph, pw = min(alpha.shape[0], h - y), min(alpha.shape[1], w - x)
        region = frame[y : y + ph, x : x + pw, :3]
        a = alpha[:ph, :pw, None]
        region[...] = from_float(to_float(region) * (1 - a) + color[:ph, :pw] * a, frame.dtype)
        return frame
