"""Generate a Word-safe modeling architecture diagram for the thesis report.

The output is a PNG image because Word renders PNG more reliably than draw.io
or SVG exports in some document themes.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "laporan"
OUT_PATH = OUT_DIR / "gambar_iv4_arsitektur_model_convlstm_word_safe.png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(34, bold=True)
FONT_SUBTITLE = load_font(18)
FONT_BOX_TITLE = load_font(21, bold=True)
FONT_BOX_TEXT = load_font(17)
FONT_NOTE = load_font(16)


def center_text(draw: ImageDraw.ImageDraw, xywh: tuple[int, int, int, int], lines: list[str], fills: list[str]) -> None:
    x, y, w, h = xywh
    line_boxes = []
    total_h = 0
    for index, line in enumerate(lines):
        font = FONT_BOX_TITLE if index == 0 else FONT_BOX_TEXT
        bbox = draw.textbbox((0, 0), line, font=font)
        line_h = bbox[3] - bbox[1]
        line_boxes.append((line, font, line_h, fills[min(index, len(fills) - 1)]))
        total_h += line_h + 6
    total_h -= 6
    current_y = y + (h - total_h) / 2
    for line, font, line_h, fill in line_boxes:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        draw.text((x + (w - line_w) / 2, current_y), line, font=font, fill=fill)
        current_y += line_h + 6


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[int, int, int, int],
    fill: str,
    outline: str,
    text_lines: list[str],
    text_fill: str,
) -> None:
    x, y, w, h = xywh
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline=outline, width=3)
    center_text(draw, xywh, text_lines, [text_fill])


def arrow_head(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    size = 16
    wing = math.radians(28)
    p1 = (ex, ey)
    p2 = (ex - size * math.cos(angle - wing), ey - size * math.sin(angle - wing))
    p3 = (ex - size * math.cos(angle + wing), ey - size * math.sin(angle + wing))
    draw.polygon([p1, p2, p3], fill=fill)


def arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str = "#1B365D") -> None:
    draw.line(points, fill=color, width=5, joint="curve")
    if len(points) >= 2:
        arrow_head(draw, points[-2], points[-1], color)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (1700, 900), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    title = "Arsitektur Model ConvLSTM Historical Risk Patch"
    subtitle = "Model menerima 7 patch historis 160x160 satu kanal dan menghasilkan peta probabilitas risiko hotspot H+1."
    bbox = draw.textbbox((0, 0), title, font=FONT_TITLE)
    draw.text(((1700 - (bbox[2] - bbox[0])) / 2, 35), title, font=FONT_TITLE, fill="#1B365D")
    bbox = draw.textbbox((0, 0), subtitle, font=FONT_SUBTITLE)
    draw.text(((1700 - (bbox[2] - bbox[0])) / 2, 82), subtitle, font=FONT_SUBTITLE, fill="#4B5563")

    boxes = {
        "input": (60, 160, 185, 110, "#D9ECFF", "#1D5F8F", ["Input", "7 x 160 x 160 x 1", "mask hotspot historis"], "#123A57"),
        "convlstm1": (315, 150, 235, 130, "#E3F7E9", "#2D7D46", ["ConvLSTM2D", "32 filter, kernel 3x3", "return sequences"], "#173F24"),
        "bn1": (620, 160, 210, 110, "#F7F7F7", "#6B7280", ["Batch Normalization"], "#374151"),
        "drop1": (895, 160, 175, 110, "#F7F7F7", "#6B7280", ["Dropout", "0,2"], "#374151"),
        "convlstm2": (895, 365, 235, 130, "#E3F7E9", "#2D7D46", ["ConvLSTM2D", "32 filter, kernel 3x3", "return final state"], "#173F24"),
        "bn2": (1200, 375, 210, 110, "#F7F7F7", "#6B7280", ["Batch Normalization"], "#374151"),
        "drop2": (1460, 375, 175, 110, "#F7F7F7", "#6B7280", ["Dropout", "0,2"], "#374151"),
        "conv2d": (570, 635, 235, 120, "#EFE5FF", "#7B4DB8", ["Conv2D", "16 filter, kernel 3x3", "ReLU"], "#2F1B4D"),
        "convout": (900, 635, 235, 120, "#FFE9B8", "#C27B00", ["Conv2D Output", "1 filter, kernel 1x1", "Sigmoid"], "#5A3600"),
        "output": (1230, 635, 235, 120, "#FFE6E2", "#B84A42", ["Output", "160 x 160 x 1", "peta probabilitas H+1"], "#5C1F1B"),
    }

    for x, y, w, h, fill, outline, lines, text_fill in boxes.values():
        rounded_box(draw, (x, y, w, h), fill, outline, lines, text_fill)

    arrow(draw, [(245, 215), (315, 215)])
    arrow(draw, [(550, 215), (620, 215)])
    arrow(draw, [(830, 215), (895, 215)])
    arrow(draw, [(982, 270), (982, 345), (1012, 345), (1012, 365)])
    arrow(draw, [(1130, 430), (1200, 430)])
    arrow(draw, [(1410, 430), (1460, 430)])
    arrow(draw, [(1548, 485), (1548, 565), (688, 565), (688, 635)])
    arrow(draw, [(805, 695), (900, 695)])
    arrow(draw, [(1135, 695), (1230, 695)])

    note = "Panah menunjukkan urutan layer model secara linear sesuai implementasi Sequential."
    bbox = draw.textbbox((0, 0), note, font=FONT_NOTE)
    draw.text(((1700 - (bbox[2] - bbox[0])) / 2, 820), note, font=FONT_NOTE, fill="#4B5563")

    img.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
