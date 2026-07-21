#!/usr/bin/env python3
"""Generate a vector concept diagram for the PINN magnitude model.

The diagram mirrors the current code path:
GNSS radial displacement + geometry metadata -> CNN/TCN/SE/Transformer ->
STF moment-rate sequence -> physical forward model and Mw consistency losses.

Outputs:
  paper/srl/figures/model_concept.svg
  paper/srl/figures/model_concept.pdf  (when rsvg-convert is available)
  paper/srl/figures/model_concept.png  (preview raster)
"""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "srl" / "figures"
SVG_PATH = OUT_DIR / "model_concept.svg"
PDF_PATH = OUT_DIR / "model_concept.pdf"
PNG_PATH = OUT_DIR / "model_concept.png"

W, H = 1800, 1000

COLORS = {
    "ink": "#1F2933",
    "muted": "#5B6472",
    "grid": "#D7DEE8",
    "soft": "#F7F9FC",
    "data": "#0072B2",
    "meta": "#009E73",
    "physics": "#D55E00",
    "loss": "#CC79A7",
    "stf": "#E69F00",
    "white": "#FFFFFF",
}


def tag(name: str, **attrs: object) -> str:
    parts = []
    for key, val in attrs.items():
        if val is None:
            continue
        key = key.replace("_", "-")
        parts.append(f'{key}="{escape(str(val), {"\"": "&quot;"})}"')
    return f"<{name} {' '.join(parts)}>"


def text(x: float, y: float, body: str, size: int = 24, weight: int | str = 400,
         fill: str = COLORS["ink"], anchor: str = "middle", extra: str = "") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>'
        f"{escape(body)}</text>"
    )


def multiline(x: float, y: float, lines: list[str], size: int = 22,
              fill: str = COLORS["ink"], anchor: str = "middle",
              line_gap: int = 28, weight: int | str = 400) -> str:
    out = []
    for i, line in enumerate(lines):
        out.append(text(x, y + i * line_gap, line, size=size, fill=fill, anchor=anchor, weight=weight))
    return "\n".join(out)


def rounded_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str,
                 rx: float = 14, sw: float = 2.2, dashed: bool = False) -> str:
    dash = "6 6" if dashed else None
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" '
        f'stroke-dasharray="{dash or ""}"/>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = COLORS["ink"],
          sw: float = 3.0, dashed: bool = False, marker: str = "arrow") -> str:
    dash = ' stroke-dasharray="8 8"' if dashed else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{sw:.1f}" fill="none" marker-end="url(#{marker})"{dash}/>'
    )


def curved_arrow(points: list[tuple[float, float]], color: str = COLORS["ink"],
                 sw: float = 3.0, dashed: bool = False, marker: str = "arrow") -> str:
    if len(points) < 4:
        raise ValueError("curved_arrow expects at least 4 points")
    p0, p1, p2, p3 = points[:4]
    d = (
        f"M {p0[0]:.1f},{p0[1]:.1f} "
        f"C {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"
    )
    dash = ' stroke-dasharray="8 8"' if dashed else ""
    return f'<path d="{d}" stroke="{color}" stroke-width="{sw:.1f}" fill="none" marker-end="url(#{marker})"{dash}/>'


def waveform(x: float, y: float, w: float, h: float, color: str,
             cycles: float = 2.5, decay: float = 1.2, sw: float = 3.0) -> str:
    pts = []
    n = 120
    for i in range(n):
        t = i / (n - 1)
        amp = math.exp(-decay * t) * (0.6 + 0.4 * math.sin(2 * math.pi * t))
        val = math.sin(2 * math.pi * cycles * t) * amp
        pts.append((x + t * w, y + h * 0.5 - val * h * 0.38))
    d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    axis = f'<line x1="{x:.1f}" y1="{y + h * 0.5:.1f}" x2="{x + w:.1f}" y2="{y + h * 0.5:.1f}" stroke="#C5CDD8" stroke-width="1.4"/>'
    return axis + f'\n<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="none"/>'


def stf_curve(x: float, y: float, w: float, h: float, color: str, sw: float = 3.2) -> str:
    pts = []
    n = 120
    for i in range(n):
        t = i / (n - 1)
        val = (t ** 2.2) * math.exp(-5.5 * t)
        val += 0.45 * math.exp(-((t - 0.55) / 0.18) ** 2)
        val = min(val * 2.2, 1.0)
        pts.append((x + t * w, y + h - val * h * 0.88))
    d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    axis = (
        f'<line x1="{x:.1f}" y1="{y + h:.1f}" x2="{x + w:.1f}" y2="{y + h:.1f}" stroke="#C5CDD8" stroke-width="1.4"/>'
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + h:.1f}" stroke="#C5CDD8" stroke-width="1.4"/>'
    )
    return axis + f'\n<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="none"/>'


def loss_chip(x: float, y: float, label: str, color: str = COLORS["loss"]) -> str:
    return rounded_rect(x, y, 124, 36, "#FFF7FC", color, rx=18, sw=2) + text(x + 62, y + 24, label, size=16, fill=color, weight=700)


def build_svg() -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M 0 0 L 12 6 L 0 12 z" fill="{COLORS["ink"]}"/></marker>',
        '<marker id="arrow-blue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M 0 0 L 12 6 L 0 12 z" fill="{COLORS["data"]}"/></marker>',
        '<marker id="arrow-orange" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M 0 0 L 12 6 L 0 12 z" fill="{COLORS["physics"]}"/></marker>',
        '<filter id="softShadow" x="-10%" y="-10%" width="120%" height="140%">'
        '<feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#233044" flood-opacity="0.13"/></filter>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="white"/>',
        text(82, 68, "A", size=34, weight=800, anchor="start"),
        text(125, 68, "Physics-informed GNSS magnitude model", size=34, weight=800, anchor="start"),
        text(125, 105, "Radial displacement is encoded as an STF moment-rate sequence; physical operators close the training loop.", size=22, fill=COLORS["muted"], anchor="start"),
    ]

    # High-level bands.
    parts += [
        rounded_rect(70, 150, 430, 720, "#F8FBFF", "#BFD3EA", rx=18, sw=2),
        rounded_rect(540, 150, 695, 720, "#FBFCFE", "#C9D2DE", rx=18, sw=2),
        rounded_rect(1275, 150, 455, 760, "#FFF9F5", "#F2C3A4", rx=18, sw=2),
        text(285, 190, "Station input and preprocessing", size=24, weight=700, fill=COLORS["data"]),
        text(887, 190, "Neural sequence encoder", size=24, weight=700),
        text(1503, 190, "Physics-informed supervision", size=24, weight=700, fill=COLORS["physics"]),
    ]

    # Input panel.
    parts += [
        rounded_rect(105, 230, 360, 150, COLORS["white"], "#A8BED8", rx=14, sw=2),
        text(285, 262, "GNSS displacement", size=22, weight=700, fill=COLORS["data"]),
        waveform(135, 288, 140, 62, COLORS["data"], cycles=2.0),
        waveform(300, 288, 140, 62, "#56B4E9", cycles=2.7),
        text(205, 365, "E(t)", size=18, fill=COLORS["muted"]),
        text(370, 365, "N(t)", size=18, fill=COLORS["muted"]),
        arrow(285, 392, 285, 438, COLORS["data"], marker="arrow-blue"),
        rounded_rect(105, 445, 360, 126, COLORS["white"], "#A8BED8", rx=14, sw=2),
        text(285, 478, "Radial projection", size=22, weight=700, fill=COLORS["data"]),
        text(285, 510, "u_r(t) = E sin(az) + N cos(az)", size=20, fill=COLORS["ink"]),
        text(285, 543, "baseline, filter, fixed T=200", size=18, fill=COLORS["muted"]),
        arrow(285, 584, 285, 630, COLORS["meta"], marker="arrow"),
        rounded_rect(105, 640, 360, 142, COLORS["white"], "#9AD0B5", rx=14, sw=2),
        text(285, 672, "Geometry metadata", size=22, weight=700, fill=COLORS["meta"]),
        multiline(285, 706, ["[log r, sin theta, cos theta,", " sin phi, cos phi]"], size=19, fill=COLORS["ink"], line_gap=25),
        text(285, 768, "phi uses slip direction for radiation loss", size=16, fill=COLORS["muted"]),
    ]

    # Network blocks.
    y0 = 250
    blocks = [
        (585, y0, 170, 105, "Conv1D", ["7-tap kernel", "GELU + GroupNorm"], COLORS["data"]),
        (795, y0, 205, 105, "Residual TCN x6", ["dilations 1..32", "multi-scale release"], COLORS["ink"]),
        (1040, y0, 150, 105, "SE attention", ["channel gates"], COLORS["meta"]),
    ]
    for x, y, bw, bh, title, desc, color in blocks:
        parts.append(rounded_rect(x, y, bw, bh, COLORS["white"], color, rx=14, sw=2.4))
        parts.append(text(x + bw / 2, y + 38, title, size=21, weight=700, fill=color))
        parts.append(multiline(x + bw / 2, y + 69, desc, size=16, fill=COLORS["muted"], line_gap=20))
    parts += [
        arrow(465, 507, 585, 302, COLORS["data"], marker="arrow-blue"),
        arrow(755, 302, 795, 302),
        arrow(1000, 302, 1040, 302),
        rounded_rect(622, 440, 525, 128, COLORS["white"], "#8794A6", rx=14, sw=2.4),
        text(884, 475, "Temporal Transformer encoder x3", size=23, weight=700),
        text(884, 508, "sinusoidal position encoding + metadata embedding", size=18, fill=COLORS["muted"]),
        text(884, 540, "LayerNorm stabilizes sequence features", size=18, fill=COLORS["muted"]),
        curved_arrow([(1115, 356), (1160, 385), (1145, 420), (1080, 440)], COLORS["ink"]),
        curved_arrow([(465, 710), (560, 710), (565, 530), (622, 507)], COLORS["meta"], dashed=True),
        text(520, 690, "add to every time step", size=16, fill=COLORS["meta"], anchor="start"),
        rounded_rect(760, 650, 255, 112, COLORS["white"], "#E3B14A", rx=14, sw=2.4),
        text(887, 684, "Rate head", size=23, weight=700, fill=COLORS["stf"]),
        text(887, 717, "MLP + ReLU/Softplus", size=18, fill=COLORS["muted"]),
        text(887, 747, "output: log1p STF rate", size=18, fill=COLORS["muted"]),
        arrow(884, 568, 884, 650),
    ]

    # Output STF.
    parts += [
        arrow(1015, 706, 1288, 706, COLORS["stf"], marker="arrow-orange"),
        rounded_rect(1295, 628, 380, 157, COLORS["white"], "#E3B14A", rx=14, sw=2.4),
        text(1485, 662, "Predicted source time function", size=22, weight=700, fill=COLORS["stf"]),
        stf_curve(1340, 690, 280, 60, COLORS["stf"]),
        text(1480, 775, "dot M0(t) after log1p decode", size=18, fill=COLORS["muted"]),
    ]

    # Physics panel.
    parts += [
        rounded_rect(1305, 230, 390, 112, COLORS["white"], "#E9A474", rx=14, sw=2.4),
        text(1500, 264, "Elastic forward operator", size=22, weight=700, fill=COLORS["physics"]),
        multiline(1500, 297, ["far-field P/S: dot M0(t-tP/S)", "intermediate P/S: M0(t-tP/S)"], size=17, fill=COLORS["muted"], line_gap=23),
        curved_arrow([(1480, 628), (1445, 540), (1495, 420), (1500, 342)], COLORS["physics"], marker="arrow-orange"),
        rounded_rect(1305, 390, 390, 92, COLORS["white"], "#E9A474", rx=14, sw=2.4),
        text(1500, 425, "Synthetic radial displacement", size=21, weight=700, fill=COLORS["physics"]),
        text(1500, 456, "u_hat_r(t; r, theta, phi)", size=20, fill=COLORS["ink"]),
        arrow(1500, 342, 1500, 390, COLORS["physics"], marker="arrow-orange"),
        rounded_rect(1305, 518, 390, 72, COLORS["white"], "#E9A474", rx=14, sw=2.4),
        text(1500, 548, "Integrate dot M0 -> M0 -> Mw", size=21, weight=700, fill=COLORS["physics"]),
        text(1500, 576, "Mw = 2/3 (log10 M0 - 9.1)", size=18, fill=COLORS["muted"]),
    ]

    # Loss chips and references.
    parts += [
        rounded_rect(1320, 790, 360, 40, "#FFF7FC", COLORS["loss"], rx=20, sw=2),
        text(1500, 817, "Training objective", size=20, weight=700, fill=COLORS["loss"]),
        loss_chip(1310, 846, "L_MSE"),
        loss_chip(1450, 846, "L_synth"),
        loss_chip(1590, 846, "L_mag"),
        loss_chip(1380, 888, "L_shape"),
        loss_chip(1520, 888, "L_nonneg"),
        arrow(1365, 785, 1372, 846, COLORS["loss"], sw=2.2, dashed=True),
        arrow(1500, 482, 1512, 846, COLORS["loss"], sw=2.2, dashed=True),
        arrow(1500, 590, 1652, 846, COLORS["loss"], sw=2.2, dashed=True),
        arrow(1485, 785, 1442, 888, COLORS["loss"], sw=2.2, dashed=True),
        arrow(1440, 785, 1582, 888, COLORS["loss"], sw=2.2, dashed=True),
        text(1372, 842, "reference STF", size=14, fill=COLORS["muted"]),
        text(1512, 842, "observed u_r(t)", size=14, fill=COLORS["muted"]),
        text(1652, 842, "catalog/STF Mw", size=14, fill=COLORS["muted"]),
    ]

    # Small radiation annotation.
    parts += [
        rounded_rect(1338, 165, 326, 38, "#FFF3EB", "#E9A474", rx=19, sw=1.6),
        text(1501, 191, "A_IP, A_IS, A_FP, A_FS from radiation pattern", size=17, fill=COLORS["physics"]),
    ]

    parts += [
        text(84, 960, "Inputs follow src/data/data_loader.py; architecture follows src/models/model.py; losses follow src/training/loss_stf_rate.py.", size=17, fill=COLORS["muted"], anchor="start"),
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")

    converter = shutil.which("rsvg-convert")
    if converter:
        subprocess.run([converter, "-f", "pdf", "-o", str(PDF_PATH), str(SVG_PATH)], check=True)
        subprocess.run([converter, "-w", "2400", "-o", str(PNG_PATH), str(SVG_PATH)], check=True)

    print(SVG_PATH)
    if PDF_PATH.exists():
        print(PDF_PATH)
    if PNG_PATH.exists():
        print(PNG_PATH)


if __name__ == "__main__":
    main()
