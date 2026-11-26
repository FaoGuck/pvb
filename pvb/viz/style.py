# 来自 S2R 505b298 的 s2r/viz/style.py，2026-11-26 复制（机械改名：S2R_→PRJ_、包名 s2r→项目名）
"""统一作图风格（总纲 8.1）：所有图从这里取字体、尺寸、颜色，脚本里不各自设。

颜色按实体固定（全课题不换）：经典特征灰；REVE / LaBraM / CBraMod 取参考配色的第 1–3 位（已校验的类别色）；
随机初始化 = 对应模型颜色的空心标记 / 浅色虚线；EEGNet 黑；反应历史（上一个 MEP）取第 4 位黄色——
黄色在白底上对比度偏低，所以凡用到它都配文字标签。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
WIDTH_1COL = 85 * MM
WIDTH_2COL = 175 * MM

INK = "#0b0b0b"          # 主文字
INK2 = "#52514e"         # 次文字
MUTED = "#898781"        # 坐标轴标签
GRID = "#e1e0d9"         # 网格线
AXIS = "#c3c2b7"         # 坐标轴 / 基线

COLORS = {
    "classical": "#6f6e69",
    "reve": "#2a78d6",
    "labram": "#eb6834",
    "cbramod": "#1baf7a",
    "eegnet": "#0b0b0b",
    "history": "#eda100",
}
NAMES = {"classical": "Classical", "reve": "REVE", "labram": "LaBraM", "cbramod": "CBraMod",
         "eegnet": "EEGNet", "history": "Prev. MEP"}


def lighten(hex_color: str, amount: float = 0.55) -> str:
    """与白色混合（随机初始化用）。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(round(c + (255 - c) * amount))  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, out_dir: Path, name: str, data=None):
    """同时存 pdf（矢量）和 png（300 dpi），以及同名 csv（图中画出的数值）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    if data is not None:
        data.to_csv(out_dir / f"{name}.csv", index=False)
    plt.close(fig)
