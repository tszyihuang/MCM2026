# -*- coding: utf-8 -*-
"""论文插图统一配色 (科研蓝-粉方案)。

色板取自 8 色卡, 按"冷 / 暖"两条明度梯级组织: 冷色表示低值或主序列, 暖色表示
高值或对照序列, 最深档用于高亮 (阈值线、采纳运行点、外点)。8 个色值依次是
``#4E79A7`` (深蓝) / ``#83B4D9`` (中蓝) / ``#DCEAF6`` (浅蓝) / ``#B7C4D9`` (灰蓝) /
``#F4C2C2`` (浅粉) / ``#E38B8B`` (中粉) / ``#CC5A6A`` (深玫) / ``#FBE9E9`` (极浅粉)。

坐标轴、刻度与文字沿用中性深灰 (``#555555`` 边框 / ``#333333`` 文字): 色卡本身
不含文字色, 深灰可保证黑白打印与投影时的可读性。``SLATE_TEXT`` 是灰蓝的加深档,
只在原本用灰色画文字或散点的地方替代 ``SLATE``, 避免浅灰在白底上看不清。

三份出图脚本 (``make_paper_figures.py`` / ``probes/q12_make_figures.py`` /
``probes/q4_make_figures.py``) 的 rcParams、配色与落盘逻辑统一由本模块提供,
以后调整配色只需改这一处。
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- 色板
PALETTE = ("#4E79A7", "#83B4D9", "#DCEAF6", "#B7C4D9",
           "#F4C2C2", "#E38B8B", "#CC5A6A", "#FBE9E9")

BLUE = "#4E79A7"          # 深蓝: 主序列
BLUE_LIGHT = "#83B4D9"    # 中蓝: 次要序列
BLUE_PALE = "#DCEAF6"     # 浅蓝: 浅填充
SLATE = "#B7C4D9"         # 灰蓝: 中性参考线
PINK_LIGHT = "#F4C2C2"    # 浅粉: 中间档
PINK = "#E38B8B"          # 中粉: 对照序列
ROSE = "#CC5A6A"          # 深玫: 高亮
PINK_PALE = "#FBE9E9"     # 极浅粉: 背景

SLATE_TEXT = "#7C8DA6"    # 灰蓝加深档 (仅用于文字/散点)
EDGE = "#555555"          # 坐标轴边框

# --------------------------------------------------------------------------- 色标
#: 单色序列 (越深值越大): 原 "Blues" / "viridis" 的替代
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "mcm_blue", ["#FBFDFF", BLUE_PALE, BLUE_LIGHT, BLUE])
#: 暖色序列: 原 "YlOrRd" 的替代 (最浅档不取纯白, 保证低值区仍看得出暖色调)
CMAP_PINK = LinearSegmentedColormap.from_list(
    "mcm_pink", [PINK_PALE, PINK_LIGHT, PINK, ROSE])
#: 双向色标 (冷-暖), 供相关系数一类正负含义相反的图使用
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "mcm_diverging", [BLUE, BLUE_LIGHT, BLUE_PALE, "#FFFFFF",
                      PINK_PALE, PINK_LIGHT, PINK, ROSE])
FONT_SANS = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
             "Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]


def apply_style(dpi: int = 160, font_size: float = 9.5) -> None:
    """套用三份出图脚本共用的 rcParams (中文字体、字号、边框、白底)。"""
    matplotlib.use("Agg")
    plt.rcParams["font.sans-serif"] = FONT_SANS
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = EDGE
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["font.size"] = font_size
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def save(fig, path: str) -> str:
    """按统一规约落盘 (白底 + 紧凑边界) 并关闭画布, 返回路径。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)
    return path
