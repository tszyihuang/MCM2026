# -*- coding: utf-8 -*-
"""生成问题3 论文正文使用的插图 -> docs/figures/。

当前生成 4 张, 与论文图 1~图 4 一一对应:
  q3_posterior      源位置后验的两种表示
  q3_nn             清除后原地补测的最近邻距离证据
  q3_results        两种收尾口径在 (平均定位清除时间, 清除比例) 平面上的折中
  q3_pareto_forest  两个口径对收尾门限 lambda 的灵敏度

图面规约: **只保留坐标轴、刻度、图例与数值标签**, 不写说明性文字、注释框或图上标题;
所有解释一律放进论文正文与图注。数据来自 solver 的确定性评测
(标定集 seed 9500-11499, 独立测试集 8000-8999) 或其解析式。
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.world import generate_case  # noqa: E402
from tools import figstyle as FS  # noqa: E402

FS.apply_style()

FIG = "docs/figures"
#: 配色见 tools/figstyle.py —— 速度优先=深蓝, 保障模式=中粉, 辅助档=中蓝,
#: 高亮/阈值=深玫, 中性参考=灰蓝
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = (
    FS.BLUE, FS.BLUE_LIGHT, FS.PINK, FS.ROSE, FS.SLATE)


def _save(fig, name):
    FS.save(fig, os.path.join(FIG, name))


def nn_stats(seeds):
    d1, d2, d3 = [], [], []
    for s in seeds:
        pts = [(x.x, x.y) for x in generate_case(s)]
        for i, p in enumerate(pts):
            dd = sorted(math.dist(p, q) for j, q in enumerate(pts) if j != i)
            d1.append(dd[0])
            d2.append(dd[1])
            d3.append(dd[2])
    return d1, d2, d3


# --------------------------------------------------------------------------- 问题3
def fig_q3_posterior():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # (1) 尚无示向度: 一次无信号观测后的占用网格
    step = 100.0
    grid = np.arange(-1800.0, 1800.0 + step, step)
    xx, yy = np.meshgrid(grid, grid)
    dist = np.hypot(xx, yy)
    weight = np.clip((dist - 1000.0) / 500.0, 0.0, 1.0)
    weight[dist > 1800.0] = np.nan
    cmap = FS.CMAP_BLUE.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    im = ax.imshow(weight, origin="lower", extent=(-1850, 1850, -1850, 1850),
                   cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    for r, ls, tag in ((1800, "-", "1800 m"), (1500, "--", "1500 m"),
                       (1000, ":", "1000 m")):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=ls, lw=1.0,
                            edgecolor=C_GRAY if r == 1800 else C_BLUE,
                            label=tag))
    ax.scatter([0], [0], marker="*", s=130, color=C_RED, zorder=5,
               label="检测点 $q$")
    ax.set_xlim(-1900, 1900)
    ax.set_ylim(-1900, 1900)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(fontsize=7.8, frameon=True, facecolor="white", framealpha=0.88,
              edgecolor="none", loc="upper left", handlelength=1.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("后验权重", fontsize=8)

    # (2) 仅有一条示向度: 128 个径向样本, 点大小与密度成正比
    theta = math.radians(38.0)
    radius = np.linspace(0.0, 1800.0, 128)
    density = radius * np.clip((1500.0 - radius) / 500.0, 0.0, 1.0)
    density = density / max(float(density.max()), 1e-12)
    rx = radius * math.cos(theta)
    ry = radius * math.sin(theta)
    ax2.add_patch(Wedge((0, 0), 1500.0, 37.0, 39.0,
                        facecolor=C_ORANGE, edgecolor="none", alpha=0.18))
    ax2.plot([0, 1500.0 * math.cos(theta)], [0, 1500.0 * math.sin(theta)],
             color=C_GRAY, lw=1.0, ls=":")
    ax2.scatter(rx, ry, s=5.0 + 70.0 * density, c=density, cmap=FS.CMAP_BLUE,
                edgecolor="none", zorder=4)
    ax2.scatter([0], [0], marker="*", s=130, color=C_RED, zorder=5)
    ax2.set_xlim(-150, 1750)
    ax2.set_ylim(-250, 1300)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$x$ / m")
    ax2.set_ylabel("$y$ / m")

    fig.subplots_adjust(wspace=0.32)
    _save(fig, "q3_posterior.png")


def fig_q3_nn():
    d1, d2, d3 = nn_stats(range(9500, 11500))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    bins = np.arange(0, 1900, 90)
    for arr, lab, col in ((d1, "1-NN", C_BLUE), (d2, "2-NN", C_GREEN), (d3, "3-NN", C_ORANGE)):
        ax.hist(arr, bins=bins, histtype="step", lw=1.8, label=lab, color=col)
    ax.axvline(1000, color=C_RED, ls="--", lw=1.4)
    ax.text(1030, ax.get_ylim()[1] * 0.55, "1000 m", color=C_RED, fontsize=9)
    ax.set_xlabel("邻居距离 / m"), ax.set_ylabel("频数")
    ax.legend(fontsize=9.5, frameon=False)
    p1 = float(np.mean(np.array(d1) <= 1000))
    p2 = float(np.mean(np.array(d2) <= 1000))
    p3 = float(np.mean(np.array(d3) <= 1000))
    vals = [p1, p2, p3]
    ax2.bar(["$\\geq$1", "$\\geq$2", "$\\geq$3"], vals,
            color=[C_BLUE, C_GREEN, C_ORANGE], width=0.55)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.015, "%.1f%%" % (100 * v), ha="center", fontsize=9.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("1000 m 内的其他源个数"), ax2.set_ylabel("概率")
    _save(fig, "q3_nn.png")


def fig_q3_results():
    """两族收尾口径在 (平均定位清除时间, 清除比例) 平面上的折中 (标定集 2000 局, 部署口径)。

    数据取自 data/reports/pareto_p3.json (保障模式) 与 pareto_p3_speed.json (速度优先)。
    """
    def load(name):
        with open("data/reports/%s" % name, encoding="utf-8") as fh:
            cs = sorted(json.load(fh)["candidates"], key=lambda c: c["gate"])
        return ([c["gate"] for c in cs], [c["avg_clear"] for c in cs],
                [c["ratio"] for c in cs])

    g_on, t_on, r_on = load("pareto_p3.json")
    g_off, t_off, r_off = load("pareto_p3_speed.json")
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    ax.plot(t_off, r_off, "-o", color=C_BLUE, lw=1.8, ms=4.5,
            label="速度优先（关闭覆盖收尾）")
    for g, x, y in zip(g_off, t_off, r_off):
        if g in (0.05, 0.20, 0.60):
            ax.annotate("$\\lambda$=%.2f" % g, (x, y), textcoords="offset points",
                        xytext=(4, -12), fontsize=7.8, color=C_BLUE)
    star = g_on.index(0.0)
    t_star, r_star = t_on[star], r_on[star]
    #: 保障模式的 9 个门限点在本平面上几乎重合 (396.4~405.2 s/源, 比例恒为 1.0000),
    #: 故只画采纳点: 从速度优先同一门限 ($\lambda$=0) 的水平虚线即它的时间代价。
    #: 保障模式族对 $\lambda$ 的灵敏度见 q3_pareto_forest.png。
    ax.plot([t_off[0], t_star], [r_off[0], r_star], ls="--", lw=1.4, color=C_GREEN,
            label="保障模式的代价（$+%.1f\\%%$ 时间）"
                  % (100.0 * (t_star / t_off[0] - 1.0)))
    ax.plot([t_star], [r_star], "*", ms=15, color=C_RED, mec="white",
            mew=0.7, zorder=5, label="采纳运行点（$\\lambda$=0.00）")
    ax.set_xlabel("平均定位清除时间 / ($s\\cdot$源$^{-1}$)")
    ax.set_ylabel("清除比例")
    ax.set_ylim(0.968, 1.005)
    #: x 轴需同时容纳速度优先族 (240.6~336.4 s/源) 与采纳点 (保障模式 396.4 s/源)
    ax.set_xlim(230, 415)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    _save(fig, "q3_results.png")


def fig_q3_pareto_forest():
    """收尾门限 lambda 的灵敏度: 两个口径的清除比例与平均时间 (标定集, 部署口径)。"""
    def load(name):
        with open("data/reports/%s" % name, encoding="utf-8") as fh:
            cs = sorted(json.load(fh)["candidates"], key=lambda c: c["gate"])
        return ([c["gate"] for c in cs], [c["ratio"] for c in cs],
                [c["avg_clear"] for c in cs])

    g_on, r_on, t_on = load("pareto_p3.json")
    g_off, r_off, t_off = load("pareto_p3_speed.json")
    xs = np.arange(len(g_on))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                  gridspec_kw={"wspace": 0.26})
    ax.plot(xs, r_off, "-o", color=C_BLUE, lw=1.7, ms=4.2, label="速度优先")
    ax.plot(xs, r_on, "-s", color=C_GREEN, lw=1.7, ms=3.8, label="保障模式")
    ax.set_ylim(0.968, 1.006)
    ax.set_ylabel("清除比例")
    ax2.plot(xs, t_off, "-o", color=C_BLUE, lw=1.7, ms=4.2, label="速度优先")
    ax2.plot(xs, t_on, "-s", color=C_GREEN, lw=1.7, ms=3.8, label="保障模式")
    ax2.set_ylabel("平均定位清除时间 / ($s\\cdot$源$^{-1}$)")
    for a in (ax, ax2):
        a.set_xticks(xs)
        a.set_xticklabels(["%.2f" % g for g in g_on], fontsize=8.2)
        a.set_xlabel("收尾门限 $\\lambda$")
        a.grid(alpha=0.22)
    ax.legend(fontsize=8.8, frameon=False, loc="lower right")
    ax2.legend(fontsize=8.8, frameon=False, loc="upper right")
    _save(fig, "q3_pareto_forest.png")


def main():
    fig_q3_posterior()
    fig_q3_nn()
    fig_q3_results()
    fig_q3_pareto_forest()


if __name__ == "__main__":
    main()
