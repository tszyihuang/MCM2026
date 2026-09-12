# -*- coding: utf-8 -*-
"""生成问题3 论文用的全部插图 -> docs/figures/。

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
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.world import generate_case  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#555555"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["figure.dpi"] = 160
plt.rcParams["font.size"] = 9.5

FIG = "docs/figures"
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = "#2f6db3", "#e08a3c", "#4a9d5b", "#c0392b", "#6b7280"


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def convex_hull(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


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


def box(ax, x, y, w, h, text, fc="#eef3fa", ec="#2f6db3", fs=10.0, lw=1.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.5)


def arrow(ax, p, q, color="#444444", rad=0.0, lw=1.3):
    ax.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=1.5, shrinkB=1.5,
                                connectionstyle="arc3,rad=%.2f" % rad))


# --------------------------------------------------------------------------- 问题3
def fig_q3_scene():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.9))
    ax.add_patch(Circle((0, 0), 1800, fill=True, facecolor="#f7f9fc", edgecolor="#333",
                        lw=1.5, label="1800 m"))
    for r, ls in ((1500, "--"), (1000, ":")):
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=C_BLUE, ls=ls, lw=1.1,
                            label="%d m" % r))
    for s in generate_case(9500):
        ax.scatter([s.x], [s.y], s=66, color=C_ORANGE, edgecolor="white", zorder=4)
    ax.scatter([0], [0], marker="*", s=250, color=C_RED, zorder=5)
    ax.annotate("$O$", (0, 0), xytext=(10, -20), textcoords="offset points",
                color=C_RED, fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", handlelength=1.4)
    ax.set_xlim(-2000, 2100), ax.set_ylim(-2050, 2050)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m"), ax.set_ylabel("$y$ / m")

    S1, S2 = (-700.0, -250.0), (520.0, 120.0)
    G = (60.0, 720.0)
    for S, col, tag in ((S1, C_BLUE, "$S_1$"), (S2, C_GREEN, "$S_2$")):
        b = math.degrees(math.atan2(G[1] - S[1], G[0] - S[0])) % 360.0
        for dlt, ls, lw, c in ((0, "-", 1.6, col), (1, "--", 0.9, C_GRAY), (-1, "--", 0.9, C_GRAY)):
            a = math.radians(b + dlt)
            ax2.plot([S[0], S[0] + 3000 * math.cos(a)], [S[1], S[1] + 3000 * math.sin(a)],
                     ls, color=c, lw=lw, zorder=2)
        ax2.scatter([S[0]], [S[1]], s=85, color=col, zorder=5)
        ax2.annotate(tag, S, fontsize=12, xytext=(6, -16), textcoords="offset points", color=col)
    xs = np.linspace(-1900, 1900, 480)
    grid = np.array([(x, y) for x in xs for y in xs if x * x + y * y <= 1900 ** 2])
    keep = []
    for (x, y) in grid:
        ok = True
        for S in (S1, S2):
            b = math.degrees(math.atan2(G[1] - S[1], G[0] - S[0])) % 360.0
            ang = math.degrees(math.atan2(y - S[1], x - S[0])) % 360.0
            if abs(wrap180(ang - b)) > 1.0:
                ok = False
                break
        if ok:
            keep.append((x, y))
    hull = convex_hull(keep)
    ax2.add_patch(Polygon(hull, closed=True, facecolor="#f6c9c4", edgecolor=C_RED, lw=1.2, zorder=3))
    ax2.scatter([G[0]], [G[1]], s=110, marker="*", color=C_RED, zorder=6)
    ax2.annotate("$G$", G, fontsize=12, xytext=(10, 4), textcoords="offset points", color=C_RED)
    cx, cy = float(np.mean([p[0] for p in hull])), float(np.mean([p[1] for p in hull]))
    axin = ax2.inset_axes([0.60, 0.06, 0.37, 0.33])
    for S, col in ((S1, C_BLUE), (S2, C_GREEN)):
        b = math.degrees(math.atan2(G[1] - S[1], G[0] - S[0])) % 360.0
        for dlt, ls, lw, c in ((0, "-", 1.4, col), (1, "--", 0.9, C_GRAY), (-1, "--", 0.9, C_GRAY)):
            a = math.radians(b + dlt)
            axin.plot([S[0], S[0] + 3000 * math.cos(a)], [S[1], S[1] + 3000 * math.sin(a)],
                      ls, color=c, lw=lw)
    axin.add_patch(Polygon(hull, closed=True, facecolor="#f6c9c4", edgecolor=C_RED, lw=1.2))
    axin.scatter([G[0]], [G[1]], s=40, marker="*", color=C_RED, zorder=5)
    axin.set_xlim(cx - 40, cx + 40), axin.set_ylim(cy - 26, cy + 26)
    axin.tick_params(labelsize=6.5)
    ax2.set_xlim(-1900, 1900), ax2.set_ylim(-900, 1900)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$x$ / m"), ax2.set_ylabel("$y$ / m")
    _save(fig, "q3_scene.png")


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


def fig_q3_bow():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    P, E = np.array([0.0, 0.0]), np.array([520.0, 0.0])
    q1 = E + np.array([0.0, 30.0])
    for a in (ax, ax2):
        a.scatter(*P, s=95, color=C_BLUE, zorder=5)
        a.annotate("$P$", P, fontsize=12, xytext=(-4, -20), textcoords="offset points")
        a.scatter(*E, s=130, marker="*", color=C_RED, zorder=5)
        a.annotate("$E$", E, fontsize=12, xytext=(-4, -22), textcoords="offset points")
        a.set_xlim(-80, 620), a.set_ylim(-90, 175)
        a.set_aspect("equal")
        a.set_xlabel("沿视线 / m"), a.set_ylabel("侧向 / m")
    ax.plot([P[0], E[0]], [P[1], E[1]], color=C_GRAY, lw=1.1, ls=":")
    ax.plot([E[0], q1[0], E[0]], [E[1], q1[1], E[1]], color=C_ORANGE, lw=1.9)
    ax.scatter([q1[0]], [q1[1]], s=40, color=C_ORANGE, zorder=5)
    ax.annotate("", xy=(q1[0], q1[1]), xytext=(E[0], E[1]),
                arrowprops=dict(arrowstyle="<->", color=C_ORANGE, lw=1.1))
    ax.annotate("$\\delta$", q1, fontsize=12, color=C_ORANGE, xytext=(8, 4),
                textcoords="offset points")
    half = 0.5 * float(np.linalg.norm(E - P))
    delta = min(max(0.010078 * half * half / 6.0, 10.0), 400.0)
    q2 = P + 0.5 * (E - P) + np.array([0.0, delta])
    ax2.plot([P[0], E[0]], [P[1], E[1]], color=C_GRAY, lw=1.1, ls=":")
    ax2.plot([P[0], q2[0], E[0]], [P[1], q2[1], E[1]], color=C_GREEN, lw=1.9)
    ax2.scatter([q2[0]], [q2[1]], s=55, color=C_GREEN, zorder=5)
    ax2.annotate("$\\delta$", q2, fontsize=12, color=C_GREEN, xytext=(8, 2),
                 textcoords="offset points")
    _save(fig, "q3_bow.png")


def fig_q3_flow():
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    ax.set_xlim(0, 10), ax.set_ylim(0.5, 10.5), ax.axis("off")
    box(ax, 5.0, 9.6, 4.4, 0.9, "① 起点盲扫", "#e8f0fb", C_BLUE)
    box(ax, 8.2, 5.9, 3.5, 1.2, "② 就近清除\nTSP + 弦式侧移", "#e9f6ec", C_GREEN, fs=9.5)
    box(ax, 5.0, 2.5, 4.6, 1.0, "③ 原地补测\n(零移动)", "#fdf1e3", C_ORANGE, fs=9.5)
    box(ax, 1.8, 5.9, 3.5, 1.2, "④ 专程探测\n收尾覆盖", "#eef3fa", C_BLUE, fs=9.5)
    for a, b, rad in (((6.6, 9.3), (7.1, 6.7), -0.30),
                      ((7.4, 5.2), (6.3, 3.2), -0.30),
                      ((3.6, 2.9), (2.5, 5.1), -0.30),
                      ((2.9, 6.7), (3.4, 9.2), -0.30)):
        arrow(ax, a, b, color=C_GRAY, rad=rad, lw=1.6)
    ax.text(5.0, 6.2, "信念更新", ha="center", va="center", fontsize=11.5)
    ax.text(5.0, 1.15, "收尾闸门 → /exit", ha="center", va="center", fontsize=11, color=C_RED)
    arrow(ax, (5.0, 3.05), (5.0, 1.5), color=C_RED, lw=1.5)
    _save(fig, "q3_flow.png")


def fig_q3_phases():
    # 来源: python tools\profile_run.py --module robotdog.solver.sweeper --seeds 9500-11499
    # (进程内口径, 2000 局; 阶段与 profile_run 的划分一一对应)
    stages = ["起点盲扫", "逼近清除", "顺路停车", "原地补测", "收尾覆盖"]
    secs = [119.0, 1666.8, 684.5, 418.7, 197.6]
    mets = [0.0, 7640.0, 2589.0, 0.0, 850.0]
    fig = plt.figure(figsize=(11.4, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(stages))[::-1]
    ax.barh(y, secs, color=[C_GRAY, C_BLUE, C_GREEN, C_ORANGE, C_RED], height=0.6)
    for yy, v in zip(y, secs):
        ax.text(v + 25, yy, "%.0f" % v, va="center", fontsize=8.5)
    ax.set_yticks(y), ax.set_yticklabels(stages, fontsize=9)
    ax.set_xlabel("秒/局"), ax.set_xlim(0, 2050)
    ax2 = fig.add_subplot(gs[1])
    ax2.barh(y, mets, color=[C_GRAY, C_BLUE, C_GREEN, C_ORANGE, C_RED], height=0.6)
    for yy, v in zip(y, mets):
        ax2.text(v + 100, yy, "%.0f" % v, va="center", fontsize=8.5)
    ax2.set_yticks(y), ax2.set_yticklabels([])
    ax2.set_xlabel("米/局"), ax2.set_xlim(0, 9200)
    ax3 = fig.add_subplot(gs[2])
    # 巡回下界取 NN+2-opt 巡回长度在 9500-11499 上的均值 (已知真值路线, 不含定位折返)
    names = ["巡回下界\n(NN+2-opt)", "策略实际"]
    vals = [9033, 11080]
    bars = ax3.bar(names, vals, color=[C_GRAY, C_BLUE], width=0.5)
    for b, v in zip(bars, vals):
        ax3.text(b.get_x() + b.get_width() / 2, v + 180, "%d" % v, ha="center", fontsize=8.5)
    ax3.axhline(9033, color=C_RED, ls="--", lw=1.1)
    ax3.set_ylim(0, 13500), ax3.set_ylabel("米/局")
    ax3.tick_params(axis="x", labelsize=8.5)
    _save(fig, "q3_phases.png")


def fig_q3_results():
    # 取数自 data/reports/pareto_p3.json (标定集 9500-11499, 2000 局, 部署口径)
    with open("data/reports/pareto_p3.json", encoding="utf-8") as fh:
        cs = sorted(json.load(fh)["candidates"], key=lambda c: c["gate"])
    thr = [c["gate"] for c in cs]
    ratio = [c["ratio"] for c in cs]
    sec = [c["avg_clear"] for c in cs]
    xs = np.arange(len(cs))            # 等距刻度: 否则 0~0.2 这段挤在一起看不清
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2),
                                  gridspec_kw={"wspace": 0.38})
    l1, = ax.plot(xs, sec, "-o", color=C_BLUE, lw=1.8, ms=4.5, label="$s$ / 源")
    ax.set_xlabel("收尾门限 $\\lambda$"), ax.set_ylabel("$s$ / 源", color=C_BLUE)
    ax.set_xticks(xs), ax.set_xticklabels(["%.2f" % g for g in thr], fontsize=8.5)
    ax.tick_params(axis="y", labelcolor=C_BLUE)
    kn = thr.index(0.20)
    ax.axvline(kn, color=C_GRAY, ls=":", lw=1.1)
    ax.plot([kn], [sec[kn]], "o", ms=8, mfc="none", mec=C_GRAY)
    axr = ax.twinx()
    l2, = axr.plot(xs, ratio, "-s", color=C_RED, lw=1.6, ms=4, label="清除比例")
    axr.axhline(0.98, color=C_RED, ls="--", lw=1.1)
    axr.set_ylabel("清除比例", color=C_RED), axr.tick_params(axis="y", labelcolor=C_RED)
    axr.set_ylim(0.966, 1.004)
    ax.legend(handles=[l1, l2], fontsize=9, frameon=False, loc="upper right")

    vers = ["问题3 策略"]
    dep = [json.load(open("data/reports/q3_deploy.json", encoding="utf-8"))["summary"]["avg_clear_mean"]]
    ins = [json.load(open("data/reports/q3_insample.json", encoding="utf-8"))["results"]
           ["robotdog.solver.sweeper"]["summary"]["avg_clear_mean"]]
    x = np.arange(1)
    w = 0.30
    b1 = ax2.bar(x - w / 2, dep, w, color=C_BLUE, label="部署口径")
    b2 = ax2.bar(x + w / 2, ins, w, color=C_ORANGE, label="进程内口径")
    for bars, vals in ((b1, dep), (b2, ins)):
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 6, "%.1f" % v, ha="center", fontsize=8.5)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_xticks(x), ax2.set_xticklabels(vers)
    ax2.set_ylim(0, 360), ax2.set_ylabel("$s$ / 源")
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)
    _save(fig, "q3_results.png")


# --------------------------------------------------------------------------- 帕累托前沿
def _pareto_data(problem):
    with open("data/reports/pareto_p%d.json" % problem, encoding="utf-8") as fh:
        hold = json.load(fh)
    with open("data/reports/pareto_p%d_gen.json" % problem, encoding="utf-8") as fh:
        gen = json.load(fh)
    return hold, gen


def fig_pareto(problem, fname):
    hold, gen = _pareto_data(problem)
    pal = [C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY]
    labels = []
    for c in hold["candidates"]:
        if c["label"] not in labels:
            labels.append(c["label"])
    color_of = {lab: pal[i % len(pal)] for i, lab in enumerate(labels)}
    front = hold["frontier"]
    fset = set(front)
    knee = hold["knee"]
    knee_c = hold["candidates"][knee]
    best = max(front, key=lambda i: (hold["candidates"][i]["ratio"],
                                     -hold["candidates"][i]["avg_clear"]))
    best_c = hold["candidates"][best]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    other = [hold["candidates"][i] for i in range(len(hold["candidates"])) if i not in fset]
    if other:
        ax.scatter([c["avg_clear"] for c in other], [c["ratio"] for c in other],
                   s=26, marker="x", color="#9aa0a6", zorder=3, label="被支配点")
    for lab in labels:
        pts = [hold["candidates"][i] for i in front if hold["candidates"][i]["label"] == lab]
        if pts:
            ax.plot([p["avg_clear"] for p in pts], [p["ratio"] for p in pts],
                    "-o", color=color_of[lab], lw=1.9, ms=5.5, zorder=4, label="前沿")
    ax.scatter([knee_c["avg_clear"]], [knee_c["ratio"]], s=190, marker="*",
               color=C_RED, zorder=6, label="膝点")
    if best != knee:
        ax.scatter([best_c["avg_clear"]], [best_c["ratio"]], s=130, marker="D",
                   facecolor="none", edgecolor=C_GREEN, lw=1.8, zorder=6, label="最高清除比例")
    ax.axhline(0.98, color=C_RED, ls="--", lw=1.2)
    ax.set_xlabel("平均定位清除时间 / ($s\\cdot$源$^{-1}$)")
    ax.set_ylabel("清除比例")
    ax.legend(fontsize=9, frameon=False, loc="best")
    ax.grid(alpha=0.2)

    for tag, data, ls, mk, col in (("标定集", hold, "-", "o", C_BLUE),
                                   ("独立测试集", gen, "--", "s", C_ORANGE)):
        fr = data["frontier"]
        ax2.plot([data["candidates"][i]["avg_clear"] for i in fr],
                 [data["candidates"][i]["ratio"] for i in fr],
                 ls, marker=mk, color=col, lw=1.9, ms=5.5, label=tag)
    ax2.axhline(0.98, color=C_RED, ls="--", lw=1.2)
    ax2.set_xlabel("平均定位清除时间 / ($s\\cdot$源$^{-1}$)")
    ax2.set_ylabel("清除比例")
    ax2.legend(fontsize=9, frameon=False, loc="lower right")
    ax2.grid(alpha=0.2)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, fname)


def main():
    fig_q3_scene()
    fig_q3_nn()
    fig_q3_bow()
    fig_q3_flow()
    fig_q3_phases()
    fig_q3_results()
    fig_pareto(3, "q3_pareto.png")


if __name__ == "__main__":
    main()
