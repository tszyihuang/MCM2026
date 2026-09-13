# -*- coding: utf-8 -*-
"""生成问题4 论文插图 -> docs/figures/q4_*.png。

五张图与 ``docs/问题4论文.md`` 的 图 1..图 5 一一对应：

1. ``q4_cover_gap``      图 1：贴边外向定向源的可检测弧带（内环全落空，只有外环站落得进去）
2. ``q4_cover_frontier`` 图 2：外环点数与最大角隙（180° 是漏检阈值，取自 (4) 式）：
   左 = 发布族（内环 970 m×6、外环在边界 1800 m），右 = 上一版族（内环 ×8、外环 1850 m）
3. ``q4_scan_sets``      图 3：发布站点集与骨架巡游路线（点数随 ``ring_spec`` 变）
4. ``q4_phases``         图 4：虚拟时间去向：分项（移动/检测/切换/清除）+ 分阶段（巡游/收尾/兜底）
5. ``q4_compare``        图 5：两版规划器对照（同一批 200 局）

图面规约与 ``tools/make_paper_figures.py`` 一致: 只保留坐标轴、刻度、图例与数值标签,
不写说明性文字或图上标题 (解释放在论文图注里)。
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.consts import ARENA_RADIUS_M                      # noqa: E402
from robotdog.solver import sweeper4 as S                       # noqa: E402
from tools import figstyle as FS                                # noqa: E402
from tools.probes.q4_opt1_layout import grid_audit              # noqa: E402

FS.apply_style()

FIG = os.path.join("docs", "figures")
#: 配色见 tools/figstyle.py —— 主色=深蓝, 对照/次类=中粉, 第三档=中蓝,
#: 高亮=深玫, 中性(旧方案、参考线)=灰蓝
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = (
    FS.BLUE, FS.PINK, FS.BLUE_LIGHT, FS.ROSE, FS.SLATE)
C_SLATE_TEXT = FS.SLATE_TEXT
ARENA = ARENA_RADIUS_M


def _save(fig, name: str) -> None:
    FS.save(fig, os.path.join(FIG, name))


def _stations() -> List[Tuple[float, float]]:
    """发布配置的站点坐标（随 ``PARAMS['ring_spec']`` 变）。"""
    return [(float(r[0]), float(r[1])) for r in S._layout_rows(S.PARAMS)]


def _ring_spec() -> Tuple[Tuple[float, int], ...]:
    """发布 ``ring_spec``（(原点, 内环, 外环) 三档）。"""
    return tuple((float(r), int(n)) for r, n in S.PARAMS["ring_spec"])


def _ring_points(r: float, n: int, ring_idx: int) -> List[Tuple[float, float]]:
    """按 ``_layout_rows`` 的相位约定展开一圈站点（``ring_idx`` 即 spec 下标）。"""
    off = float(S.PARAMS.get("ring_phase", 0.0)) * ring_idx * math.pi / 3.0
    return [(r * math.cos(off + 2.0 * math.pi * k / n),
             r * math.sin(off + 2.0 * math.pi * k / n)) for k in range(n)]


def _family(r_in: float, n_in: int, r_out: float, n_out: int) -> List[Tuple[float, float]]:
    """``原点 + 内环 + 外环`` 的站点集（相位与发布配置一致）。"""
    return ([(0.0, 0.0)] + _ring_points(r_in, n_in, 1) + _ring_points(r_out, n_out, 2))


def _best_tour(pts: Sequence[Tuple[float, float]]):
    """全起点（含反向扰动）2-opt/Or-opt 最优开链巡游；返回 (路线, 长度)。"""
    xy = [(float(p[0]), float(p[1])) for p in pts]
    best = None
    for s0 in range(len(xy)):
        rem = [i for i in range(len(xy)) if i != s0]
        cur, order = s0, [s0]
        while rem:
            k = min(rem, key=lambda i: math.dist(xy[i], xy[cur]))
            rem.remove(k)
            order.append(k)
            cur = k
        for cand in (order, list(reversed(order))):
            o = S._ls_order(cand, xy, (0.0, 0.0), 3)
            L = S._order_len(o, xy, (0.0, 0.0))
            if best is None or L < best[0]:
                best = (L, o)
    return [xy[i] for i in best[1]], best[0]


# ------------------------------------------------------------------ 1. 覆盖缺口

def fig_cover_gap() -> None:
    """贴边外向定向源的可检测弧带（真实案例 seed 9504 ch13：只有 1 个外环站看得见）。"""
    sx, sy, R, psi = -1620.7, -604.6, 1096.4, 187.7
    pts = _stations()
    r_out = max(_ring_spec(), key=lambda rn: rn[0])[0]
    inside = [(x, y) for (x, y) in pts if math.hypot(x, y) < 1500.0]
    outside = [(x, y) for (x, y) in pts if math.hypot(x, y) >= 1500.0]
    seen = [(x, y) for (x, y) in pts
            if math.hypot(x - sx, y - sy) <= R
            and math.cos(math.radians(math.degrees(math.atan2(y - sy, x - sx)) - psi)) >= 0.0]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.add_patch(Circle((0, 0), ARENA, fill=False, ec="#333333", lw=1.4))
    ax.add_patch(Circle((sx, sy), R, fill=False, ec=C_GRAY, ls=":", lw=1.2))
    th0, th1 = psi - 90.0, psi + 90.0
    ax.add_patch(Wedge((sx, sy), ARENA * 2.2, th0, th1, fc=C_BLUE, alpha=0.08, ec="none"))
    # 可检测区 = 接收圆 ∩ ψ±90° 半平面
    ang = np.radians(np.linspace(th0, th1, 720))
    xs, ys = sx + R * np.cos(ang), sy + R * np.sin(ang)
    ax.fill(np.append(xs, sx), np.append(ys, sy), color=C_BLUE, alpha=0.30, lw=0)

    ax.scatter([p[0] for p in inside], [p[1] for p in inside], s=34, color=C_BLUE,
               zorder=5, label="内环 + 原点 (%d)" % len(inside))
    ax.scatter([p[0] for p in outside], [p[1] for p in outside], s=34, color=C_ORANGE,
               zorder=5, label="外环 $r$=%.0f m (%d)" % (r_out, len(outside)))
    ax.scatter([p[0] for p in seen], [p[1] for p in seen], s=150, facecolor="none",
               edgecolor=C_RED, lw=1.8, zorder=6, label="落在可检测区内 (%d)" % len(seen))
    ax.scatter([sx], [sy], s=95, marker="*", color=C_RED, zorder=7, label="定向源 $S$")
    ax.annotate("", xy=(sx + 420 * math.cos(math.radians(psi)),
                        sy + 420 * math.sin(math.radians(psi))),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.6))
    ax.set_xlim(-2400, 2400)
    ax.set_ylim(-2400, 2400)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.18)
    _save(fig, "q4_cover_gap.png")


# ------------------------------------------------------------------ 2. 站点集与骨架

def fig_scan_sets() -> None:
    """发布站点集与骨架巡游（原点 + 内环 + 外环，逐项取 ``ring_spec``）。"""
    spec = _ring_spec()
    r_in, n_in = spec[1]
    r_out, n_out = spec[2]
    route, tour_m = _best_tour(_stations())
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.add_patch(Circle((0, 0), ARENA, fill=False, ec="#333333", lw=1.4))
    ax.plot([0.0] + [p[0] for p in route], [0.0] + [p[1] for p in route],
            "-", color=C_BLUE, lw=1.1, alpha=0.75, zorder=3,
            label="骨架巡游 %.0f m（%d 站）" % (tour_m, len(route)))
    inner = [p for p in route if math.hypot(*p) < ARENA]
    outer = [p for p in route if math.hypot(*p) >= ARENA]
    ax.scatter([p[0] for p in inner], [p[1] for p in inner], s=34, color=C_BLUE,
               zorder=5, label="原点 + 内环 $r$=%.0f m (%d)" % (r_in, len(inner)))
    ax.scatter([p[0] for p in outer], [p[1] for p in outer], s=34, color=C_ORANGE,
               zorder=5, label="外环 $r$=%.0f m (%d)" % (r_out, len(outer)))
    ax.scatter([0.0], [0.0], s=90, marker="*", color=C_RED, zorder=6, label="起点")
    ax.set_xlim(-2100, 2100)
    ax.set_ylim(-2100, 2100)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ / m")
    ax.set_ylabel("$y$ / m")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.18)
    _save(fig, "q4_scan_sets.png")


# ------------------------------------------------------------------ 3. 时间去向

def fig_time_breakdown() -> None:
    """虚拟时间去向（独立测试集 1000 局平均，见 data/reports/q4_opt1_holdout1000_deploy.json）。

    数值口径: ``t_travel / t_detect / t_switch / t_clear``（= 模拟器字段）与
    ``q4_stages`` 的 ``survey_s / clear_s / hunt_s``。
    """
    labels = ["移动", "检测", "频道切换", "清除"]
    vals = [4416.6, 1147.1, 210.2, 113.3]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 120, "%.0f s" % v,
                ha="center", fontsize=9)
    ax.set_ylabel("虚拟时间 / s")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(vals) * 1.18)

    stages = ["巡游期\n(含顺路清除)", "收尾清除", "兜底搜索"]
    svals = [5474.4, 94.8, 318.0]
    bars2 = ax2.bar(stages, svals, color=[C_BLUE, C_ORANGE, C_GREEN], width=0.6)
    for b, v in zip(bars2, svals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 120, "%.0f s" % v,
                 ha="center", fontsize=9)
    ax2.set_ylabel("虚拟时间 / s")
    ax2.grid(axis="y", alpha=0.2)
    ax2.set_ylim(0, max(svals) * 1.18)
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "q4_phases.png")


# ------------------------------------------------------------------ 4. 两版对照

def fig_route_compare() -> None:
    """上一版规划器 vs 本文（同一批 200 局，tools/probes/q4_opt1_route.py）。"""
    items = ["平均定位清除\n/ (s·源$^{-1}$)", "每局移动\n/ m", "$L_{actual}-L_{off}$\n/ m"]
    old = [544.9, 25490.0, 9548.0]
    new = [466.6, 22218.0, 4629.0]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    for ax, name, a, b in zip(axes, items, old, new):
        bars = ax.bar(["上一版", "本文"], [a, b], color=[C_GRAY, C_BLUE], width=0.55)
        for bar, v in zip(bars, (a, b)):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                    "%.0f" % v if v >= 100 else "%.1f" % v, ha="center", fontsize=9.5)
        ax.set_ylabel(name)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, max(a, b) * 1.2)
    fig.subplots_adjust(wspace=0.42)
    _save(fig, "q4_compare.png")


# ------------------------------------------------------------------ 5. 角隙判据

def fig_cover_frontier() -> None:
    """外环点数 n 与最大角隙（接收半径取最坏情况 1000 m，内环与相位同发布配置）。

    左：发布族（内环 970 m×6、外环钉在边界 1800 m）—— 放宽后的速度优先布局；
    右：上一版族（内环 970 m×8、外环在场外 1850 m）—— (4) 式的零漏检设计。
    """
    spec = _ring_spec()
    r_in, n_in = spec[1]
    r_out_release = spec[2][0]

    def curve(rad_in: float, cnt_in: int, rad_out: float, ns) -> List[float]:
        out = []
        for n in ns:
            g, _where = grid_audit(_family(rad_in, cnt_in, rad_out, n), 5.0, 1000.0)
            out.append(g)
        return out

    ns_rel = list(range(8, 20))
    gaps_rel = curve(r_in, n_in, r_out_release, ns_rel)
    ns_old = list(range(10, 18))
    gaps_old = curve(r_in, 8, 1850.0, ns_old)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for _ax, ns, gaps, pub, pub_name in (
            (ax, ns_rel, gaps_rel, spec[2][1],
             "发布 $n$=%d（内环×%d, 外环%.0f m）" % (spec[2][1], n_in, r_out_release)),
            (ax2, ns_old, gaps_old, 15, "上一版 $n$=15（内环×8, 外环1850 m）")):
        _ax.plot(ns, gaps, "-o", color=C_BLUE, lw=1.5, ms=5.5, zorder=4)
        _ax.axhline(180.0, color=C_RED, ls="--", lw=1.3, zorder=3)
        ok = [(n, g) for n, g in zip(ns, gaps) if g <= 180.0]
        bad = [(n, g) for n, g in zip(ns, gaps) if g > 180.0]
        _ax.scatter([n for n, _g in ok], [g for _n, g in ok], s=64, color=C_BLUE,
                    zorder=5, label="零漏检")
        _ax.scatter([n for n, _g in bad], [g for _n, g in bad], s=64, facecolor="none",
                    edgecolor=C_RED, lw=1.6, zorder=5, label="存在漏检方向")
        if pub in ns:
            _ax.scatter([pub], [gaps[ns.index(pub)]], s=170, facecolor="none",
                        edgecolor=C_ORANGE, lw=1.9, zorder=6, label=pub_name)
        _ax.set_xlabel("外环点数 $n$")
        _ax.set_xticks(ns)
        _ax.grid(alpha=0.2)
    ax.set_ylabel("最大角隙 / °")
    for _ax, ns, gaps in ((ax, ns_rel, gaps_rel), (ax2, ns_old, gaps_old)):
        for n, g in zip(ns, gaps):
            _ax.annotate("%.0f°" % g, (n, g), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8.2, color="#333333")
    ax.set_ylim(190, 380)
    ax2.set_ylim(165, 280)
    ax2.text(10.1, 183.0, "$180°$ (漏检阈值)", color=C_RED, fontsize=8.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax2.legend(fontsize=8.5, frameon=False, loc="upper right")
    _save(fig, "q4_cover_frontier.png")


def _use_layout(name: str) -> None:
    """切换本脚本的站点集口径（论文插图与正文必须同一档，否则图、文互相矛盾）。

    ``current`` = 当前发布策略（22 站零漏检 + 中继，``PARAMS['ring_def']``）；
    ``ring18``  = 上一版论文口径（18 站，内环 970 m×6 + 外环 1800 m×11）。
    """
    if name == "ring18":
        S.PARAMS['ring_def'] = None
        S.PARAMS['ring_spec'] = ((0, 1), (970.0, 6), (1800.0, 11))
    elif name != "current":
        raise SystemExit("--layout 只能是 current | ring18")
    print("[figures] 站点集口径 = %s（%d 站，骨架 %.0f m）"
          % (name, len(S._layout_rows(S.PARAMS)), S.skeleton_length_m()))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="问题4 论文插图（写入 docs/figures/）")
    ap.add_argument("--layout", default="ring18",
                    choices=("current", "ring18"),
                    help="站点集口径；论文正文目前是 ring18（默认），"
                         "换策略后要重跑请显式写 --layout current")
    a = ap.parse_args(argv)
    _use_layout(a.layout)
    fig_cover_gap()
    fig_scan_sets()
    fig_time_breakdown()
    fig_route_compare()
    fig_cover_frontier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
