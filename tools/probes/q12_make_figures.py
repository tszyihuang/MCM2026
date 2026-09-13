# -*- coding: utf-8 -*-
"""生成问题一、问题二论文插图 -> docs/figures/q12_*.png。

四张图与 ``docs/问题一与问题二论文.md`` 的图 1~图 4 一一对应:
  q12_q1_region     楔形交会与"直径圆不能覆盖"(含两类退化示意)
  q12_q1_coverage   覆盖率的布站依赖与采样口径敏感性
  q12_q2_feasible   第二检测点几何构造 + (L,β) 可行域与最坏直径地形
  q12_q2_worst_r    最坏源距 D(r) 的非单调性 + 候选区域 + "L 尖 β 平"

数据全部取自 ``tools/probes/q12_review.py`` 的独立复核实现(同一套几何代码,
不另算): 楔形半平面交、暴力直径、暴力最小包围圆、可行域判据与各实验函数。
图面规约与 ``tools/make_paper_figures.py`` / ``tools/probes/q4_make_figures.py``
一致: 只保留坐标轴、刻度、图例与数值标签, 解释写在论文图注里。

用法:
    python tools\\probes\\q12_make_figures.py
    python tools\\probes\\q12_make_figures.py 3 4    # 只重生成图 3、图 4
"""
from __future__ import annotations

import math
import os
import sys

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import q12_review as Q  # noqa: E402
from tools import figstyle as FS  # noqa: E402

FS.apply_style()

FIG = os.path.join("docs", "figures")
#: 配色见 tools/figstyle.py —— 主几何=深蓝, 区域/对照=中粉, 次几何=中蓝,
#: 高亮/阈值=深玫, 中性参考=灰蓝 (文字与散点用其加深档 C_SLATE_TEXT)
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = (
    FS.BLUE, FS.PINK, FS.BLUE_LIGHT, FS.ROSE, FS.SLATE)
C_SLATE_TEXT = FS.SLATE_TEXT

R_LO_CONS = 5.0          # 保守档 r_lo
R_LO_ENG = 100.0         # 工程档 r_lo
R_MAX = 1500.0
L_STAR, B_STAR = 1004.10, 34.80          # 直径口径最优(保守档)
L_AREA, B_AREA = 1004.53, 24.88          # 面积口径最优(保守档)


def _save(fig, name: str) -> None:
    FS.save(fig, os.path.join(FIG, name))


def _tight(fig) -> None:
    """tight_layout: 屏蔽 inset_axes 等与它不兼容的告警(版面本身已逐图核对)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def _ray(ax, s, deg: float, length: float, **kw) -> None:
    a = math.radians(deg)
    ax.plot([s[0], s[0] + length * math.cos(a)],
            [s[1], s[1] + length * math.sin(a)], **kw)


#: 楔形交的裁剪框 (与 Q.BOX 同量级): 交区域触到框 = 无界
_WEDGE_BOX = 60.0


def _wedge_intersection(stations, thetas, eps: float, box: float = _WEDGE_BOX):
    """若干楔形 ``{P: ∠(P−S_i, θ_i) ≤ eps}`` 之交 ∩ [−box, box]²。

    直接复用 ``Q.wedge_halfplanes`` / ``Q.clip``——与正文算法是同一套半平面裁剪, 图上的
    形状不是手画的; 返回空表即"交为空", 触到裁剪框即"无界"。示意图的 ``eps`` 比真实的
    ±1° 大, 否则两条边线在图上分不开 (图注里已注明"误差角已放大")。
    """
    poly = [np.array([-box, -box]), np.array([box, -box]),
            np.array([box, box]), np.array([-box, box])]
    for s, th in zip(stations, thetas):
        for p, n in Q.wedge_halfplanes(np.asarray(s, float), th, eps):
            poly = Q.clip(poly, p, n)
            if not poly:
                return []
    return poly


def _example_region(seed: int = 17, trials: int = 4000):
    """从 §1.3 均匀布站采样器里取一个 η≈1.10 的真实算例(只用于图 1 右幅)。

    返回 (stations, thetas, poly, d, r*, eta); 找不到就抛错, 不静默降级。
    """
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        rho = R_MAX * np.sqrt(rng.uniform(0.0, 1.0, size=3))
        az = rng.uniform(0.0, 360.0, size=3)
        pts = [rho[i] * Q._u(float(az[i])) for i in range(3)]
        errs = rng.uniform(-1.0, 1.0, size=3)
        ths = [math.degrees(math.atan2(-p[1], -p[0])) + e for p, e in zip(pts, errs)]
        poly = Q.region(pts, ths)
        if not poly or len(poly) not in (4, 5, 6):
            continue
        d, _, _ = Q.diameter_pair(poly)
        if d > 1e4 or d < 20.0:
            continue
        ok, d, r, eta = Q.covered_by_diameter_circle(poly)
        if 1.09 <= eta <= 1.11:
            return pts, ths, poly, d, r, eta
    raise RuntimeError("未找到满足条件的示例区域, 请放宽筛选区间")


# ------------------------------------------------------------------ 图 1 楔形交会与直径圆

def fig_q1_region() -> None:
    pts, ths, poly, d, r_mec, eta = _example_region()

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1.0, 1.05, 1.0]})

    # ---- (a) 楔形 = 两条半平面之交; 三站相交 = 凸多边形 R(示意图, 误差角已放大)
    st = [np.array([0.0, 0.0]), np.array([4.6, 0.8]), np.array([2.1, 4.2])]
    g = np.array([2.05, 1.90])
    th = [math.degrees(math.atan2(g[1] - s[1], g[0] - s[0])) for s in st]
    half = 10.0                      # 示意用放大半角(真实 ±1°, 见正文)
    reg = Q.region(st, th, box=60.0)
    assert reg, "示意图的半平面交为空"
    ax_a.add_patch(MplPolygon(np.array(reg), closed=True, fc=C_ORANGE,
                              alpha=0.55, ec=C_ORANGE, lw=1.4, zorder=3))
    for k, (s, t) in enumerate(zip(st, th)):
        _ray(ax_a, s, t - half, 9.0, color=C_BLUE, lw=1.0, ls="--", alpha=0.85)
        _ray(ax_a, s, t + half, 9.0, color=C_BLUE, lw=1.0, ls="--", alpha=0.85)
        _ray(ax_a, s, t, 6.2, color=C_GRAY, lw=0.8, ls=":", alpha=0.8)
        ax_a.plot(*s, marker="s", ms=5.0, color=C_BLUE, zorder=4)
        ax_a.annotate(rf"$S_{k + 1}$", s, textcoords="offset points",
                      xytext=(4, -11), color=C_BLUE)
    ax_a.plot(*g, marker="*", ms=9.0, color=C_RED, zorder=5)
    ax_a.annotate("真源", g, textcoords="offset points", xytext=(7, 2), color=C_RED)
    ax_a.annotate(r"$R$", (2.05, 1.05), color=C_RED, ha="center")
    ax_a.set_xlim(-4.6, 7.4)
    ax_a.set_ylim(-2.6, 6.6)
    ax_a.set_aspect("equal")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax_a.spines[sp].set_visible(False)
    ax_a.set_xlabel(r"(a) 楔形 $W_i$ 交会 $\Rightarrow$ 凸多边形 $R$（示意：误差角已放大）")

    # ---- (b) 真实算例: 以直径对为直径的圆 vs 最小包围圆
    P = np.array(poly)
    ctr, r_star = Q.min_enclosing_circle(poly)
    dd, i_d, j_d = Q.diameter_pair(poly)
    A, B = poly[i_d], poly[j_d]
    mid = (A + B) / 2.0
    ax_b.add_patch(Circle(mid, dd / 2.0, fill=False, ec=C_RED, ls="--", lw=1.3,
                          zorder=2, label=r"以直径对 $AB$ 为直径的圆"))
    ax_b.add_patch(Circle(ctr, r_star, fill=False, ec=C_GREEN, lw=1.3,
                          zorder=2, label=r"最小包围圆 $r^*$"))
    ax_b.add_patch(MplPolygon(P, closed=True, fc=C_ORANGE, alpha=0.28,
                              ec=C_ORANGE, lw=1.4, zorder=3))
    ax_b.plot([A[0], B[0]], [A[1], B[1]], color=C_BLUE, lw=1.3, zorder=4)
    ax_b.plot([A[0], B[0]], [A[1], B[1]], ls="none", marker="o", ms=4.0,
              color=C_BLUE, zorder=5)
    ax_b.annotate(r"$A$", A, textcoords="offset points", xytext=(3, 3), color=C_BLUE)
    ax_b.annotate(r"$B$", B, textcoords="offset points", xytext=(3, 3), color=C_BLUE)
    outside = [p for p in poly if float(np.dot(p - A, p - B)) > 0.0]
    if outside:
        o = np.array(outside)
        ax_b.plot(o[:, 0], o[:, 1], ls="none", marker="o", ms=5.5,
                  mfc="none", mec=C_RED, mew=1.4, zorder=6)
    ax_b.plot(*ctr, marker="+", ms=6.0, color=C_GREEN, zorder=5)
    ax_b.annotate(rf"$d={d:.1f}$ m", (float(mid[0]), float(mid[1])),
                  textcoords="offset points", xytext=(-14, -16), color=C_BLUE)
    ax_b.annotate(rf"$r^*={r_mec:.1f}$ m", (float(ctr[0]), float(ctr[1])),
                  textcoords="offset points", xytext=(-16, 10), color=C_GREEN)
    ax_b.annotate(rf"$\eta=2r^*/d={eta:.3f}>1$", (0.03, 0.03),
                  xycoords="axes fraction", ha="left", color=C_RED)
    ax_b.set_aspect("equal")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.legend(loc="upper right", fontsize=8.0, frameon=False)
    ax_b.set_xlabel(r"(b) 真实算例：直径圆漏掉区域顶点 $\Rightarrow$ 不能覆盖 $R$")

    # ---- (c) 两类退化(示意): 无界 / 交为空
    #: 两类退化画在同一个坐标系里上下排开 (inset_axes 会撑动主图版面, 不用);
    #: 半角比真实 ±1° 放大, 否则两条边线在图上分不开 —— 图注已注明"示意"。
    EPS_W, EPS_C = 5.0, 4.0
    OFF_C = np.array([0.0, -0.6])            # (ii) 整体下移, 与 (i) 分开
    ax_c.set_xlim(-7.6, 7.6)
    ax_c.set_ylim(-3.8, 8.4)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.axis("off")
    ax_c.set_xlabel("(c) 两类退化情形（示意：误差角已放大）")

    # (i) 两站方向差 < 2ε ⇒ 两楔形方向区间重叠, 交区域无界 (直径无穷)
    S1, TH1 = [(-6.9, 4.9), (-4.1, 5.4)], [0.5, 1.0]
    unbounded = _wedge_intersection(S1, TH1, EPS_W)
    assert unbounded, "(c-i) 两楔形必须有交"
    assert max(p[0] for p in unbounded) >= _WEDGE_BOX - 1e-6, "(c-i) 交区域应为无界"
    ax_c.add_patch(MplPolygon(np.array(unbounded), closed=True, fc=C_ORANGE,
                              alpha=0.50, ec="none", zorder=2))
    for s, th in zip(S1, TH1):
        for deg in (th - EPS_W, th + EPS_W):
            _ray(ax_c, s, deg, 14.4, color=C_BLUE, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
        _ray(ax_c, s, th, 14.4, color=C_GRAY, lw=0.8, ls=(0, (1, 2)), zorder=3)
        ax_c.plot(*s, marker="s", ms=5.0, color=C_BLUE, zorder=6)
    ax_c.annotate(r"$S_1$", S1[0], textcoords="offset points", xytext=(3, -11),
                  color=C_BLUE, fontsize=8.5)
    ax_c.annotate(r"$S_2$", S1[1], textcoords="offset points", xytext=(3, 5),
                  color=C_BLUE, fontsize=8.5)
    ax_c.annotate("", xy=(7.55, 5.60), xytext=(6.55, 5.42),
                  arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.3))
    ax_c.annotate("无界", xy=(6.9, 6.55), ha="right", color=C_RED, fontsize=8.5)
    ax_c.text(-7.5, 8.3, r"(i) 方向差 $<2\varepsilon$", color="#333333",
              fontsize=8.0, va="top")

    # (ii) n ≥ 3 时 ±1° 误差自相矛盾 ⇒ 2n 条半平面交为空 (两两相交, 但无公共点)
    #: 三条示向轴线都与半径 0.8 的同心圆相切 (经典"三条方位线不共点"): 站点取在切线
    #: 的延长线上, 示向度 = 站点指向切点的方向; 数据自洽时三元公共点应落在切圆圆心 (× 处)
    S2 = [tuple(np.array(p) + OFF_C) for p in ((-6.0, -1.6), (6.0, -1.6), (0.0, 2.4))]
    T2 = [tuple(np.array(p) + OFF_C)
          for p in ((-0.304, 0.740), (-0.105, -0.793), (0.754, 0.267))]
    TH2 = [math.degrees(math.atan2(t[1] - s[1], t[0] - s[0])) for s, t in zip(S2, T2)]
    assert not _wedge_intersection(S2, TH2, EPS_C), "(c-ii) 三元交必须为空"
    for i, j in ((0, 1), (1, 2), (0, 2)):
        assert _wedge_intersection([S2[i], S2[j]], [TH2[i], TH2[j]], EPS_C), \
            "(c-ii) 两两必须相交"
    for i, (s, th) in enumerate(zip(S2, TH2)):
        for deg in (th - EPS_C, th + EPS_C):
            _ray(ax_c, s, deg, 9.5, color=C_BLUE, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
        _ray(ax_c, s, th, 9.5, color=C_GRAY, lw=0.8, ls=(0, (1, 2)), zorder=3)
        ax_c.plot(*s, marker="s", ms=5.0, color=C_BLUE, zorder=6)
        ax_c.annotate(r"$S_%d$" % (i + 1), s, textcoords="offset points",
                      xytext=(4, -11), color=C_BLUE, fontsize=8.5)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        pair = _wedge_intersection([S2[i], S2[j]], [TH2[i], TH2[j]], EPS_C)
        ax_c.add_patch(MplPolygon(np.array(pair), closed=True, fc=C_ORANGE,
                                  alpha=0.60, ec=C_RED, lw=0.8, zorder=5))
    ax_c.plot([OFF_C[0]], [OFF_C[1]], marker="x", ms=8.0, mew=1.6, color=C_RED, zorder=8)
    ax_c.annotate("交为空", xy=tuple(OFF_C), xytext=(0.0, -1.95), ha="center",
                  color=C_RED, fontsize=8.5,
                  arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.0,
                                  shrinkA=2, shrinkB=5), zorder=9)
    ax_c.text(-7.5, 3.5, r"(ii) $n\geq3$ 且 $\pm1^\circ$ 矛盾", color="#333333",
              fontsize=8.0, va="top")

    _tight(fig)
    _save(fig, "q12_q1_region.png")
    print("  [图1] 示例区域 d=%.3f m  r*=%.3f m  eta=%.5f (%d 个顶点)"
          % (d, r_mec, eta, len(poly)))


# ------------------------------------------------------------------ 图 2 覆盖率与采样口径

def fig_q1_coverage() -> None:
    tab = Q.exp13_coverage_table(trials=2000, seed=17)
    keys = ["uniform", "arc100", "arc90"]
    names = ["全场均匀\n(半径 1500 m\n圆域)",
             "绕源圆弧\n$R$=1000 m\n张角 100°~360°",
             "绕源圆弧\n$R$=1000 m\n张角 90°~360°"]
    cov = [tab[k][0] for k in keys]
    eta_mean = [tab[k][4] for k in keys]
    eta_max = [tab[k][5] for k in keys]

    sens = Q.exp14_sampler_sensitivity(trials=1200, seed=5)
    sk = [("equal", "fixed"), ("equal", "random"),
          ("uniform", "fixed"), ("uniform", "random")]
    sname = ["等角间隔\n等半径", "等角间隔\n随机半径",
             "张角内均匀\n等半径", "张角内均匀\n随机半径"]
    scov = [sens[k][0] for k in sk]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.8, 4.2),
                                     gridspec_kw={"width_ratios": [1.0, 1.12]})

    xs = np.arange(3)
    ax_a.bar(xs, cov, width=0.56, color=[C_BLUE, C_ORANGE, C_ORANGE],
             alpha=0.85, label="覆盖率（左轴）", zorder=2)
    for x, v in zip(xs, cov):
        ax_a.annotate("%.3f" % v, (x, v), textcoords="offset points",
                      xytext=(0, 3), ha="center", color="#333333")
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(names, fontsize=8.0)
    ax_a.set_ylim(0.0, 0.95)
    ax_a.set_ylabel("直径圆覆盖率")
    ax_a.grid(axis="y", color="#dddddd", lw=0.7)
    ax_a.set_axisbelow(True)

    ax_a2 = ax_a.twinx()
    ax_a2.plot(xs, eta_max, marker="o", ms=5.0, color=C_RED, lw=1.2,
               label=r"最大 $\eta$（右轴）", zorder=3)
    ax_a2.plot(xs, eta_mean, marker="^", ms=4.5, color=C_SLATE_TEXT, lw=1.0,
               ls="--", label=r"平均 $\eta$（右轴）", zorder=3)
    for x, v in zip(xs, eta_max):
        ax_a2.annotate("%.3f" % v, (x, v), textcoords="offset points",
                       xytext=(-6, 6), color=C_RED, fontsize=8.5)
    ax_a2.set_ylim(0.96, 1.24)
    ax_a2.set_ylabel(r"$\eta=2r^*/d$")
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8.0, frameon=False)
    ax_a.set_xlabel("(a) 布站方案（2000 例/行，同一判据）")

    xs2 = np.arange(4)
    ax_b.plot(xs2, scov, marker="o", ms=6.0, color=C_BLUE, lw=1.4, zorder=3,
              label="圆弧布站口径")
    for x, v in zip(xs2, scov):
        ax_b.annotate("%.3f" % v, (x, v), textcoords="offset points",
                      xytext=(0, 8), ha="center", color=C_BLUE)
    ax_b.axhline(cov[0], color=C_GREEN, ls="--", lw=1.2,
                 label="全场均匀基线 %.3f" % cov[0])
    ax_b.set_xticks(xs2)
    ax_b.set_xticklabels(sname, fontsize=8.0)
    ax_b.set_xlim(-0.42, 4.10)
    ax_b.set_ylim(0.24, 0.88)
    ax_b.set_ylabel("直径圆覆盖率")
    ax_b.grid(axis="y", color="#dddddd", lw=0.7)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right", fontsize=8.0, frameon=False)
    ax_b.set_xlabel("(b) 圆弧布站的口径敏感性（1200 例/格）")
    ax_b.annotate("", xy=(3.45, scov[3]), xytext=(3.45, scov[0]),
                  arrowprops=dict(arrowstyle="<->", color=C_SLATE_TEXT, lw=1.0))
    ax_b.annotate("+%.1f pt" % (100 * (scov[3] - scov[0])), xy=(3.55, 0.55),
                  color=C_SLATE_TEXT, fontsize=8.5, ha="left")

    _tight(fig)
    _save(fig, "q12_q1_coverage.png")
    print("  [图2] 覆盖率 %s ; 口径敏感性 %s"
          % (["%.3f" % v for v in cov], ["%.3f" % v for v in scov]))


# ------------------------------------------------------------------ 图 3 几何构造与可行域

def fig_q2_feasible() -> None:
    b = math.radians(B_STAR)
    s1 = np.array([0.0, 0.0])
    s2 = L_STAR * np.array([math.cos(b), math.sin(b)])
    g = np.array([R_MAX, 0.0])
    th2 = math.degrees(math.atan2(g[1] - s2[1], g[0] - s2[0]))
    rho2 = float(np.linalg.norm(g - s2))
    alpha = Q.q2_alpha(L_STAR, B_STAR, R_MAX)
    near = (L_STAR * math.cos(b), 0.0)          # r = u = L·cosβ

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.4, 4.7),
                                     gridspec_kw={"width_ratios": [1.05, 1.0]})

    # ---- (a) 几何构造
    _ray(ax_a, s1, 0.0, 1560.0, color=C_GRAY, lw=0.9, ls=":")
    _ray(ax_a, s1, -1.0, 1620.0, color=C_BLUE, lw=0.9, ls="--", alpha=0.8)
    _ray(ax_a, s1, +1.0, 1620.0, color=C_BLUE, lw=0.9, ls="--", alpha=0.8)
    _ray(ax_a, s2, th2 - 1.0, 1250.0, color=C_BLUE, lw=0.9, ls="--", alpha=0.8)
    _ray(ax_a, s2, th2 + 1.0, 1250.0, color=C_BLUE, lw=0.9, ls="--", alpha=0.8)
    ax_a.plot([s1[0], s2[0]], [s1[1], s2[1]], color=C_BLUE, lw=1.6)
    ax_a.plot([s2[0], g[0]], [s2[1], g[1]], color=C_GREEN, lw=1.6)
    ax_a.plot([s1[0], s2[0], g[0]], [s1[1], s2[1], g[1]], ls="none")
    for s, lab, off in ((s1, r"$S_1$", (-4, -17)), (s2, r"$S_2$", (5, 8))):
        ax_a.plot(*s, marker="s", ms=6.5, color="#333333", zorder=5)
        ax_a.annotate(lab, s, textcoords="offset points", xytext=off, color="#333333")
    ax_a.plot(*g, marker="*", ms=11.0, color=C_RED, zorder=6)
    ax_a.annotate(r"$G=(r,0)$", g, textcoords="offset points",
                  xytext=(-16, 8), color=C_RED)
    ax_a.plot(*near, marker="|", ms=9.0, color=C_SLATE_TEXT, zorder=5)
    ax_a.annotate(r"$r=u=L\cos\beta$", near, textcoords="offset points",
                  xytext=(-10, -18), color=C_SLATE_TEXT, fontsize=8.5)
    arc = np.linspace(0.0, b, 40)
    rr = 470.0
    ax_a.plot(rr * np.cos(arc), rr * np.sin(arc), color="#333333", lw=1.0)
    ax_a.annotate(r"$\beta=%.2f^\circ$" % B_STAR, (rr, 0.0),
                  textcoords="offset points", xytext=(0, 12),
                  ha="center", color="#333333")
    u1 = (s1 - g) / np.linalg.norm(s1 - g)
    u2 = (s2 - g) / np.linalg.norm(s2 - g)
    a0 = math.atan2(u1[1], u1[0])
    a1 = math.atan2(u2[1], u2[0])
    garc = np.linspace(a0, a1, 60)
    ax_a.plot(g[0] + 330.0 * np.cos(garc), g[1] + 330.0 * np.sin(garc),
              color="#333333", lw=1.0)
    am = 0.5 * (a0 + a1)
    ax_a.annotate(r"$\alpha=%.1f^\circ$" % alpha,
                  (g[0] + 330.0 * math.cos(am), g[1] + 330.0 * math.sin(am)),
                  textcoords="offset points", xytext=(0, -16),
                  ha="center", va="top", color="#333333")
    ax_a.annotate(r"$L=%.1f$ m" % L_STAR,
                  (0.62 * s2[0], 0.62 * s2[1]),
                  textcoords="offset points", xytext=(0, -20), color=C_BLUE)
    ax_a.annotate(r"$\rho_2(r)=%.0f$ m" % rho2,
                  (0.5 * (s2[0] + g[0]), 0.5 * (s2[1] + g[1])),
                  textcoords="offset points", xytext=(12, 12), color=C_GREEN)
    ax_a.set_xlim(-200.0, 1740.0)
    ax_a.set_ylim(-300.0, 940.0)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel(r"(a) 第二检测点几何（以 $r=r_{\max}$ 的最坏源为例）")
    ax_a.set_ylabel("y / m")

    # ---- (b) 可行域与最坏直径地形
    rs = np.concatenate([np.linspace(R_LO_CONS, R_MAX, 31), [L_STAR * math.cos(b)]])
    Ls = np.arange(500.0, 1070.0 + 1e-9, 10.0)
    bs = np.arange(15.0, 46.0 + 1e-9, 0.5)
    Z = np.full((bs.size, Ls.size), np.nan)
    for i, bb in enumerate(bs):
        for j, LL in enumerate(Ls):
            ok, _ = Q._feasible(float(LL), float(bb), R_LO_CONS, R_MAX)
            if not ok:
                continue
            w = 0.0
            for r in rs:
                poly, _, _ = Q.q2_geometry(float(LL), float(bb), float(r))
                if not poly:
                    w = float("nan")
                    break
                w = max(w, Q.diameter(poly))
            Z[i, j] = w
    Zm = np.ma.masked_invalid(Z)
    pcm = ax_b.pcolormesh(Ls, bs, Zm, cmap=FS.CMAP_PINK, shading="auto",
                          vmin=121.674, vmax=320.0)
    ax_b.contour(Ls, bs, Zm, levels=[130.0, 140.0, 160.0, 200.0, 260.0],
                 colors=FS.ROSE, linewidths=0.7, alpha=0.9)
    ob = np.arange(15.0, 46.0, 0.1)
    ax_b.plot([Q.boundary_L(float(t), R_LO_CONS) for t in ob], ob,
              color=C_BLUE, lw=1.6,
              label=r"$\rho_2(r_{lo})=R_{eff}$（可行域上界）")
    ax_b.plot([L_STAR], [B_STAR], marker="*", ms=13.0, color=C_BLUE,
              mec="white", mew=0.8, zorder=6,
              label=r"直径口径最优 $(%.1f,\,%.2f^\circ)$" % (L_STAR, B_STAR))
    ax_b.plot([L_AREA], [B_AREA], marker="D", ms=6.0, color=C_GREEN,
              mec="white", mew=0.8, zorder=6,
              label=r"面积口径最优 $(%.1f,\,%.2f^\circ)$" % (L_AREA, B_AREA))
    ax_b.set_xlim(500.0, 1070.0)
    ax_b.set_ylim(15.0, 46.0)
    ax_b.set_xlabel(r"(b) 可行域与 $\max_r D$ 地形：$L$ / m")
    ax_b.set_ylabel(r"$\beta$ / °")
    ax_b.legend(loc="upper left", fontsize=7.6, frameon=False)
    cb = fig.colorbar(pcm, ax=ax_b, pad=0.02)
    cb.set_label(r"$\max_r D$ / m", fontsize=8.5)
    cb.ax.tick_params(labelsize=8.0)

    _tight(fig)
    _save(fig, "q12_q2_feasible.png")
    ok = np.isfinite(Z)
    Ls_f = [float(Ls[j]) for j in range(Ls.size) if ok[:, j].any()]
    bs_f = [float(bs[i]) for i in range(bs.size) if ok[i, :].any()]
    print("  [图3] 可行域 beta∈[%.1f, %.1f]°  L∈[%.0f, %.0f] m  alpha(r_max)=%.2f°"
          % (min(bs_f), max(bs_f), min(Ls_f), max(Ls_f), alpha))


# ------------------------------------------------------------------ 图 4 最坏源距与候选区域

def fig_q2_worst_r() -> None:
    rs = np.logspace(math.log10(R_LO_CONS), math.log10(R_MAX), 600)
    D = []
    for r in rs:
        poly, _, _ = Q.q2_geometry(L_STAR, B_STAR, float(r))
        D.append(Q.diameter(poly) if poly else np.nan)
    D = np.array(D)
    i_max = int(np.nanargmax(D))
    i_near = int(np.nanargmax(np.where(rs < 200.0, D, np.nan)))
    u = L_STAR * math.cos(math.radians(B_STAR))
    i_min = int(np.nanargmin(np.where(np.abs(rs - u) < 60.0, D, np.nan)))
    d_star = float(D[i_max])

    fig = plt.figure(figsize=(13.2, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # ---- (a) D(r) 的非单调性
    ax_a.plot(rs, D, color=C_BLUE, lw=1.5, label=r"$D(r)$（直径）")
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"(a) 源距 $r$ / m（对数轴），$L^*=%.2f$ m, $\beta^*=%.2f^\circ$"
                    % (L_STAR, B_STAR))
    ax_a.set_ylabel("定位区域直径 / m")
    for idx, col, lab, off, ha in ((i_near, C_ORANGE, r"近场次峰 $%.1f$ m" % D[i_near], (8, 4), "left"),
                                   (i_min, C_SLATE_TEXT, r"局部极小 $%.1f$ m" % D[i_min], (-10, -12), "right"),
                                   (i_max, C_RED, r"全局最大 $%.3f$ m" % D[i_max], (-96, -20), "left")):
        ax_a.plot([rs[idx]], [float(D[idx])], marker="o", ms=5.0, color=col)
        ax_a.annotate(lab, (rs[idx], float(D[idx])), textcoords="offset points",
                      xytext=off, color=col, fontsize=8.5, ha=ha)
    for r in (R_LO_CONS, u, R_MAX):
        ax_a.axvline(r, color=C_GRAY, lw=0.8, ls="--", alpha=0.8)
    ax_a.annotate("三点比较法只取这三处", (R_LO_CONS * 1.15, 0.55),
                  xycoords=("data", "axes fraction"), color=C_SLATE_TEXT,
                  fontsize=8.5)
    ax_a.set_ylim(0.0, 1.22 * d_star)
    ax_a.legend(loc="upper left", fontsize=8.0, frameon=False)

    # ---- (b) 候选区域: 相对最优放宽 2% / 5% / 10%
    rs2 = np.concatenate([np.linspace(R_LO_CONS, R_MAX, 31), [u]])
    Ls = np.arange(915.0, 1010.0 + 1e-9, 1.0)
    bs = np.arange(20.0, 45.0 + 1e-9, 0.25)
    Z = np.full((bs.size, Ls.size), np.nan)
    for i, bb in enumerate(bs):
        for j, LL in enumerate(Ls):
            ok, _ = Q._feasible(float(LL), float(bb), R_LO_CONS, R_MAX)
            if not ok:
                continue
            w = 0.0
            for r in rs2:
                poly, _, _ = Q.q2_geometry(float(LL), float(bb), float(r))
                if not poly:
                    w = float("nan")
                    break
                w = max(w, Q.diameter(poly))
            Z[i, j] = w
    Zm = np.ma.masked_invalid(Z)
    lev2, lev5, lev10 = 1.02 * d_star, 1.05 * d_star, 1.10 * d_star
    ax_b.contourf(Ls, bs, Zm, levels=[0.0, 1.0e9], colors=[FS.BLUE_PALE])
    ax_b.contourf(Ls, bs, Zm, levels=[0.0, lev10], colors=[FS.PINK_PALE])
    ax_b.contourf(Ls, bs, Zm, levels=[0.0, lev5], colors=[FS.PINK_LIGHT])
    ax_b.contourf(Ls, bs, Zm, levels=[0.0, lev2], colors=[FS.PINK])
    ax_b.contour(Ls, bs, Zm, levels=[lev2, lev5, lev10],
                 colors=[FS.ROSE, C_ORANGE, FS.PINK_LIGHT], linewidths=1.2)
    ob = np.arange(20.0, 45.0, 0.1)
    ax_b.plot([Q.boundary_L(float(t), R_LO_CONS) for t in ob], ob,
              color=C_BLUE, lw=1.6, label=r"$L_{cap}(\beta)$")
    ax_b.plot([L_STAR], [B_STAR], marker="*", ms=13.0, color=C_RED,
              mec="white", mew=0.8, zorder=6, label=r"$S_2^*$")
    for col, lab in ((FS.PINK, r"$\leq 2\%$"), (FS.PINK_LIGHT, r"$\leq 5\%$"),
                     (FS.PINK_PALE, r"$\leq 10\%$")):
        ax_b.plot([], [], color=col, lw=6.0, label=lab)
    ax_b.set_xlim(915.0, 1010.0)
    ax_b.set_ylim(20.0, 45.0)
    ax_b.set_xlabel(r"(b) 候选区域：$L$ / m")
    ax_b.set_ylabel(r"$\beta$ / °")
    ax_b.legend(loc="lower left", fontsize=7.8, frameon=False, ncol=2)

    # ---- (c) "L 尖 beta 平": 三条 beta 剖面
    bb = np.arange(24.0, 44.0, 0.5)
    for off, col, lab in ((0.0, C_BLUE, r"$L=L_{cap}(\beta)$"),
                          (-15.0, C_ORANGE, r"$L=L_{cap}(\beta)-15$ m"),
                          (-30.0, C_GREEN, r"$L=L_{cap}(\beta)-30$ m")):
        ys = []
        for t in bb:
            LL = Q.boundary_L(float(t), R_LO_CONS) + off
            if not Q._feasible(LL, float(t), R_LO_CONS, R_MAX)[0]:
                ys.append(np.nan)
                continue
            w = 0.0
            for r in rs2:
                poly, _, _ = Q.q2_geometry(LL, float(t), float(r))
                if not poly:
                    w = np.nan
                    break
                w = max(w, Q.diameter(poly))
            ys.append(w)
        ax_c.plot(bb, np.array(ys) / d_star, marker="o", ms=3.4, lw=1.3,
                  color=col, ls="-" if off == 0.0 else "--", label=lab)
    ax_c.axhline(1.0, color=C_GRAY, lw=0.8, ls=":")
    ax_c.set_xlabel(r"(c) $\beta$ 剖面：$\beta$ 方向平坦、$L$ 方向陡")
    ax_c.set_ylabel(r"$\max_r D \;/\; D^*$")
    ax_c.set_ylim(0.98, 1.22)
    ax_c.legend(loc="upper right", fontsize=8.0, frameon=False)

    _tight(fig)
    _save(fig, "q12_q2_worst_r.png")
    print("  [图4] u=%.1f m  D*=%.3f @ r=%.1f m ; 次峰 %.3f @ r=%.1f m ; 极小 %.3f @ r=%.1f m"
          % (u, d_star, rs[i_max], D[i_near], rs[i_near], D[i_min], rs[i_min]))


def main() -> None:
    want = set(sys.argv[1:]) or {"1", "2", "3", "4"}
    print("生成 docs/figures/q12_*.png ... %s" % ",".join(sorted(want)))
    if "1" in want:
        fig_q1_region()
    if "2" in want:
        fig_q1_coverage()
    if "3" in want:
        fig_q2_feasible()
    if "4" in want:
        fig_q2_worst_r()
    print("done.")


if __name__ == "__main__":
    main()
