# -*- coding: utf-8 -*-
"""问题4 站点布局的覆盖审计（当前策略 = 中继测量版的双环骨架，点数随发布配置而变）。

发布配置用 ``PARAMS['ring_def']``（显式环定义，优先级最高）描述站点集，
``ring_spec`` 只是旧口径的兼容字段；本脚本一律读**生效**的那一份（见
:func:`layout_spec`），因此"外环少一点"这类对照不会因为字段优先级而失真。

回答两个问题，全部是纯几何、不跑模拟器：

1. **覆盖缺口**：以最坏情况 R = 1000 m（接收半径下界）计算，对圆域内任意 (位置, 定向方向)，
   站点集里是否**必然**至少有一站落在源的 ψ±90° 半平面内？
   —— 判据：把"在接收半径内的站点"按从源看过去的角度排序，若最大角隙 ≤ 180°，
   则无论 ψ 取何值都至少罩住一站；反之为漏检方向。
2. **覆盖重数 k**：按题目分布抽样真实的 (位置, 类型, ψ)，统计每个源被几站看到。

用法::

    python tools/probes/q4_opt1_layout.py --grid-step 5 --cases 20000
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.consts import ARENA_RADIUS_M, RADIUS_MIN_M              # noqa: E402
from robotdog.solver import sweeper4 as S                             # noqa: E402
from tools.probes.q4_cover import draw_sources, visible               # noqa: E402


def stations() -> list:
    """当前策略的站点坐标（发布配置的有效环定义展开）。"""
    return [(float(r[0]), float(r[1])) for r in S._layout_rows(S.PARAMS)]


def layout_spec() -> tuple:
    """发布配置的**生效**环定义 ``((r, n), ...)``：``ring_def`` 优先于 ``ring_spec``。"""
    rd = S.PARAMS.get('ring_def')
    if rd:
        return tuple((float(item[0]), int(item[1])) for item in rd)
    return tuple((float(r), int(n)) for r, n in S.PARAMS['ring_spec'])


def layout_points(spec) -> list:
    """按给定环定义 ``((r, n), ...)`` 展开站点坐标。

    ``spec`` 里的半径与发布配置的环一一对应；**相位沿用发布配置的 ``ring_def``**，
    这样"外环少一点"这类对照只改点数、不改相位（否则对照会与发布布局不可比）。
    """
    P = dict(S.PARAMS)
    rd = P.get('ring_def')
    if rd:
        phase_of = {round(float(item[0]), 6):
                    (float(item[2]) if len(item) > 2 and item[2] is not None else 0.0)
                    for item in rd}
        P['ring_def'] = tuple((float(r), int(n),
                               phase_of.get(round(float(r), 6), 0.0))
                              for r, n in spec)
    else:
        P['ring_def'] = None      # 显式抑制 ring_def, 否则它会盖掉 ring_spec
        P['ring_spec'] = spec
    return [(float(r[0]), float(r[1])) for r in S._layout_rows(P)]


def worst_gap(px: float, py: float, pts, rmax: float) -> float:
    """从 (px,py) 看过去，接收半径内各站的最大角隙（度）；无站可见时返回 360。"""
    ang = sorted(math.degrees(math.atan2(qy - py, qx - px)) % 360.0
                 for (qx, qy) in pts
                 if math.hypot(qx - px, qy - py) <= rmax + 1e-9)
    if not ang:
        return 360.0
    if len(ang) == 1:
        return 360.0
    gaps = [ang[i + 1] - ang[i] for i in range(len(ang) - 1)]
    gaps.append(ang[0] + 360.0 - ang[-1])
    return max(gaps)


def grid_audit(pts, step: float, rmax: float):
    """扫 (ρ, θ) 网格，找最大角隙（= 最坏情况下的覆盖缺口）。"""
    worst = (-1.0, None)
    n_rho = int(ARENA_RADIUS_M / step) + 1
    n_th = max(1, int(round(360.0 / step)))
    for i in range(n_rho):
        rho = i * step
        for j in range(n_th):
            th = 2.0 * math.pi * j / n_th
            px, py = rho * math.cos(th), rho * math.sin(th)
            g = worst_gap(px, py, pts, rmax)
            if g > worst[0]:
                worst = (g, (round(px, 1), round(py, 1), round(rho, 1)))
    return worst


def sample_audit(pts, n_cases: int):
    """按题目分布抽样，统计覆盖率、覆盖重数 k 与"只有外环看得见"的份额。"""
    src = draw_sources(n_cases)
    outer = [q for q in pts if math.hypot(*q) >= 1500.0]
    inner = [q for q in pts if math.hypot(*q) < 1500.0]
    cov_all = 0
    outside_only = 0          # 场内站点全部落空、只有场外外环看得见（D1 族）
    invisible = 0             # 任何站点都看不见
    ks = []
    k_dir = []
    for s in src:
        k = sum(1 for q in pts if visible(q, s))
        ks.append(k)
        if s[3]:
            k_dir.append(k)
        if k > 0:
            cov_all += 1
        else:
            invisible += 1
        if k > 0 and not any(visible(q, s) for q in inner) \
                and any(visible(q, s) for q in outer):
            outside_only += 1
    return src, ks, k_dir, cov_all / max(len(src), 1), outside_only, invisible


def main() -> int:
    ap = argparse.ArgumentParser(description="问题4 站点布局覆盖审计")
    ap.add_argument("--grid-step", type=float, default=5.0, help="(ρ, θ) 网格步长 (m/度)")
    ap.add_argument("--rmax", type=float, default=RADIUS_MIN_M,
                    help="最坏情况接收半径 (默认取题目下界 1000 m)")
    ap.add_argument("--cases", type=int, default=20000)
    a = ap.parse_args()

    pts = stations()
    spec = layout_spec()
    outer_spec = [(float(r), int(n)) for r, n in spec if float(r) >= 1500.0]
    inner_spec = [(float(r), int(n)) for r, n in spec if 1e-9 < float(r) < 1500.0]
    outer = [p for p in pts if math.hypot(*p) >= 1500.0]
    print("=== 站点集 ===")
    desc = (" + ".join(["原点 1"]
                       + ["内环 r=%.0f %d 点" % (r, n) for r, n in inner_spec]
                       + ["外环 r=%.0f %d 点" % (r, n) for r, n in outer_spec]))
    print("  %s = %d 站" % (desc, len(pts)))
    print("  外环 %d 点，均分角隙 = %.1f°" % (len(outer), 360.0 / len(outer)))
    print("  骨架巡游长度 = %.1f m" % S.skeleton_length_m())

    print("=== 全方向覆盖审计 (R = %.0f m, ρ×θ 网格 %.0f m/%.0f°) ==="
          % (a.rmax, a.grid_step, a.grid_step))
    g, where = grid_audit(pts, a.grid_step, a.rmax)
    print("  最大角隙 = %.2f°  (成对出现于 %s)" % (g, where))
    print("  判据 max_gap ≤ 180° ⇒ %s" % ("**零漏检**" if g <= 180.0 else "存在漏检方向"))
    if len(outer_spec) == 1:
        r_out, n_out = outer_spec[0]
        for n, name in ((n_out, "发布外环"), (n_out - 1, "少一点"), (n_out - 2, "再少一点")):
            alt = tuple((r, n) if float(r) == r_out else (r, k) for r, k in spec)
            gg, _ = grid_audit(layout_points(alt), a.grid_step, a.rmax)
            print("  外环 %2d 点 (%s): 最大角隙 %.1f° ⇒ %s"
                  % (n, name, gg, "零漏检" if gg <= 180.0 else "漏检"))

    print("=== 抽样审计 (题目分布, %d 局) ===" % a.cases)
    src, ks, k_dir, cov, only_out, invisible = sample_audit(pts, a.cases)
    print("  源数 %d  覆盖率 = %.6f  (未被任何站点看见 %d 个)"
          % (len(src), cov, sum(1 for k in ks if k == 0)))
    print("  只被外环看见（场内全落空）: %d (%.3f%%)；任何站点都看不见: %d (%.3f%%)"
          % (only_out, 100.0 * only_out / len(src), invisible, 100.0 * invisible / len(src)))
    print("  覆盖重数 k: 均值 %.2f  最小 %d  最大 %d"
          % (statistics.fmean(ks), min(ks), max(ks)))
    for thr in (1, 2, 3):
        n = sum(1 for k in ks if k < thr)
        print("    k < %d 的源: %d (%.3f%%)" % (thr, n, 100.0 * n / len(ks)))
    if k_dir:
        print("  定向源单独统计 (%d 个): 均值 %.2f, k=0 有 %d 个"
              % (len(k_dir), statistics.fmean(k_dir), sum(1 for k in k_dir if k == 0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
