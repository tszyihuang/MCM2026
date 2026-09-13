# -*- coding: utf-8 -*-
"""问题4 覆盖几何探针: 评估任意补扫点集对「(位置, 类型, 定向方向)」空间的覆盖率。

纯几何, 不跑模拟器。用来回答一个具体问题:
**要让 99% 的源至少被看见一次, 最少要走多少米?**

覆盖判据与 ``simulator.core.InterferenceSource`` 一致:
  可见 ⟺ |Q − P| ≤ R  且  (全向源 或 Q 落在 ψ±90° 扇区内)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

Vec = Tuple[float, float]


def draw_sources(n_cases: int, seed0: int = 424242) -> List[Tuple[float, float, float, bool, float]]:
    """按 ``CaseGenerator.generate(4)`` 的分布抽样: (x, y, R, is_dir, psi)。"""
    rng = random.Random(int(seed0))
    out = []
    for _ in range(n_cases):
        n = rng.randint(10, 16)
        chans = rng.sample(range(1, 21), n)
        for _ch in sorted(chans):
            r = 1800.0 * math.sqrt(rng.random())
            th = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(th), r * math.sin(th)
            R = rng.uniform(1000.0, 1500.0)
            is_dir = rng.random() < 0.5
            psi = rng.uniform(0.0, 360.0) if is_dir else 0.0
            out.append((x, y, R, is_dir, psi))
    return out


def visible(q: Vec, s: Tuple[float, float, float, bool, float]) -> bool:
    x, y, R, is_dir, psi = s
    dx, dy = x - q[0], y - q[1]
    if dx * dx + dy * dy > R * R:
        return False
    if not is_dir:
        return True
    bearing = math.degrees(math.atan2(q[1] - y, q[0] - x)) % 360.0
    return abs(((bearing - psi + 180.0) % 360.0) - 180.0) <= 90.0 + 1e-9


def coverage(points: Sequence[Vec], sources, idx: int = -1) -> float:
    src = sources if idx < 0 else sources[:idx]
    hit = 0
    for s in src:
        for q in points:
            if visible(q, s):
                hit += 1
                break
    return hit / max(len(src), 1)


def tour_len(points: Sequence[Vec]) -> float:
    """最近邻 + 2-opt 的巡游长度 (从原点出发, 不返回)。点数小, 直接精确 DP 更准。"""
    n = len(points)
    if n == 0:
        return 0.0
    if n > 13:
        # 最近邻 + 2-opt
        unvisited = set(range(n))
        cur = (0.0, 0.0)
        order = []
        while unvisited:
            j = min(unvisited, key=lambda k: math.dist(cur, points[k]))
            unvisited.discard(j)
            order.append(j)
            cur = points[j]
        improved = True
        while improved:
            improved = False
            for a in range(len(order) - 1):
                for b in range(a + 1, len(order)):
                    i, j = order[a], order[b]
                    pa = points[order[a - 1]] if a else (0.0, 0.0)
                    pb = points[order[b + 1]] if b + 1 < len(order) else None
                    old = math.dist(pa, points[i]) + (math.dist(points[j], pb) if pb else 0.0)
                    new = math.dist(pa, points[j]) + (math.dist(points[i], pb) if pb else 0.0)
                    if new < old - 1e-9:
                        order[a:b + 1] = order[a:b + 1][::-1]
                        improved = True
        pts = [points[i] for i in order]
    else:
        pts = list(points)
    d = 0.0
    cur = (0.0, 0.0)
    for p in pts:
        d += math.dist(cur, p)
        cur = p
    return d


def ring(radius: float, n: int, phase: float = 0.0) -> List[Vec]:
    return [(radius * math.cos(phase + 2 * math.pi * i / n),
             radius * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]


def hex_grid(spacing: float, rmax: float = 1800.0) -> List[Vec]:
    pts: List[Vec] = []
    dy = spacing * math.sqrt(3.0) / 2.0
    j = 0
    y = -rmax
    while y <= rmax:
        x0 = -rmax + (spacing / 2.0 if j % 2 else 0.0)
        x = x0
        while x <= rmax:
            if math.hypot(x, y) <= rmax:
                pts.append((x, y))
            x += spacing
        y += dy
        j += 1
    return pts


def square_grid(spacing: float, rmax: float = 1800.0) -> List[Vec]:
    pts = []
    n = int(rmax / spacing)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = i * spacing, j * spacing
            if math.hypot(x, y) <= rmax:
                pts.append((x, y))
    return pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sources = draw_sources(a.cases)
    nd = sum(1 for s in sources if s[3])
    print("样本: %d 源 (定向 %d, %.1f%%)" % (len(sources), nd, 100.0 * nd / len(sources)))
    print("%-46s %6s %9s %10s %10s" % ("点集", "点数", "覆盖", "巡游m", "锚点内覆盖"))

    def report(name: str, pts: List[Vec]) -> None:
        c = coverage(pts, sources)
        t = tour_len(pts)
        print("%-46s %6d %9.4f %10.0f %10s" % (name, len(pts), c, t, ""))

    report("原点", [(0.0, 0.0)])
    report("原点+近场环 r600×6", [(0.0, 0.0)] + ring(600.0, 6))
    report("原点+环 r800×6", [(0.0, 0.0)] + ring(800.0, 6))
    report("原点+环 r900×6", [(0.0, 0.0)] + ring(900.0, 6))
    report("原点+环 r900×8", [(0.0, 0.0)] + ring(900.0, 8))
    report("原点+环 r1000×8", [(0.0, 0.0)] + ring(1000.0, 8))
    report("原点+环 r800×8+环 r1400×8", [(0.0, 0.0)] + ring(800.0, 8) + ring(1400.0, 8))
    report("原点+环 r900×8+环 r1500×8", [(0.0, 0.0)] + ring(900.0, 8) + ring(1500.0, 8))
    report("环 r900×8 (无原点)", ring(900.0, 8))
    report("环 r1000×8 (无原点)", ring(1000.0, 8))
    report("环 r1100×6", ring(1100.0, 6))
    report("环 r1273×6", ring(1273.0, 6))
    for sp in (600.0, 700.0, 800.0, 900.0, 1000.0):
        report("六边形网格 %.0f m" % sp, hex_grid(sp))
    for sp in (800.0, 1000.0, 1273.0):
        report("正方形网格 %.0f m" % sp, square_grid(sp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
