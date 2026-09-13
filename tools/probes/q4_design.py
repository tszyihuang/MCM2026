# -*- coding: utf-8 -*-
"""问题4 补扫点集的**离线设计探针** (不跑策略, 只算几何 + 计数)。

对给定点集与种子区间, 逐局统计:

* 巡游路程 ``tour_m`` (按 :func:`robotdog.solver.sweeper4.order_points` 排序);
* 每个源被多少个补扫点"看得见" (``|Q-P| ≤ R`` 且 Q 落在 ψ±90° 扇区内);
* **自适应测量**的动作数: 每个点只测"从未见过"的频道 + "见过但还没拿到第二条
  够长基线"的频道 —— 这正是新策略要花的检测预算, 用来在改动策略前先比较点集。

用法::

    python tools/probes/q4_design.py --seeds 9500-9999
    python tools/probes/q4_design.py --seeds 9500-9999 --spacing 1500 --edge 1830:8
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.consts import N_CHANNELS  # noqa: E402
from robotdog.solver.scan4 import boundary_ring, hex_points  # noqa: E402
from robotdog.solver.sweeper4 import order_points  # noqa: E402
from robotdog.solver.world import generate_case_q4  # noqa: E402

Vec = Tuple[float, float]
MIN_BASE = 400.0


def tour_len(pts: Sequence[Vec], start: Vec = (0.0, 0.0)) -> float:
    if not pts:
        return 0.0
    tot = 0.0
    cur = start
    for p in pts:
        tot += math.hypot(p[0] - cur[0], p[1] - cur[1])
        cur = p
    return tot


def make_points(spacing: float, edge: Tuple[float, int]) -> List[Vec]:
    pts = list(hex_points(spacing))
    r, n = edge
    if r and n:
        pts += boundary_ring(r, n)
    uniq: List[Vec] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    return order_points(uniq, (0.0, 0.0))


def eval_seed(seed: int, pts: Sequence[Vec], min_obs: int = 4,
              min_base: float = MIN_BASE, max_extra: int = 10) -> Dict[str, float]:
    """"每站测**全部**未清频道"策略下的可见性统计。

    检测时间是"每站 20 频道 x 6 s"的固定开销, 与可见性无关 —— 所以设计点集时
    真正要压的是两个量: **站数** 与 **可见站数 < 2 的源数** (后者要专程跑一趟补
    基线, 一次一两百米, 是唯一会炸掉的项)。
    """
    srcs = generate_case_q4(seed)
    vis: Dict[int, List[Vec]] = {}
    for q in pts:
        for s_ in srcs:
            if math.hypot(q[0] - s_.x, q[1] - s_.y) <= s_.radius_m and s_.in_coverage(q[0], q[1]):
                vis.setdefault(s_.channel, []).append(q)
    cnt = [len(vis.get(s_.channel, ())) for s_ in srcs]
    return {
        "n": float(len(srcs)),
        "zero": float(sum(1 for c in cnt if c == 0)),
        "one": float(sum(1 for c in cnt if c == 1)),
        "two": float(sum(1 for c in cnt if c == 2)),
        "thin3": float(sum(1 for c in cnt if c < 3)),
        "obs": float(sum(cnt)),
        "tour_m": tour_len(pts), "npts": float(len(pts)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9500-9999")
    ap.add_argument("--spacing", type=float, default=1100.0)
    ap.add_argument("--edge", default="1830:12", help="半径:点数, 0:0 = 不加")
    ap.add_argument("--grid", action="append", default=[],
                    help="spacing:edgeR:edgeN (可重复)")
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--min-base", type=float, default=MIN_BASE)
    ap.add_argument("--max-extra", type=int, default=8)
    args = ap.parse_args()

    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))

    specs: List[Tuple[float, Tuple[float, int]]] = []
    if args.grid:
        for g in args.grid:
            sp, er, en = g.split(":")
            specs.append((float(sp), (float(er), int(en))))
    else:
        er, en = args.edge.split(":")
        specs.append((args.spacing, (float(er), int(en))))

    for sp, edge in specs:
        pts = make_points(sp, edge)
        rows = [eval_seed(sd, pts, args.min_obs, args.min_base, args.max_extra)
                for sd in seeds]
        tot_n = sum(r["n"] for r in rows)

        ncase = float(len(rows))
        print("sp=%-5.0f edge=%-9s npt=%2d tour=%6.0f | 每局: 未见%5.2f 单见%5.2f 双见%5.2f "
              "薄3%5.2f 均观测%5.2f | 检测%5.0f s 总%5.0f s"
              % (sp, "%g:%d" % edge, len(pts), statistics.mean(r["tour_m"] for r in rows),
                 sum(r["zero"] for r in rows)/ncase, sum(r["one"] for r in rows)/ncase,
                 sum(r["two"] for r in rows)/ncase, sum(r["thin3"] for r in rows)/ncase,
                 sum(r["obs"] for r in rows)/ncase,
                 len(pts)*20*6.0, len(pts)*20*6.0 + statistics.mean(r["tour_m"] for r in rows)/5.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
