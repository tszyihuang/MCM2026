"""移动下界分解: 真实策略的移动 vs "已知真值 + 最优巡回" 的移动 (集成者用).

回答一个只有集成者需要回答的问题: **11314 m/局 里有多少是"必须走的", 多少是策略开销?**
不涉及任何策略修改, 只用真值做离线下界计算 (纯诊断, 不进入策略)。

三个口径:
  route_opt      真值点上的精确 TSP (Held-Karp, n<=16) —— 起点 (0,0) 出发, 无返回
  route_nn2opt   最近邻 + 2-opt (与策略内 ``_tour_first`` 同族, 用于看启发式差多少)
  policy         发布版策略实际走的

用法::

    python tools\move_bound.py --seeds 9500-9549
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.consts import SPEED_MPS  # noqa: E402
from robotdog.solver.eval import parse_seeds  # noqa: E402

Vec = Tuple[float, float]


def tour_len(pts: Sequence[Vec], start: Vec = (0.0, 0.0)) -> float:
    tot = math.hypot(pts[0][0] - start[0], pts[0][1] - start[1])
    for a, b in zip(pts, pts[1:]):
        tot += math.hypot(b[0] - a[0], b[1] - a[1])
    return tot


def tsp_optimal(pts: Sequence[Vec], start: Vec = (0.0, 0.0), limit: int = 17) -> float:
    """Held-Karp 精确解 (开放路径, 从 start 出发)。n > limit 时退化为 2-opt。"""
    n = len(pts)
    if n == 0:
        return 0.0
    if n > limit:
        return tour_nn2opt(pts, start)
    full = 1 << n
    INF = float("inf")
    # dp[mask][j] = 从 start 出发, 访问 mask 中的点, 最后停在 j 的最短长度
    dp = [[INF] * n for _ in range(full)]
    for j in range(n):
        dp[1 << j][j] = math.hypot(pts[j][0] - start[0], pts[j][1] - start[1])
    for mask in range(full):
        row = dp[mask]
        for j in range(n):
            cur = row[j]
            if cur == INF:
                continue
            for k in range(n):
                if mask & (1 << k):
                    continue
                nm = mask | (1 << k)
                d = cur + math.hypot(pts[k][0] - pts[j][0], pts[k][1] - pts[j][1])
                if d < dp[nm][k]:
                    dp[nm][k] = d
    return min(dp[full - 1])


def tour_nn2opt(pts: Sequence[Vec], start: Vec = (0.0, 0.0)) -> float:
    n = len(pts)
    if n == 0:
        return 0.0
    left = set(range(n))
    seq: List[int] = []
    cur = start
    while left:
        j = min(left, key=lambda i: math.hypot(pts[i][0] - cur[0], pts[i][1] - cur[1]))
        seq.append(j)
        left.discard(j)
        cur = pts[j]
    best = tour_len([pts[i] for i in seq], start)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                L = tour_len([pts[k] for k in trial], start)
                if L < best - 1e-9:
                    seq, best, improved = trial, L, True
    return best


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="移动下界分解")
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)

    from tools.candidates import eval_parallel

    base = eval_parallel("robotdog.solver.sweeper", seeds, args.jobs)
    from robotdog.solver.world import World

    rows: List[Dict[str, float]] = []
    for r in base["rows"]:
        w = World(seed=r["seed"])
        pts = [(s.x, s.y) for s in w.sources]
        opt = tsp_optimal(pts)
        nn = tour_nn2opt(pts)
        rows.append({"seed": r["seed"], "n": len(pts), "policy": r["moved_m"],
                     "opt": opt, "nn2opt": nn})

    def mean(k: str) -> float:
        return statistics.mean(r[k] for r in rows)

    print("%d 局 | 平均真值源数 %.1f" % (len(rows), mean("n")))
    print()
    print("%-26s %10s %10s" % ("口径", "移动 m/局", "折合秒/局"))
    for k, label in (("opt", "真值最优巡回 (Held-Karp)"),
                     ("nn2opt", "真值最近邻+2-opt"),
                     ("policy", "发布版策略实际移动")):
        print("%-26s %10.0f %10.0f" % (label, mean(k), mean(k) / SPEED_MPS))
    print()
    excess = mean("policy") - mean("opt")
    print("策略移动 - 最优巡回 = %.0f m = %.0f s/局" % (excess, excess / SPEED_MPS))
    print("策略移动 / 最优巡回  = %.2fx" % (mean("policy") / max(mean("opt"), 1e-9)))
    print()
    per_src = mean("policy") / mean("n")
    print("每源移动: 策略 %.0f m (%.0f s) | 最优巡回 %.0f m (%.0f s) | 差 %.0f m (%.0f s)"
          % (per_src, per_src / SPEED_MPS,
             mean("opt") / mean("n"), mean("opt") / mean("n") / SPEED_MPS,
             per_src - mean("opt") / mean("n"),
             (per_src - mean("opt") / mean("n")) / SPEED_MPS))
    print()
    worst = sorted(rows, key=lambda r: -(r["policy"] / max(r["opt"], 1e-9)))[:8]
    print("策略/最优 比值最大的 8 局 (路线最差的局):")
    for r in worst:
        print("  seed %d n=%2d policy %6.0f m opt %6.0f m ratio %.2f"
              % (r["seed"], r["n"], r["policy"], r["opt"],
                 r["policy"] / max(r["opt"], 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
