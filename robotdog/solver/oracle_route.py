"""问题3 的"理想路线"下界探针: 假设**路线已知 (真值 TSP)**, 但检测/定位仍走真实物理。

目的: 判断 200 s/源 是否可达, 以及差距到底在"路线"还是在"信息"。

用法:
    python -m robotdog.solver.oracle_route --seeds 9500-9549
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Any, Dict, List, Sequence, Tuple

from ..consts import CLEAR_FIRE_S, CLEAR_LOCATE_S, DETECT_DURATION_S, SPEED_MPS
from .world import World


def tsp_order(pts: Sequence[Tuple[float, float]], start: Tuple[float, float] = (0.0, 0.0),
              iters: int = 60) -> List[int]:
    n = len(pts)
    if n == 0:
        return []
    left = set(range(n))
    cur = start
    seq: List[int] = []
    while left:
        j = min(left, key=lambda i: math.hypot(pts[i][0] - cur[0], pts[i][1] - cur[1]))
        seq.append(j)
        left.discard(j)
        cur = pts[j]

    def L(s: Sequence[int]) -> float:
        tot = math.hypot(pts[s[0]][0] - start[0], pts[s[0]][1] - start[1])
        for a, b in zip(s, s[1:]):
            tot += math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
        return tot

    improved = True
    while improved:
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                trial = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                if L(trial) < L(seq) - 1e-9:
                    seq, improved = trial, True
    return seq


def run_oracle(seed: int, mode: str = "route+2meas") -> Dict[str, Any]:
    """mode:
      'tsp+1'      理想: 已知真值, 每源 1 次检测 + 1 次清除 (纯下界)
      'tsp+2'      已知路线, 每源在"路上 ~1100 m 处"和"垂直基线点"各测一次后清除
      'tsp+2on'    已知路线, 两次检测都放在路上 (不绕路, 只验证信息够不够)
    """
    w = World(seed=seed)
    pts = [(s.x, s.y) for s in w.sources]
    chans = [s.channel for s in w.sources]
    order = tsp_order(pts)
    for k in order:
        x, y, c = pts[k][0], pts[k][1], chans[k]
        pos = (w.x, w.y)
        if mode == "tsp+1":
            p1 = _toward(pos, (x, y), min(1050.0, math.hypot(x - pos[0], y - pos[1])))
            w.measure(p1[0], p1[1], c)
            w.clear(x, y, c)
            continue
        # 第一次检测: 在通往源的路上 ~1050 m 处 (在 1000~1500 的有效半径内)
        d = math.hypot(x - pos[0], y - pos[1])
        off = min(1050.0, d)
        p1 = _toward(pos, (x, y), d - off)
        w.measure(p1[0], p1[1], c)
        if mode == "tsp+2on":
            p2 = _toward(pos, (x, y), max(0.0, d - 450.0))
        else:
            # 第二次检测: 垂直于视线偏移, 形成基线
            ux, uy = (x - p1[0]) / max(off, 1e-6), (y - p1[1]) / max(off, 1e-6)
            px, py = -uy, ux
            p2 = (p1[0] + px * 420.0, p1[1] + py * 420.0)
        tr = w.belief.tracks[c - 1]
        if not tr.is_measured(p2):
            w.measure(p2[0], p2[1], c)
        # 用信念估计去清除 (真值只用于选路线, 不用于定位)
        tr = w.belief.tracks[c - 1]
        est = tr.est if tr.est is not None else (x, y)
        info = w.clear(est[0], est[1], c)
        if info.result != "success":
            w.clear(x, y, c)
    return {"seed": seed, "cleared": w.cleared_count, "total": w.n_sources,
            "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
            "avg_clear_s": w.virtual_t / max(w.cleared_count, 1)}


def _toward(a: Tuple[float, float], b: Tuple[float, float], dist_from_a: float) -> Tuple[float, float]:
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    if d < 1e-6:
        return a
    f = dist_from_a / d
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--modes", nargs="*", default=["tsp+1", "tsp+2on", "tsp+2"])
    args = ap.parse_args()
    seeds: List[int] = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    for mode in args.modes:
        rows = [run_oracle(sd, mode) for sd in seeds]
        print("%-9s 清除率 %.4f 总虚拟 %6.0f 平均定位清除 %6.1f s 移动 %6.0f 检测 %5.1f"
              % (mode, sum(r["cleared"] for r in rows) / sum(r["total"] for r in rows),
                 statistics.mean(r["virtual_s"] for r in rows),
                 statistics.mean(r["avg_clear_s"] for r in rows),
                 statistics.mean(r["moved_m"] for r in rows),
                 statistics.mean(r["measures"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
