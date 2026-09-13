# -*- coding: utf-8 -*-
"""问题4 失败归因: 对若干局打印"漏掉的源"的真值与策略所见。

服务于**留档的旧版规划器** ``robotdog.solver.sweeper4_legacy``。当前发布策略
``sweeper4`` (opt1 在线路线重优化) 在 500+1000 局标定/留出集上 0 漏源, 若仍要
归因单局, 可直接看该策略写进 ``world.q4_stages`` 的三段计时与 ``q4_hunt_found``。

用法::

    python tools/probes/q4_missed.py --seeds 9500-9569 --spacing 900 --point-set hex
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.solver import sweeper4_legacy as S     # noqa: E402
from robotdog.solver.world import World              # noqa: E402


class Probe(S.Solver4):
    """记录每个频道的观测与清除尝试, 用于归因 (只读, 不改策略)。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.log: List[str] = []
        self.clear_actions: Dict[int, int] = {}

    def measure(self, x, y, ch):                     # noqa: D102
        r = super().measure(x, y, ch)
        if r.result != "no_signal":
            self.log.append("M ch%-2d (%7.1f,%7.1f) %s" % (ch, x, y, r.result))
        return r

    def clear(self, x, y, ch):                       # noqa: D102
        self.clear_actions[ch] = self.clear_actions.get(ch, 0) + 1
        r = super().clear(x, y, ch)
        if r.result == "success":
            self.log.append("C ch%-2d (%7.1f,%7.1f) SUCCESS" % (ch, x, y))
        return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9500-9569")
    ap.add_argument("--spacing", type=float, default=900.0)
    ap.add_argument("--point-set", default="hex")
    ap.add_argument("--detail", type=int, default=3, help="打印几局的完整观测日志")
    ap.add_argument("--max-detail", type=int, default=400)
    a = ap.parse_args()

    seeds: List[int] = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    cfg = S.build_cfg(point_set=a.point_set, spacing=a.spacing, edge_ring=(0, 0),
                      measure_mode="all", pursue_near_m=0.0)
    shown = 0
    for sd in seeds:
        w = World(seed=sd, problem=4, **S.world_kwargs())
        ctx = S._Ctx(w, cfg)
        solver = Probe(ctx, cfg)
        solver.solve()
        missed = [s for s in w.sources if s.alive]
        if not missed:
            continue
        print("=== seed %d  cleared %d/%d  virtual %.0f s  moved %.0f m  漏 %d ==="
              % (sd, w.cleared_count, w.n_sources, w.virtual_t, w.moved_m, len(missed)))
        for s in missed:
            seen = [o for o in solver.obs.get(s.channel, [])]
            near_pts = min((math.hypot(o[0][0] - s.x, o[0][1] - s.y) for o in seen),
                           default=float("inf"))
            print("  ch%-2d %-11s pos=(%8.1f,%8.1f) R=%6.1f psi=%s |P|=%6.1f"
                  "  观测 %d 条, 最近观测点距源 %.0f m, 清除尝试 %d"
                  % (s.channel, s.kind, s.x, s.y, s.radius_m,
                     "-" if s.direction_deg is None else "%.1f" % s.direction_deg,
                     math.hypot(s.x, s.y), len(seen), near_pts,
                     solver.clear_actions.get(s.channel, 0)))
        if shown < a.detail:
            shown += 1
            print("  ---- 观测日志 (尾部 %d 条) ----" % a.max_detail)
            for line in solver.log[-a.max_detail:]:
                print("   ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
