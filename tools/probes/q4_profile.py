# -*- coding: utf-8 -*-
"""问题4 单局剖析: 把虚拟时间拆到"扫描 / 交会清除 / 残余"三段, 并给出动作明细。

服务于**留档的旧版规划器** ``robotdog.solver.sweeper4_legacy``; 当前发布策略
``sweeper4`` (opt1 在线路线重优化) 的阶段分解由 ``run_sweeper4`` 写进
``world.q4_stages`` (survey/clear/hunt 三段 + 重规划与中继计数)。

用法::

    python tools/probes/q4_profile.py --seeds 9500-9509 --spacing 900
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.solver import sweeper4_legacy as S     # noqa: E402
from robotdog.solver.world import World              # noqa: E402


class Probe(S.Solver4):
    """只记录, 不改策略。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.stage_time: Dict[str, float] = {}
        self.stage_move: Dict[str, float] = {}
        self.stage_actions: Dict[str, int] = {}
        self.stage_clears: Dict[str, int] = {}
        self._last_t = 0.0
        self._last_m = 0.0
        self._last_a = 0
        #: 每次"交会清除"的移动距离: ch -> 米
        self.fix_travel: Dict[int, float] = {}
        self.fix_clear_count: Dict[int, int] = {}
        self.fix_time: Dict[int, float] = {}
        self.fix_attempts: Dict[int, int] = {}
        self.tail_move: Dict[int, float] = {}
        self.tail_time: Dict[int, float] = {}
        self._entry_moved: Optional[float] = None
        self._entry_ch: Optional[int] = None
        self._micro_calls = 0

    def _acc(self) -> None:
        dt = self.ctx.w.virtual_t - self._last_t
        dm = self.ctx.w.moved_m - self._last_m
        da = self.ctx.actions - self._last_a
        self._last_t = self.ctx.w.virtual_t
        self._last_m = self.ctx.w.moved_m
        self._last_a = self.ctx.actions
        self.stage_time[self.phase] = self.stage_time.get(self.phase, 0.0) + dt
        self.stage_move[self.phase] = self.stage_move.get(self.phase, 0.0) + dm
        self.stage_actions[self.phase] = self.stage_actions.get(self.phase, 0) + da

    def measure(self, x, y, ch):
        self._acc()
        return super().measure(x, y, ch)

    def clear(self, x, y, ch):
        self._acc()
        self.stage_clears[self.phase] = self.stage_clears.get(self.phase, 0) + 1
        return super().clear(x, y, ch)

    def _close_in(self, ch, tx, ty):
        m0, c0, t0 = self.ctx.w.moved_m, self.ctx.w.clears, self.ctx.w.virtual_t
        ok = super()._close_in(ch, tx, ty)
        self.fix_travel[ch] = self.fix_travel.get(ch, 0.0) + (self.ctx.w.moved_m - m0)
        self.fix_clear_count[ch] = self.fix_clear_count.get(ch, 0) + (self.ctx.w.clears - c0)
        self.fix_time[ch] = self.fix_time.get(ch, 0.0) + (self.ctx.w.virtual_t - t0)
        self.fix_attempts[ch] = self.fix_attempts.get(ch, 0) + 1
        return ok

    def _second_baseline(self, ch):
        m0, t0 = self.ctx.w.moved_m, self.ctx.w.virtual_t
        ok = super()._second_baseline(ch)
        self.tail_move[ch] = self.tail_move.get(ch, 0.0) + (self.ctx.w.moved_m - m0)
        self.tail_time[ch] = self.tail_time.get(ch, 0.0) + (self.ctx.w.virtual_t - t0)
        return ok

    def _micro_at(self, x, y, ch):
        self._micro_calls += 1
        return super()._micro_at(x, y, ch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9500-9509")
    ap.add_argument("--point-set", default="hex")
    ap.add_argument("--spacing", type=float, default=900.0)
    ap.add_argument("--edge-r", type=float, default=1900.0)
    ap.add_argument("--edge-n", type=int, default=12)
    ap.add_argument("--fix-pass", action="store_true", default=True)
    ap.add_argument("--fix-hop", type=float, default=600.0)
    ap.add_argument("--detail", type=int, default=0)
    a = ap.parse_args()
    seeds: List[int] = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    cfg = S.build_cfg(point_set=a.point_set, spacing=a.spacing, edge_ring=(a.edge_r, a.edge_n),
                      measure_mode="all", pursue_near_m=0.0, fix_pass=a.fix_pass,
                      fix_pass_hop_m=a.fix_hop)
    agg: Dict[str, List[Tuple[float, float, int, int]]] = {}
    tot_fix_travel = 0.0
    tot_fix_clears = 0
    tot_fix_n = 0
    tot_src_dist = 0.0
    tot_src_n = 0
    tot_micro = 0
    tot_tail = 0
    tot_fix_moves: List[float] = []
    tot_fix_times: List[float] = []
    tot_tail_moves: List[float] = []
    tot_tail_times: List[float] = []
    for sd in seeds:
        w = World(seed=sd, problem=4, **S.world_kwargs())
        ctx = S._Ctx(w, cfg)
        p = Probe(ctx, cfg)
        p.solve()
        # 真值归因
        det = {s.channel: None for s in w.sources}
        for s in w.sources:
            n_obs = len(p.obs.get(s.channel, []))
            if n_obs:
                dmin = min(math.hypot(o[0][0] - s.x, o[0][1] - s.y) for o in p.obs[s.channel])
                det[s.channel] = (n_obs, dmin)
        missed = [s for s in w.sources if s.alive]
        print("seed %d: 清除 %d/%d  虚拟 %6.0f s (%.0f s/源)  移动 %6.0f m  "
              "检测 %3d  清除 %3d  漏 %d"
              % (sd, w.cleared_count, w.n_sources, w.virtual_t,
                 w.virtual_t / max(w.cleared_count, 1), w.moved_m, w.measures, w.clears,
                 len(missed)))
        for k in ("scan", "fix", "tail"):
            agg.setdefault(k, []).append(
                (p.stage_time.get(k, 0.0), p.stage_move.get(k, 0.0),
                 p.stage_actions.get(k, 0), p.stage_clears.get(k, 0)))
        for ch, tn in p.fix_travel.items():
            tot_fix_travel += tn
            tot_fix_n += 1
            tot_fix_moves.append(tn)
            tot_fix_times.append(p.fix_time.get(ch, 0.0))
        for ch, tn in p.tail_move.items():
            tot_tail_moves.append(tn)
            tot_tail_times.append(p.tail_time.get(ch, 0.0))
        tot_fix_clears += sum(p.fix_clear_count.values())
        tot_micro += p._micro_calls
        if a.detail:
            print("  ch  观测  最近观测距源  交会移动  清除  耗时s  尝试  真值")
            for s in w.sources:
                dd = det[s.channel]
                print("  %-3d %4s  %10s  %8.0f  %4d  %5.0f  %4d  %-8s |P|=%6.0f R=%6.0f %s"
                      % (s.channel, dd[0] if dd else 0,
                         "%.0f" % dd[1] if dd else "-",
                         p.fix_travel.get(s.channel, 0.0),
                         p.fix_clear_count.get(s.channel, 0),
                         p.fix_time.get(s.channel, 0.0),
                         p.fix_attempts.get(s.channel, 0),
                         "已清除" if not s.alive else "**漏**",
                         math.hypot(s.x, s.y), s.radius_m, s.kind))

    print("\n%-6s %10s %10s %8s %8s" % ("阶段", "时间s", "移动m", "动作", "清除"))
    for k in ("scan", "fix", "tail"):
        if k not in agg:
            continue
        n = len(agg[k])
        print("%-6s %10.0f %10.0f %8.1f %8.1f"
              % (k, sum(x[0] for x in agg[k]) / n, sum(x[1] for x in agg[k]) / n,
                 sum(x[2] for x in agg[k]) / n, sum(x[3] for x in agg[k]) / n))
    print("\n交会清除: %d 个源, 平均移动 %.0f m, 平均清除次数 %.1f"
          % (tot_fix_n, tot_fix_travel / max(tot_fix_n, 1),
             tot_fix_clears / max(tot_fix_n, 1)))
    for lbl, mp_, tp_ in (("交会", tot_fix_moves, tot_fix_times), ("残余", tot_tail_moves, tot_tail_times)):
        if mp_:
            print("%s: %d 个源, 平均移动 %.0f m, 平均耗时 %.0f s" % (lbl, len(mp_), sum(mp_)/len(mp_), sum(tp_)/len(tp_)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
