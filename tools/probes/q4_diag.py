# -*- coding: utf-8 -*-
"""问题4 单局细粒度诊断: 每个频道的观测数 / 交会误差 / 清除动作来自哪个阶段。

服务于**留档的旧版规划器** ``robotdog.solver.sweeper4_legacy`` (覆盖巡游 + 双基线
交会 + 残余阶段)。当前发布策略 ``sweeper4`` 是 opt1 的在线路线重优化方案, 它的
走线分解/归因工具在 ``Desktop/opt1/`` 下 (``analyze.py`` / ``ladder.py`` / ``_t*.py``),
结论见该目录的 ``REPORT.md``。

用法::

    python tools/probes/q4_diag.py --seeds 9500 9501 --set measure_mode=adaptive
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.solver import sweeper4_legacy as sw  # noqa: E402
from robotdog.solver.world import World  # noqa: E402


class Diag(sw.Solver4):
    def __init__(self, ctx, cfg):
        super().__init__(ctx, cfg)
        self.ev: List[Any] = []
        self.fix_fail: Dict[int, int] = {}
        self.phase_of_clear: Dict[str, int] = {}

    def measure(self, x, y, ch):
        r = super().measure(x, y, ch)
        self.ev.append(("m", self.phase, ch, r.result, round(x, 1), round(y, 1)))
        return r

    def clear(self, x, y, ch):
        r = super().clear(x, y, ch)
        self.ev.append(("c", self.phase, ch, r.result, round(x, 1), round(y, 1)))
        self.phase_of_clear[self.phase] = self.phase_of_clear.get(self.phase, 0) + 1
        return r

    def fix(self, ch):
        r = super().fix(ch)
        if r is None:
            self.fix_fail[ch] = self.fix_fail.get(ch, 0) + 1
        return r


def run_one(seed: int, cfg, verbose: bool = True) -> Dict[str, Any]:
    w = World(seed=seed, problem=4)
    ctx = sw._Ctx(w, cfg)
    s = Diag(ctx, cfg)
    s.solve()
    truth = {t["channel"]: t for t in w.truth_q4()}
    rows = []
    for s_ in w.sources:
        ch = s_.channel
        obs = s.obs.get(ch, [])
        fx = s.source_fix(ch)
        acc = s.fix(ch)
        err = (math.hypot(fx[0] - s_.x, fx[1] - s_.y) if fx else float("nan"))
        sig = s.fix_sigma(ch)
        n_clear = sum(1 for e in s.ev if e[0] == "c" and e[2] == ch)
        rows.append(dict(ch=ch, kind=s_.kind, R=round(s_.radius_m), n_obs=len(obs),
                         acc=(acc is not None), err=round(err, 1) if fx else None,
                         sig=None if sig == float("inf") else round(sig, 1),
                         n_clear=n_clear, alive=s_.alive, probes=s.probe_n.get(ch, 0),
                         rho=round(math.hypot(s_.x, s_.y))))
    if verbose:
        print("  seed=%d n=%d virtual=%.0f moved=%.0f measures=%d clears=%d"
              % (seed, w.n_sources, w.virtual_t, w.moved_m, w.measures, w.clears))
        print("    %-3s %-11s %5s %5s %6s %6s %6s %6s %6s"
              % ("ch", "kind", "R", "nobs", "probe", "err", "sigma", "nclr", "alive"))
        for r in rows:
            print("    %-3d %-11s %5d %5d %6d %6s %6s %6d %6s"
                  % (r["ch"], r["kind"], r["R"], r["n_obs"], r["probes"],
                     r["err"], r["sig"], r["n_clear"], r["alive"]))
        print("    清除阶段分布:", s.phase_of_clear, " fix失败次数:", s.fix_fail)
        print("    阶段: scan %.0fs/%.0fm  fix %.0fs/%.0fm  tail %.0fs/%.0fm"
              % (s.t_scan, s.d_scan, s.t_fix, s.d_fix, s.t_tail, s.d_tail))
    return dict(rows=rows, w=w, s=s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="9500-9502")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--watch", type=int, default=0)
    args = ap.parse_args()
    from tools.eval_q4 import parse_seeds
    seeds = parse_seeds(args.seeds)
    cfg = sw.build_cfg()
    for kv in args.set:
        k, _, v = kv.partition("=")
        from tools.eval_q4 import parse_overrides
        for kk, vv in parse_overrides(["%s=%s" % (k, v)]).items():
            setattr(cfg, kk, vv)
    for sd in seeds:
        out = run_one(sd, cfg)
        if args.watch:
            w = out["w"]
            tr = {t["channel"]: t for t in w.truth_q4()}[args.watch]
            print("    --- ch=%d 真值 (%.1f, %.1f) kind=%s R=%.0f ---"
                  % (args.watch, tr["x"], tr["y"], tr["kind"], tr["radius_m"]))
            for e in out["s"].ev:
                if e[2] != args.watch:
                    continue
                d = math.hypot(e[4] - tr["x"], e[5] - tr["y"])
                print("      %s %-5s %-12s (%7.1f,%7.1f) d=%6.1f" % (e[0], e[1], e[3], e[4], e[5], d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
