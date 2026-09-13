# -*- coding: utf-8 -*-
"""问题4 走线审计（当前策略 = 中继测量 + 在线路线重优化）。

``--module`` 决定审计哪一代规划器：``sweeper4``（当前发布策略）/
``sweeper4_ring18``（上一版 18 站档）/ ``sweeper4_legacy``（更早的覆盖巡游档）。
两代实现的动作入口名字不同（``_Solver.do_measure/try_clear`` 与
``Solver4.measure/clear``），本脚本按模块自动选择（见 :func:`make_classes`）。

真值**只用于离线分析**，绝不进入策略。输出四组论文用数：

1. ``L_actual`` 一局所有动作位移之和；
2. ``L_tour`` 发布站点集骨架的最优开链巡游（静态，随 ``ring_spec`` 变）；
3. ``L_off``（站点 ∪ 真值源）的最优开链巡游 —— **全知上界**；
4. 冗余分解：站序浪费（实际站点拜访折线 − 实际到过站点的最优回路）、
   清除绕行（离站片段实际 vs 用真值算的理想）。

``--oracle`` 再加一档**全知对照**：把真值位置直接塞进 ``self.fix``（仅离线分析），
其余一字不改。它回答"剩余冗余是不是算法侧的锅" —— 若给足信息后只差全知最优一点点，
那么瓶颈在**信息到达时序**（首条示向度 → 可交会定位的延迟），而不在路线优化。

用法::

    python tools/probes/q4_opt1_route.py --seeds 9500-9559 --oracle
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.solver import sweeper4 as S                     # noqa: E402
from robotdog.solver.world import World                       # noqa: E402


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def open_tour_len(pts) -> float:
    """开链巡游长度（与策略内同一套算子：多起点 NN + 2-opt + Or-opt）。

    ``_tour_index`` 按半径去重起手点，点集大时是近似；对站点点集另用
    :func:`best_open_tour_len` 做全起点搜索。
    """
    xy = [(float(p[0]), float(p[1])) for p in pts]
    if not xy:
        return 0.0
    return S._order_len(S._tour_index(xy, (0.0, 0.0), 40), xy, (0.0, 0.0))


def best_open_tour_len(pts) -> float:
    """全起点（每个点各起手一次，含反向扰动）的 2-opt/Or-opt 最优开链巡游。"""
    xy = [(float(p[0]), float(p[1])) for p in pts]
    n = len(xy)
    if n == 0:
        return 0.0
    if n == 1:
        return dist((0.0, 0.0), xy[0])
    best = None
    for s0 in range(n):
        rem = [i for i in range(n) if i != s0]
        cur, order = s0, [s0]
        while rem:
            k = min(rem, key=lambda i: dist(xy[i], xy[cur]))
            rem.remove(k)
            order.append(k)
            cur = k
        for cand in (order, list(reversed(order))):
            o = S._ls_order(cand, xy, (0.0, 0.0), 3)
            L = S._order_len(o, xy, (0.0, 0.0))
            if best is None or L < best:
                best = L
    return best


def ideal_open_path(start, targets, end):
    """从 start 出发依次经过全部 targets（顺序自由）到 end（None = 终点自由）。"""
    if not targets:
        return 0.0 if end is None else dist(start, end)
    if len(targets) <= 6:
        best = None
        for perm in itertools.permutations(range(len(targets))):
            tot = dist(start, targets[perm[0]])
            for k in range(len(perm) - 1):
                tot += dist(targets[perm[k]], targets[perm[k + 1]])
            if end is not None:
                tot += dist(targets[perm[-1]], end)
            if best is None or tot < best:
                best = tot
        return best
    rem = list(range(len(targets)))
    cur, tot = start, 0.0
    while rem:
        k = min(rem, key=lambda i: dist(targets[i], cur))
        rem.remove(k)
        tot += dist(targets[k], cur)
        cur = targets[k]
    if end is not None:
        tot += dist(cur, end)
    return tot


def make_classes(mod):
    """按模块生成"只记录"子类；``oracle=True`` 时再加一档全知对照。

    当前版 ``_Solver`` 的动作入口是 ``do_measure`` / ``try_clear``（返回
    ``(result, raw)`` / ``bool``）；留档的旧版是 ``measure`` / ``clear``
    （返回 ``StepInfo``）。这里按类上有哪个入口自动选，函数体保持同一份。
    """
    base = getattr(mod, "_Solver", None) or mod.Solver4

    if hasattr(base, "do_measure"):          # 当前版: _Solver

        class Trace(base):
            def __init__(self, ctx, cfg=None, truth=None) -> None:
                self.log = []               # (path, x, y, channel, result)
                self._truth = dict(truth or {})
                super().__init__(ctx, None)

            def do_measure(self, c, x, y):
                res, raw = super().do_measure(c, x, y)
                self.log.append(('m', float(x), float(y), int(c), res))
                return res, raw

            def try_clear(self, c, x, y):
                ok = super().try_clear(c, x, y)
                self.log.append(('c', float(x), float(y), int(c),
                                 'success' if ok else 'no_target_in_range'))
                return ok

    else:                                    # 留档版: Solver4

        class Trace(base):
            def __init__(self, ctx, cfg, truth=None) -> None:
                self.log = []               # (path, x, y, channel, result)
                self._truth = dict(truth or {})
                super().__init__(ctx, cfg)

            def measure(self, x, y, c):
                r = super().measure(x, y, c)
                self.log.append(('m', float(x), float(y), int(c), r.result))
                return r

            def clear(self, x, y, c):
                r = super().clear(x, y, c)
                self.log.append(('c', float(x), float(y), int(c), r.result))
                return r

    class Oracle(Trace):
        """全知对照：真值位置直接进 ``self.fix``（**仅离线分析用**）。"""

        def update_fixes(self):
            for ch, (x, y) in self._truth.items():
                if ch not in self.cleared:
                    self.fix[ch] = (float(x), float(y), 0.0)

    return Trace, Oracle


def load_module(name: str):
    import importlib
    return importlib.import_module("robotdog.solver." + name)


def run_one(mod, seed: int, cfg, cls, truth=None):
    w = World(seed=seed, problem=4, **mod.world_kwargs())
    for k in getattr(mod, "PARAMS", {}):        # 旧版规划器的参数在 cfg 里, 无模块级 PARAMS
        if hasattr(cfg, k):
            mod.PARAMS[k] = getattr(cfg, k)
    ctx = mod._Ctx(w, cfg)
    solver = cls(ctx, cfg, truth)
    if hasattr(solver, "run"):
        solver.run()
    else:                                   # 旧版规划器的入口是 solve()
        solver.solve()
    return w, solver


def analyse(w, solver, stations, tol: float = 2.0):
    """把一局的动作序列拆成 L_actual / 站序折线 / 离站片段三段。"""
    log = solver.log
    truth = {s["channel"]: (s["x"], s["y"]) for s in w.truth_q4()}
    sta = [tuple(p) for p in stations]

    prev = (0.0, 0.0)
    steps = []
    for (_path, x, y, ch, res) in log:
        p = (x, y)
        steps.append((prev, p, dist(prev, p),
                      any(dist(p, s) < tol for s in sta), ch, res))
        prev = p

    L_actual = sum(s[2] for s in steps)
    to_station = sum(s[2] for s in steps if s[3])

    # 实际站点拜访顺序（把连续重复的同一站合并）
    visit = [(0.0, 0.0)]
    for (_a, b, _d, is_sta, _c, _r) in steps:
        if is_sta and dist(visit[-1], b) > 1.0:
            visit.append(b)
    visit_order_len = sum(dist(visit[i], visit[i + 1]) for i in range(len(visit) - 1))
    vis_set = visit[1:]

    # 离站往返片段：从"上一个站点动作之后"到"下一个站点动作"之间的整段路
    # （含最后回到站点的那一腿，否则实际值与理想值口径不一致）。
    # 理想值只对"这一段真正清掉的源"算（用真值位置做开放路径）：既不会把
    # "测了一下但没去清"的源算进来，也不会低估必须走的进场路。
    runs = []
    cur = []
    for (a, b, d, is_sta, ch, res) in steps:
        if is_sta:
            if cur:
                runs.append((cur, b))
            cur = []
        else:
            cur.append((a, b, d, ch, res))
    if cur:
        runs.append((cur, None))
    exc_act = exc_ideal = 0.0
    for run, end in runs:
        act = sum(r[2] for r in run)
        if end is not None:
            act += dist(run[-1][1], end)      # 回到下一个站点的那一腿
        chans = []
        for r in run:
            if r[4] == "success" and r[3] not in chans:
                chans.append(r[3])
        tgt = [truth[c] for c in chans if c in truth]
        exc_act += act
        exc_ideal += ideal_open_path(run[0][0], tgt, end)

    srcs = [(s["x"], s["y"]) for s in w.truth_q4()]
    L_off = open_tour_len(sta + srcs)
    L_tour = best_open_tour_len(sta)
    L_tour_vis = best_open_tour_len(vis_set) if len(vis_set) > 1 else 0.0
    return {
        "seed": w.seed, "cleared": w.cleared_count, "total": w.n_sources,
        "vt": w.virtual_t, "avgT": w.virtual_t / max(w.cleared_count, 1),
        "L_actual": L_actual, "to_station": to_station,
        "to_source": L_actual - to_station,
        "L_tour": L_tour, "L_tour_vis": L_tour_vis, "L_off": L_off,
        "visit_order_len": visit_order_len, "n_visits": len(vis_set),
        "exc_act": exc_act, "exc_ideal": exc_ideal, "n_exc": len(runs),
        "measures": w.measures, "clears": w.clears,
    }


def mean(rows, key) -> float:
    return statistics.fmean(r[key] for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="问题4 走线审计")
    ap.add_argument("--seeds", default="9500-9559")
    ap.add_argument("--module", default="sweeper4",
                    choices=("sweeper4", "sweeper4_ring18", "sweeper4_legacy"),
                    help="默认审计当前策略；留档版用于同批对照")
    ap.add_argument("--oracle", action="store_true", help="加跑全知对照")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置 k=v")
    a = ap.parse_args()
    if a.oracle and a.module == "sweeper4_legacy":
        ap.error("--oracle 只对 __Solver/Solver4.update_fixes 形态的规划器有意义")
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    from tools.eval_q4 import parse_overrides
    mod = load_module(a.module)
    cfg = mod.build_cfg()
    for k, v in parse_overrides(a.set).items():
        setattr(cfg, k, v)
    sta = [(float(p[0]), float(p[1])) for p in mod.build_points(cfg)]
    Trace, Oracle = make_classes(mod)

    rows = [analyse(*run_one(mod, sd, cfg, Trace), stations=sta) for sd in seeds]
    n = len(rows)
    print("=== %s（在线）  %d 局 | %d 站 ===" % (a.module, n, len(sta)))
    print("  全清 %d/%d   平均定位清除 %.1f s/源   检测 %.1f 次/局"
          % (sum(1 for r in rows if r["cleared"] == r["total"]), n,
             mean(rows, "avgT"), mean(rows, "measures")))
    print("  L_actual = %8.1f m   (终点在站点上 %.1f / 其余 %.1f)"
          % (mean(rows, "L_actual"), mean(rows, "to_station"), mean(rows, "to_source")))
    print("  L_tour   = %8.1f m   骨架（静态）" % mean(rows, "L_tour"))
    print("  L_off    = %8.1f m   站点 ∪ 真值源 的全知上界" % mean(rows, "L_off"))
    print("  L_actual − L_off = %8.1f m  (%.1f%%)"
          % (mean(rows, "L_actual") - mean(rows, "L_off"),
             100.0 * (mean(rows, "L_actual") - mean(rows, "L_off")) / mean(rows, "L_off")))
    waste = mean(rows, "visit_order_len") - mean(rows, "L_tour_vis")
    print("  站点拜访折线 = %.1f m  实际到过站点的最优回路 = %.1f m  ⇒ 站序浪费 %.1f m"
          % (mean(rows, "visit_order_len"), mean(rows, "L_tour_vis"), waste))
    exc_red = mean(rows, "exc_act") - mean(rows, "exc_ideal")
    print("  离站往返片段 %.1f 段: 实际 %.1f m vs 用真值算的理想 %.1f m ⇒ 绕行冗余 %.1f m"
          % (mean(rows, "n_exc"), mean(rows, "exc_act"), mean(rows, "exc_ideal"), exc_red))
    print("  实际到过站点数 %.1f / %d" % (mean(rows, "n_visits"), len(sta)))

    if a.oracle:
        orows = []
        for sd in seeds:
            w0 = World(seed=sd, problem=4, **S.world_kwargs())
            truth = {s["channel"]: (s["x"], s["y"]) for s in w0.truth_q4()}
            w1, sol = run_one(mod, sd, cfg, Oracle, truth)
            orows.append(analyse(w1, sol, stations=sta))
        print("=== 全知对照（真值位置直接进 fix，仅离线）  %d 局 ===" % len(orows))
        print("  全清 %d/%d   平均定位清除 %.1f s/源   检测 %.1f 次/局"
              % (sum(1 for r in orows if r["cleared"] == r["total"]), len(orows),
                 mean(orows, "avgT"), mean(orows, "measures")))
        gap_on = 100.0 * (mean(rows, "L_actual") - mean(rows, "L_off")) / mean(rows, "L_off")
        gap_or = 100.0 * (mean(orows, "L_actual") - mean(orows, "L_off")) / mean(orows, "L_off")
        print("  全知 L_actual = %8.1f m   L_actual − L_off = %8.1f m (%.1f%%)"
              % (mean(orows, "L_actual"),
                 mean(orows, "L_actual") - mean(orows, "L_off"), gap_or))
        print("  ⇒ 同一套规划器，给足信息后只差全知最优 %.1f%%；在线差 %.1f%%"
              % (gap_or, gap_on))
        print("     ⇒ 差额来自**信息到达时序**（首条示向度 → 可交会定位的延迟），不是路线优化。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
