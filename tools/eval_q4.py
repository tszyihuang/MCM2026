# -*- coding: utf-8 -*-
"""问题4 策略的统一评测 / 参数研究 (多核并行)。

与其他评测工具的口径完全一致:
  * 清除比例 = Σ已清除 / Σ真值总数
  * 平均定位清除 = 每局 (虚拟总时间 / 该局已清除数), 再对局取平均
  * ``--deploy`` 切换 ``knows_total=False`` (正式测试口径: 看不到真值总数)

**问题4 只有一条路线**:

  ==============  ==========================================  ==========================
  别名             模块                                        路线
  ==============  ==========================================  ==========================
  ``sweeper4``    ``robotdog.solver.sweeper4``            集合覆盖 + 锚点射线 + 边界环
  ==============  ==========================================  ==========================

用法::

    # 完整成绩, 双口径
    python tools/eval_q4.py --planners sweeper4 --seeds 9500-9799 --jobs 8
    python tools/eval_q4.py --planners sweeper4 --seeds 9500-9799 --jobs 8 --deploy

    # 切档: 关掉边界环 = 极速档 (实测 0.99962); 默认开 = 全清档 (1.00000)
    python tools/eval_q4.py --planners sweeper4 --seeds 20000-20999 --jobs 8 --set boundary_ring=False

    # 一组配方横扫
    python tools/eval_q4.py --seeds 9500-9799 --jobs 8 --set boundary_ring_n=36

**规划器可以自带世界装配**: 若模块暴露 ``world_kwargs()``, 它会被透传给 ``World(...)``。
这是必需的 —— ``World(problem=4)`` 默认装 ``Belief4`` (ψ 分箱), 而 ``sweeper4``
不做贝叶斯推断, 要的是一个空信念。
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: ``--planners`` 的别名表。问题4 只有 ``sweeper4`` 一条路线。
PLANNER_ALIASES: Dict[str, str] = {
    "sweeper4": "robotdog.solver.sweeper4",
}


def _run(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    mod = importlib.import_module(payload["module"])
    from robotdog.solver.world import World
    cfg = mod.build_cfg() if hasattr(mod, "build_cfg") else None
    for k, v in payload["overrides"].items():
        setattr(cfg, k, v)
    fn = getattr(mod, "run_sweeper4", None) or getattr(mod, "run_candidate", None) \
        or getattr(mod, "run_sweeper")
    # 规划器可以自带世界装配 (信念类因路线而异; 见模块 docstring)
    wk = mod.world_kwargs() if hasattr(mod, "world_kwargs") else {}
    rows = []
    for sd in payload["seeds"]:
        w = World(seed=sd, problem=payload["problem"], **wk)
        if payload["deploy"]:
            fn(w, cfg=cfg, knows_total=False)
        else:
            fn(w, cfg)
        rows.append({
            "seed": sd, "cleared": w.cleared_count, "total": w.n_sources,
            "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
            "clears": w.clears, "failed_clears": w.failed_clears,
            "ndir": w.n_directional,
            "hunt_found": getattr(w, "q4_hunt_found", 0),
            "n_dir_cleared": sum(1 for s in w.sources if not s.is_omni and not s.alive),
        })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    return {
        "n": len(rows), "ratio": clr / max(tot, 1), "cleared": clr, "total": tot,
        "full": sum(1 for r in rows if r["cleared"] == r["total"]),
        "avg_clear": statistics.mean(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "median_clear": statistics.median(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "virtual": statistics.mean(r["virtual_s"] for r in rows),
        "moved": statistics.mean(r["moved_m"] for r in rows),
        "measures": statistics.mean(r["measures"] for r in rows),
        "failed_clears": statistics.mean(r["failed_clears"] for r in rows),
        "ndir": sum(r["ndir"] for r in rows),
        "ndir_cleared": sum(r["n_dir_cleared"] for r in rows),
        "hunt_found": sum(r["hunt_found"] for r in rows),
    }


def run_config(module: str, seeds: List[int], overrides: Dict[str, Any],
               jobs: int = 8, problem: int = 4, deploy: bool = False) -> Dict[str, Any]:
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [dict(module=module, seeds=ch, overrides=overrides, problem=problem,
                     deploy=deploy) for ch in chunks if ch]
    if len(payloads) == 1:
        rows = _run(payloads[0])
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_run, payloads)
        rows = [r for p in parts for r in p]
    rows.sort(key=lambda r: r["seed"])
    return {"summary": summarize(rows), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="问题4 策略统一评测")
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--modules", nargs="+", default=["robotdog.solver.sweeper4"])
    ap.add_argument("--planners", nargs="+", default=None, choices=sorted(PLANNER_ALIASES),
                    help="按别名选策略 (sweeper4); 与 --modules 二选一, "
                         "传了 --planners 就以它为准")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--deploy", action="store_true", help="部署口径 (knows_total=False)")
    ap.add_argument("--problem", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置: k=v [k=v ...]")
    ap.add_argument("--sweep", default="", help="单旋钮扫描: name=v1,v2,...")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.planners:
        args.modules = [PLANNER_ALIASES[p] for p in args.planners]

    seeds: List[int] = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    jobs = max(1, min(args.jobs, len(seeds)))

    overrides: Dict[str, Any] = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        if v in ("True", "False"):
            overrides[k] = (v == "True")
        elif any(ch in v for ch in ".eE"):
            overrides[k] = float(v)
        else:
            overrides[k] = int(v)

    jobs = max(1, min(args.jobs, len(seeds)))
    t0 = time.time()
    if args.sweep:
        name, _, vals = args.sweep.partition("=")
        results = []
        for v in vals.split(","):
            ov = dict(overrides)
            ov[name] = float(v)
            s = run_config(args.modules[0], seeds, ov, jobs, args.problem, args.deploy)
            results.append((ov, s["summary"]))
        print("%d 局 | 扫描 %s | 墙钟 %.0f s" % (len(seeds), name, time.time() - t0))
        print("  %-40s %8s %10s %12s %10s %8s" %
              ("覆盖", "清除率", "全清", "平均定位清除", "移动m", "检测"))
        for ov, s in results:
            print("  %-40s %8.4f %6d/%-3d %12.1f %10.0f %8.1f" %
                  (str(ov), s["ratio"], s["full"], s["n"], s["avg_clear"],
                   s["moved"], s["measures"]))
        return 0

    report: Dict[str, Any] = {"seeds": seeds, "results": {}}
    for m in args.modules:
        r = run_config(m, seeds, overrides, jobs, args.problem, args.deploy)
        s = r["summary"]
        report["results"][m] = r
        print("%-34s 清除率 %.4f  全清 %3d/%-3d  平均定位清除 %6.1f s (中位 %6.1f)  "
              "总虚拟 %6.0f s  移动 %6.0f m  检测 %5.1f  失败清除 %.2f  定向源 %d/%d  收尾+%d"
              % (m.split(".")[-1], s["ratio"], s["full"], s["n"], s["avg_clear"],
                 s["median_clear"], s["virtual"], s["moved"], s["measures"],
                 s["failed_clears"], s["ndir_cleared"], s["ndir"], s["hunt_found"]))
    print("口径: %s | %d 局 | 墙钟 %.0f s" %
          ("部署 knows_total=False" if args.deploy else "进程内 knows_total=True",
           len(seeds), time.time() - t0))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        print("明细 ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
