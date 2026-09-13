# -*- coding: utf-8 -*-
"""问题4 策略的统一评测 / 参数研究 (多核并行)。

口径与其他评测工具完全一致:
  * 清除比例 = Σ已清除 / Σ真值总数
  * 平均定位清除 = 每局 (虚拟总时间 / 该局已清除数), 再对局取平均
     —— 与模拟器报告里的 ``avg_clear_duration_s`` 同口径
  * ``--deploy`` 切换 ``knows_total=False`` (正式测试口径: 看不到真值总数)

用法::

    python tools/eval_q4.py --seeds 9500-9799 --jobs 16
    python tools/eval_q4.py --seeds 9500-9799 --jobs 16 --deploy
    python tools/eval_q4.py --seeds 20000-20999 --jobs 16 --set boundary_ring=False

规划器可以自带世界装配: 若模块暴露 ``world_kwargs()``, 它会被透传给 ``World(...)``。
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

#: ``--planners`` 的别名表。
PLANNER_ALIASES: Dict[str, str] = {
    "sweeper4": "robotdog.solver.sweeper4",
}


def run_rows(module: str, seeds: List[int], overrides: Dict[str, Any],
             problem: int = 4, deploy: bool = False,
             trace_seeds: Tuple[int, ...] = ()) -> List[Dict[str, Any]]:
    mod = importlib.import_module(module)
    from robotdog.solver.world import World
    cfg = mod.build_cfg() if hasattr(mod, "build_cfg") else None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    fn = (getattr(mod, "run_sweeper4", None) or getattr(mod, "run_candidate", None)
          or getattr(mod, "run_sweeper"))
    wk = mod.world_kwargs() if hasattr(mod, "world_kwargs") else {}
    rows = []
    for sd in seeds:
        w = World(seed=sd, problem=problem, **wk)
        tr = sd in trace_seeds
        try:
            fn(w, cfg=cfg, knows_total=not deploy)
        except TypeError:
            fn(w, cfg=cfg)
        row = {
            "seed": sd, "cleared": w.cleared_count, "total": w.n_sources,
            "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
            "clears": w.clears, "failed_clears": w.failed_clears,
            "ndir": w.n_directional,
            "n_dir_cleared": sum(1 for s in w.sources if not s.is_omni and not s.alive),
            "t_travel": w.time_travel, "t_switch": w.time_switch,
            "t_detect": w.time_detect, "t_clear": w.time_clear,
            "hunt_found": getattr(w, "q4_hunt_found", 0),
            "stages": getattr(w, "q4_stages", None),
        }
        rows.append(row)
        if tr:
            print("  [trace seed=%d] %s" % (sd, row.get("stages")), flush=True)
    return rows


def _run(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_rows(payload["module"], payload["seeds"], payload["overrides"],
                    payload["problem"], payload["deploy"], payload.get("trace_seeds", ()))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    n = len(rows)
    return {
        "n": n, "ratio": clr / max(tot, 1), "cleared": clr, "total": tot,
        "full": sum(1 for r in rows if r["cleared"] == r["total"]),
        "avg_clear": statistics.mean(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "median_clear": statistics.median(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "worst_clear": max(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "virtual": statistics.mean(r["virtual_s"] for r in rows),
        "moved": statistics.mean(r["moved_m"] for r in rows),
        "measures": statistics.mean(r["measures"] for r in rows),
        "failed_clears": statistics.mean(r["failed_clears"] for r in rows),
        "ndir": sum(r["ndir"] for r in rows),
        "ndir_cleared": sum(r["n_dir_cleared"] for r in rows),
        "hunt_found": sum(r["hunt_found"] for r in rows),
        "t_travel": statistics.mean(r["t_travel"] for r in rows),
        "t_switch": statistics.mean(r["t_switch"] for r in rows),
        "t_detect": statistics.mean(r["t_detect"] for r in rows),
        "t_clear": statistics.mean(r["t_clear"] for r in rows),
        "missed": [r["seed"] for r in rows if r["cleared"] != r["total"]],
    }


def run_config(module: str, seeds: List[int], overrides: Dict[str, Any],
               jobs: int = 8, problem: int = 4, deploy: bool = False,
               trace_seeds: Tuple[int, ...] = ()) -> Dict[str, Any]:
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [dict(module=module, seeds=ch, overrides=overrides, problem=problem,
                     deploy=deploy, trace_seeds=trace_seeds) for ch in chunks if ch]
    if len(payloads) == 1:
        rows = _run(payloads[0])
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_run, payloads)
        rows = [r for p in parts for r in p]
    rows.sort(key=lambda r: r["seed"])
    return {"summary": summarize(rows), "rows": rows}


def parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def parse_overrides(items: List[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for kv in items:
        k, _, v = kv.partition("=")
        if v in ("True", "False"):
            overrides[k] = (v == "True")
        elif v in ("None", ""):
            overrides[k] = None
        elif v.startswith("(") or v.startswith("["):
            import ast
            try:
                overrides[k] = ast.literal_eval(v)
            except Exception:
                overrides[k] = v
        else:
            try:
                overrides[k] = int(v)
            except ValueError:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    overrides[k] = v
    return overrides


def main() -> int:
    ap = argparse.ArgumentParser(description="问题4 策略统一评测")
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--modules", nargs="+", default=None)
    ap.add_argument("--planners", nargs="+", default=["sweeper4"],
                    choices=sorted(PLANNER_ALIASES))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--deploy", action="store_true", help="部署口径 (knows_total=False)")
    ap.add_argument("--problem", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], help="覆盖配置: k=v [k=v ...]")
    ap.add_argument("--trace", nargs="*", default=[], help="打印这些种子的阶段剖析")
    ap.add_argument("--quiet-rows", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    modules = args.modules or [PLANNER_ALIASES[p] for p in args.planners]
    seeds = parse_seeds(args.seeds)
    jobs = max(1, min(args.jobs, len(seeds)))
    overrides = parse_overrides(args.set)
    trace_seeds = tuple(parse_seeds(",".join(args.trace))) if args.trace else ()

    t0 = time.time()
    report: Dict[str, Any] = {"seeds": seeds, "deploy": args.deploy,
                              "overrides": overrides, "results": {}}
    for m in modules:
        r = run_config(m, seeds, overrides, jobs, args.problem, args.deploy, trace_seeds)
        s = r["summary"]
        report["results"][m] = r
        print("%-14s 清除率 %.4f  全清 %4d/%-4d  平均定位清除 %6.1f s (中位 %6.1f, 最差 %6.1f)"
              "  %s/源 | 虚拟 %6.0f  移动 %6.0f m  检测 %5.1f  空清 %.1f  定向 %d/%d"
              % (m.split(".")[-1], s["ratio"], s["full"], s["n"], s["avg_clear"],
                 s["median_clear"], s["worst_clear"],
                 ("**达标**" if s["avg_clear"] <= 600.0 and s["ratio"] >= 0.98 else "未达标"),
                 s["virtual"], s["moved"], s["measures"], s["failed_clears"],
                 s["ndir_cleared"], s["ndir"]))
        print("        时间分解: 移动 %.0f s | 切换 %.0f s | 检测 %.0f s | 清除 %.0f s"
              % (s["t_travel"], s["t_switch"], s["t_detect"], s["t_clear"]))
        if s["missed"]:
            print("        未全清局 (%d): %s" % (len(s["missed"]), s["missed"][:25]))
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
