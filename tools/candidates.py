"""问题3 求解器的批量评测入口 (进程内口径 ``knows_total=True``).

被测模块按 dotted path 给出, 并暴露两个**可选**入口之一:

  1. ``run_candidate(world, cfg=None) -> SweepResult``   —— 直接替换 ``run_sweeper``
  2. ``build_cfg() -> SweepConfig``                       —— 只改参数 (走 ``run_sweeper``)

口径与 ``robotdog.solver.eval`` 完全一致, **同一批种子、同一统计口径**。

> 正式测试口径 (``knows_total=False``) 请用 ``tools/eval_deploy.py`` ——
> 被打分的是那个口径, 两者不可直接比较。

用法::

    # 进程内口径
    python tools/candidates.py --modules robotdog.solver.sweeper --seeds 9500-11499 --jobs 6

    # 打印逐步明细
    python tools/candidates.py --modules robotdog.solver.sweeper --seeds 9500 --trace

指标口径 (与 README §4.1 一致):
  * 清除比例 ratio        = Σ已清除 / Σ真值总数        —— 越大越好, 且是前置目标
  * 全清率   full_rate    = 全清局数 / 局数
  * 平均定位清除 avg_clear_mean = 每局 (虚拟总时间 / 该局已清除数), 再对局取平均
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.eval import fmt, parse_seeds, summarize  # noqa: E402


def run_module_case(module_name: str, seed: int, trace: bool = False) -> Dict[str, Any]:
    import inspect
    from robotdog.solver.sweeper import build_cfg as release_cfg, run_sweeper
    from robotdog.solver.world import World
    mod = importlib.import_module(module_name)
    w = World(seed=seed)
    # 配置必须与发布版一致: 模块没有 build_cfg() 时回退到 sweeper.build_cfg(),
    # 绝不传 cfg=None —— 那会退化成默认 SweepConfig(), 静默换成另一套参数。
    cfg = mod.build_cfg() if hasattr(mod, "build_cfg") else release_cfg()
    fn = getattr(mod, "run_candidate", None) or run_sweeper
    if len(inspect.signature(fn).parameters) >= 2:
        res = fn(w, cfg, trace=trace)
    else:
        res = fn(w, trace=trace)
    return {
        "seed": w.seed, "cleared": w.cleared_count, "total": w.n_sources,
        "ratio": w.cleared_count / w.n_sources,
        "virtual_s": w.virtual_t,
        "avg_clear_s": w.virtual_t / w.cleared_count if w.cleared_count else float("inf"),
        "moved_m": w.moved_m, "measures": w.measures, "clears": w.clears,
        "failed_clears": w.failed_clears, "steps": getattr(res, "steps", 0),
    }


def _worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [run_module_case(payload["module"], sd) for sd in payload["seeds"]]


def eval_parallel(module_name: str, seeds: Sequence[int], jobs: int = 0) -> Dict[str, Any]:
    import multiprocessing as mp
    if jobs <= 0:
        jobs = min(os.cpu_count() or 4, max(1, len(seeds) // 4), 16)
    jobs = max(1, min(jobs, len(seeds)))
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [dict(module=module_name, seeds=ch) for ch in chunks if ch]
    if len(payloads) == 1:
        rows = _worker(payloads[0])
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_worker, payloads)
        rows = [r for p in parts for r in p]
    rows.sort(key=lambda r: r["seed"])
    return {"summary": summarize(rows), "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="策略统一评测")
    ap.add_argument("--modules", nargs="+", required=True, help="候选模块 (dotted path)")
    ap.add_argument("--seeds", default="7500-7699")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="逐局明细 (单进程, 只跑前几个种子)")
    ap.add_argument("--out", default="", help="逐局明细 JSON 路径")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)

    if args.trace:
        for m in args.modules:
            for sd in seeds[:5]:
                print("=== %s seed %d ===" % (m, sd), flush=True)
                run_module_case(m, sd, trace=True)
        return 0

    report: Dict[str, Any] = {"seeds": seeds, "results": {}}
    print("评测 %d 局, 种子 %d-%d" % (len(seeds), seeds[0], seeds[-1]))
    for m in args.modules:
        t0 = time.time()
        res = eval_parallel(m, seeds, args.jobs)
        res["summary"]["wall_s"] = time.time() - t0
        report["results"][m] = res
        print("%-42s %s" % (m.split(".")[-2] + "." + m.split(".")[-1] if "." in m else m,
                            fmt("", res["summary"]).strip()), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        print("明细 ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
