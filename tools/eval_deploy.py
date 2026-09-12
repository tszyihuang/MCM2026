"""部署口径评测: ``knows_total=False`` (正式测试看不到真值总数) 下的成绩.

``robotdog.solver.eval`` 用的是 ``knows_total=True`` (进程内能看到真值总数, 一到
``all_cleared`` 就停), 而**正式测试里机器狗看不到总数**, 收尾只能靠信念判据。
两个口径的差别是真实存在的开销 (收尾会多扫几轮), 因此优化时必须两边都看,
否则会优化出一个"进程内很快、部署时白烧几百秒"的策略。

用法::

    python tools/eval_deploy.py --module robotdog.solver.sweeper --seeds 9500-9599 --jobs 4
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.eval import parse_seeds, summarize  # noqa: E402


def run_case(module_name: str, seed: int) -> Dict[str, Any]:
    from robotdog.solver.sweeper import build_cfg as release_cfg, run_sweeper
    from robotdog.solver.world import World
    mod = importlib.import_module(module_name)
    # 与 tools/candidates.py 同一约定: 没有 build_cfg() 时回退到 sweeper.build_cfg(),
    # 绝不传 cfg=None —— 那会退化成默认 SweepConfig(), 静默换成另一套参数。
    cfg = mod.build_cfg() if hasattr(mod, "build_cfg") else release_cfg()
    fn = getattr(mod, "run_candidate", None) or run_sweeper
    w = World(seed=seed)
    # 用关键字传参, 避免位置参数错位。
    res = fn(w, cfg=cfg, knows_total=False)
    return {
        "seed": w.seed, "cleared": w.cleared_count, "total": w.n_sources,
        "ratio": w.cleared_count / w.n_sources, "virtual_s": w.virtual_t,
        "avg_clear_s": w.virtual_t / w.cleared_count if w.cleared_count else float("inf"),
        "moved_m": w.moved_m, "measures": w.measures, "clears": w.clears,
        "failed_clears": w.failed_clears, "steps": getattr(res, "steps", 0),
    }


def _worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [run_case(payload["module"], sd) for sd in payload["seeds"]]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="部署口径 (knows_total=False) 评测")
    ap.add_argument("--module", default="robotdog.solver.sweeper")
    ap.add_argument("--seeds", default="9500-9599")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)

    import multiprocessing as mp
    jobs = max(1, min(args.jobs, len(seeds)))
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [dict(module=args.module, seeds=ch) for ch in chunks if ch]
    t0 = time.time()
    if len(payloads) == 1:
        rows = _worker(payloads[0])
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_worker, payloads)
        rows = [r for p in parts for r in p]
    rows.sort(key=lambda r: r["seed"])
    s = summarize(rows)
    s["wall_s"] = time.time() - t0
    print("部署口径 (knows_total=False) | %s | %d 局" % (args.module, len(rows)))
    print("  清除率 %.4f  全清率 %.2f (%d/%d)  平均定位清除 %6.1f s (中位 %6.1f)  "
          "总虚拟 %6.0f s  移动 %6.0f m  检测 %5.1f  失败清除 %.2f"
          % (s["ratio"], s["full_rate"], s["full_count"], s["n"], s["avg_clear_mean"],
             s["avg_clear_median"], s["virtual_mean"], s["moved_mean"],
             s["measures_mean"],
             statistics.mean(r["failed_clears"] for r in rows)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"module": args.module, "seeds": seeds, "summary": s, "rows": rows},
                      fh, ensure_ascii=False, indent=1)
        print("明细 ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
