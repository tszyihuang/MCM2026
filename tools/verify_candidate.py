"""发布版规划器验收: 确定性 + 双口径成绩.

一次调用完成两件事:

  1. **确定性**: 同一模块在同一批种子上跑两遍, 逐局逐字段比对, 必须完全一致;
  2. **成绩**: 进程内口径 (`knows_total=True`) 与部署口径 (`knows_total=False`) 同时给;

明细写到 ``data/reports/``。

用法::

    python tools\verify_candidate.py                                    # 验收发布版
    python tools\verify_candidate.py --seeds 9500-9599 --jobs 3
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.eval import parse_seeds, summarize  # noqa: E402

OUT_DIR = "data/reports"


def _worker(p: Tuple[str, List[int], str]) -> List[Dict[str, Any]]:
    """进程池任务必须是**模块级**函数 (闭包无法 pickle, Windows spawn 会直接报错)。"""
    module_name, seeds, mode = p
    if mode == "deploy":
        from tools.eval_deploy import run_case
    else:
        from tools.candidates import run_module_case as run_case  # type: ignore
    return [run_case(module_name, sd) for sd in seeds]


def _run(module_name: str, seeds: Sequence[int], jobs: int,
         mode: str) -> List[Dict[str, Any]]:
    import multiprocessing as mp

    jobs = max(1, min(jobs, len(seeds)))
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [(module_name, ch, mode) for ch in chunks if ch]
    if len(payloads) == 1:
        rows = _worker(payloads[0])
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_worker, payloads)
        rows = [r for p in parts for r in p]
    rows.sort(key=lambda r: r["seed"])
    return rows


def _fmt(name: str, rows: List[Dict[str, Any]]) -> str:
    s = summarize(rows)
    return ("%-14s 清除率 %.4f 全清 %3d/%3d  平均定位清除 %6.1f s (中位 %6.1f)  "
            "总虚拟 %6.0f  移动 %6.0f  检测 %5.1f"
            % (name, s["ratio"], s["full_count"], s["n"], s["avg_clear_mean"],
               s["avg_clear_median"], s["virtual_mean"], s["moved_mean"],
               s["measures_mean"]))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="发布版规划器验收 (确定性 + 双口径)")
    ap.add_argument("--module", default="robotdog.solver.sweeper",
                    help="被测模块 (dotted path); 默认 robotdog.solver.sweeper")
    ap.add_argument("--seeds", default="9500-9599")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    module = args.module

    print("模块: %s | 种子 %d 局 %d-%d" % (module, len(seeds), seeds[0], seeds[-1]))
    print()
    out: Dict[str, Any] = {"module": module, "seeds": seeds}

    for mode, label in (("inproc", "进程内"), ("deploy", "部署")):
        r1 = _run(module, seeds, args.jobs, mode)
        r2 = _run(module, seeds, args.jobs, mode)
        keys = ("cleared", "total", "virtual_s", "moved_m", "measures", "clears",
                "failed_clears", "steps")
        bad = [a["seed"] for a, b in zip(r1, r2)
               if any(abs(a[k] - b[k]) > 1e-9 for k in keys)]
        print("[%s口径]" % label)
        print("  " + _fmt(module.split(".")[-2] + "." + module.split(".")[-1], r1))
        print("  确定性: %s" % ("逐位一致 (跑两遍完全相同)"
                                if not bad else "**不一致** 的种子: %s" % bad[:10]))
        print()
        out[mode] = {"summary": summarize(r1), "deterministic": not bad,
                     "rows": r1}

    out_path = os.path.join(OUT_DIR, "verify_%s.json" % module.replace(".", "_"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("明细 ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
