"""小批量多核基准: 一口气把 N 个种子跑完, 并报出每个种子的结果与整体统计.

与 ``tools/profile_run.py`` 的区别: 那个脚本回答"时间去哪儿了";
本脚本面向"尽快知道这批种子表现如何", 用全部核心并行。

两种用法, 按批量大小选:

* **一次性** —— 直接给种子跑完就退出。每次都要付一次进程池启动开销 (本机
  16 进程约 0.33 s, 与批量无关), 所以批量小时它会占大头。
* **常驻 (推荐用于反复回归)** —— ``--serve`` 只启动一次进程池, 之后从标准输入
  逐批读种子, 省掉每批 0.3 s 的启动开销; 批量 16~64 局时整批通常 0.1~0.3 s。

    python tools/bench_cases.py                      # 问题3, 种子 1001-1016, 全部核心
    python tools/bench_cases.py --problem 4 --seeds 2001-2016
    python tools/bench_cases.py --seeds 3001-3024 --workers 4   # 小批量下 4 进程常常更快
    python tools/bench_cases.py --serve               # 常驻: 之后每次输入一行种子回车
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _run(job: Tuple[int, int, Dict[str, Any]]) -> Dict[str, Any]:
    problem, seed, params = job
    from tools.tune import run_case

    c0 = time.process_time()
    row = run_case(problem, seed, params)
    row["cpu_s"] = time.process_time() - c0
    row.pop("params", None)
    return row


def parse_seeds(text: str) -> List[int]:
    """解析 ``1001-1016`` / ``1001,1005`` / ``1001-1008,2001``。"""
    out: List[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def suggest_workers(
    cases: int,
    workers: int,
    startup_per_worker_s: float = 0.031,
    ms_per_case: float = 45.0,
) -> int:
    """按"总时间 = w 个进程的启动开销 + 每核分到的局数 x 单局耗时"挑并行度。

    小批量时进程数开满反而更慢: 16 个进程各自只跑 1~2 局, 却要先分摊约
    0.5 s 的进程池启动。本机实测 (16 局, 问题3): 4 进程 0.48 s / 8 进程
    0.47 s / 16 进程 0.58 s, 所以自动值不能简单取 "cpu_count"。

    ``startup_per_worker_s`` 由实测进程池启动开销拟合 (16 进程约 0.33~0.5 s,
    即每个进程约 0.03 s)。
    """
    workers = max(1, min(workers, cases))
    best, best_cost = workers, float("inf")
    for w in range(1, workers + 1):
        cost = startup_per_worker_s * w + math.ceil(cases / w) * ms_per_case / 1000.0
        if cost < best_cost:
            best, best_cost = w, cost
    return best


def run_batch(
    pool: Optional[ProcessPoolExecutor],
    problem: int,
    seeds: Sequence[int],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """跑一批种子。``pool`` 为 None 时串行执行。"""
    params = params or {}
    jobs = [(problem, s, params) for s in seeds]
    t0 = time.perf_counter()
    if pool is None:
        rows = [_run(j) for j in jobs]
    else:
        rows = list(pool.map(_run, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    cleared = sum(r["cleared"] for r in rows)
    total = sum(r["total"] for r in rows)
    cpu = sum(r["cpu_s"] for r in rows)
    return {
        "problem": problem,
        "cases": len(rows),
        "wall_s": round(wall, 4),
        "cpu_s": round(cpu, 4),
        "cpu_ms_per_case": round(cpu / len(rows) * 1000.0, 2) if rows else 0.0,
        "parallel_speedup": round(cpu / wall, 2) if wall > 0 else 0.0,
        "cleared": cleared,
        "total": total,
        "cleared_ratio": round(cleared / total, 6) if total else 0.0,
        "mean_virtual_s": round(statistics.fmean(r["virtual_s"] for r in rows), 1) if rows else 0.0,
        "mean_measures": round(statistics.fmean(r["measures"] for r in rows), 1) if rows else 0.0,
        "mean_moved_m": round(statistics.fmean(r["moved_m"] for r in rows), 1) if rows else 0.0,
        "rows": rows,
    }


def report(summary: Dict[str, Any], workers: int, verbose: bool = True) -> None:
    rows = summary["rows"]
    if verbose:
        print("%-8s %-10s %-12s %-12s %s" % ("seed", "清除", "虚拟秒", "检测次数", "移动(m)"))
        print("-" * 58)
        for r in rows:
            print("%-8d %-10s %-12.1f %-12d %.0f"
                  % (r["seed"], "%d/%d" % (r["cleared"], r["total"]), r["virtual_s"],
                     r["measures"], r["moved_m"]))
        print("-" * 58)
    print("问题%d  %d 局 / %d 进程   墙钟 %.3f s   CPU %.3f s (%.1f ms/局, 并行 %.1fx)"
          % (summary["problem"], summary["cases"], workers, summary["wall_s"],
             summary["cpu_s"], summary["cpu_ms_per_case"], summary["parallel_speedup"]))
    print("平均清除比例 %.1f%% (%d/%d)   平均虚拟耗时 %.1f s   平均检测 %.1f 次"
          % (100.0 * summary["cleared_ratio"], summary["cleared"], summary["total"],
             summary["mean_virtual_s"], summary["mean_measures"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="小批量多核对局基准")
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--seeds", default=None, help="如 1001-1016 或 1001,1005")
    ap.add_argument("--n", type=int, default=16, help="未给 --seeds 时, 从 --start 起跑 n 局")
    ap.add_argument("--start", type=int, default=1001)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = 自动 (按批量大小与进程池启动开销取最优)")
    ap.add_argument("--serve", action="store_true",
                    help="常驻模式: 只启动一次进程池, 之后从标准输入逐行读种子")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cap = os.cpu_count() or 1
    if args.seeds:
        seeds = parse_seeds(args.seeds)
    else:
        seeds = list(range(args.start, args.start + args.n))

    if args.workers > 0:
        workers = max(1, min(args.workers, len(seeds), cap))
    elif args.serve:
        # 常驻模式启动开销只付一次, 直接开满核
        workers = max(1, min(cap, len(seeds)))
    else:
        workers = suggest_workers(len(seeds), cap)

    if args.serve:
        return serve(args.problem, workers, args.json)

    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        summary = run_batch(pool, args.problem, seeds)
    finally:
        if pool is not None:
            pool.shutdown()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        report(summary, workers)
    return 0


def serve(problem: int, workers: int, as_json: bool) -> int:
    """常驻模式: 进程池只建一次, 之后反复读种子批次。

    输入一行种子 (``1001-1016``); 空行默认跑最近一批的下一个区间;
    ``q`` / ``quit`` / EOF 退出。省掉每批 0.3 s 的进程池启动开销,
    因此"改一版策略 -> 立刻回归 16 局"可以在 0.2 s 左右完成。
    """
    print("常驻基准就绪: 问题%d, %d 个进程, 池启动一次后复用。" % (problem, workers))
    print("输入种子区间 (如 1001-1016), 回车重复下一批, q 退出。")
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    start = 1001
    step = 16
    try:
        while True:
            try:
                line = input("seeds> ").strip()
            except EOFError:
                break
            if line.lower() in ("q", "quit", "exit"):
                break
            if not line:
                seeds = list(range(start, start + step))
            else:
                try:
                    seeds = parse_seeds(line)
                except ValueError:
                    print("无法解析: %r" % line)
                    continue
                if seeds:
                    step = max(1, len(seeds))
            summary = run_batch(pool, problem, seeds)
            if as_json:
                print(json.dumps(summary, ensure_ascii=False))
            else:
                report(summary, workers)
            start = (seeds[-1] + 1) if seeds else start
    except KeyboardInterrupt:
        pass
    finally:
        if pool is not None:
            pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
