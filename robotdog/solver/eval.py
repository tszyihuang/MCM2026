"""问题3 求解器的进程内评测入口 (批量对局 / 单局逐步明细)。

同一批随机案例 (同一 seed ⇒ 同一案例), 求解器确定性执行, 一局一局铺到多个核上,
300 局只需约 90 s。

用法::

    # 留出集上 (300 局)
    python -m robotdog.solver.eval --seeds 9500-9799 --jobs 8

    # 打印前 N 局的逐步明细 (单进程)
    python -m robotdog.solver.eval --seeds 9500 --trace 1

口径 (与仓库其余部分一致):
  * 清除比例  = 已清除数 / 真值总数
  * 全清率    = 全清局数 / 局数
  * 平均定位清除时间 = 单局总虚拟时间 / 该局已清除数, 再对局取平均

这里是**进程内口径** (能读到真值总数, 清完立刻停)。正式测试口径看不到总数, 只能靠
信念判断何时收手, 用 ``tools/eval_deploy.py`` 评测 —— 两个口径不能混比。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# --------------------------------------------------------------------------------------
# 单局执行 (worker 内)
# --------------------------------------------------------------------------------------
def _row(w: Any, steps: int) -> Dict[str, Any]:
    return {
        "seed": w.seed, "cleared": w.cleared_count, "total": w.n_sources,
        "ratio": w.cleared_count / w.n_sources,
        "virtual_s": w.virtual_t,
        "avg_clear_s": w.virtual_t / w.cleared_count if w.cleared_count else float("inf"),
        "moved_m": w.moved_m, "measures": w.measures, "clears": w.clears,
        "failed_clears": w.failed_clears, "steps": steps,
    }


def run_case(seed: int, trace: bool = False) -> Dict[str, Any]:
    """跑一局问题3, 返回一行统计。"""
    from robotdog.solver.sweeper import build_cfg, run_sweeper
    from robotdog.solver.world import World
    w = World(seed=seed)
    res = run_sweeper(w, build_cfg(), trace=trace)
    return _row(w, res.steps)


def _worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [run_case(sd) for sd in payload["seeds"]]


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    full = sum(1 for r in rows if r["cleared"] == r["total"])
    return {
        "n": len(rows), "cleared": clr, "total": tot, "ratio": clr / max(tot, 1),
        "full_rate": full / max(len(rows), 1), "full_count": full,
        "avg_clear_mean": statistics.mean([r["avg_clear_s"] for r in rows]),
        "avg_clear_median": statistics.median([r["avg_clear_s"] for r in rows]),
        "virtual_mean": statistics.mean([r["virtual_s"] for r in rows]),
        "moved_mean": statistics.mean([r["moved_m"] for r in rows]),
        "measures_mean": statistics.mean([r["measures"] for r in rows]),
        "clears_mean": statistics.mean([r["clears"] for r in rows]),
    }


def fmt(name: str, s: Dict[str, Any]) -> str:
    return ("%-8s 清除率 %.4f  全清率 %.2f (%d/%d)  平均定位清除 %6.1f s (中位 %6.1f)  "
            "总虚拟 %6.0f s  移动 %6.0f m  检测 %5.1f"
            % (name, s["ratio"], s["full_rate"], s["full_count"], s["n"],
               s["avg_clear_mean"], s["avg_clear_median"], s["virtual_mean"],
               s["moved_mean"], s["measures_mean"]))


def eval_parallel(seeds: Sequence[int], jobs: int = 0) -> Dict[str, Any]:
    import multiprocessing as mp
    if jobs <= 0:
        jobs = min(os.cpu_count() or 4, max(1, len(seeds) // 4), 32)
    jobs = max(1, min(jobs, len(seeds)))
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [dict(seeds=ch) for ch in chunks if ch]
    if len(payloads) == 1:
        rows = _worker(payloads[0])
    else:
        # fork 在 Linux/macOS 上最省 (子进程直接继承已导入的 numpy); Windows 只有
        # spawn, 因此按可用性回退 —— 两条路径的 _worker 都是模块级函数, 可 pickle。
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_worker, payloads)
        rows = [r for p in parts for r in p]
    rows.sort(key=lambda r: r["seed"])
    return {"summary": summarize(rows), "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="问题3 求解器统一评测")
    ap.add_argument("--seeds", default="9500-9599", help="种子区间/列表, 如 9500-9799,9001")
    ap.add_argument("--jobs", type=int, default=0, help="并行进程数 (0=自动)")
    ap.add_argument("--trace", type=int, default=0, help=">0 时打印前 N 局的逐步明细 (单进程)")
    ap.add_argument("--out", default="", help="把逐局明细写成 JSON")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        ap.error("--seeds 解析为空")

    if args.trace:
        for sd in seeds[:args.trace]:
            print("=== seed %d ===" % sd, flush=True)
            run_case(sd, trace=True)
        return 0

    t0 = time.time()
    res = eval_parallel(seeds, args.jobs)
    res["summary"]["wall_s"] = time.time() - t0
    report: Dict[str, Any] = {"seeds": seeds, "results": {"sweep": res}}
    print("评测 %d 局, 种子 %d-%d" % (len(seeds), seeds[0], seeds[-1]))
    print(fmt("sweep", res["summary"]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        print("明细 ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
