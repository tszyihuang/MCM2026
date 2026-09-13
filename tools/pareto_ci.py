# -*- coding: utf-8 -*-
"""收尾闸门的**配对**置信区间: 给 Pareto 前沿的每个候选点配一条不确定性。

``tools/pareto_front.py`` 只报点估计, 无法回答"λ=0.20 比 λ=0.60 多花的
16.4 s/源 是真的, 还是案例噪声"。本脚本用**逐局配对**的方式补上这一层:

* 同一批 seed 下, 每个门限跑的是**同一个案例**, 于是
  ``Δratio = ratio(λ) - ratio(λ_ref)`` 与 ``Δt = avg_clear(λ) - avg_clear(λ_ref)``
  都定义在案例层面;
* 对**案例**做有放回重采样 (cluster bootstrap, 默认 4000 次), 同时重采样两个门限,
  得到考虑了案例间波动的 95% 区间。

该区间比"按源个数算二项区间"更保守, 也更贴合"案例是独立重复、源在案例内聚类"
的评测口径。

全部在**部署口径** (``knows_total=False``, 即正式测试口径) 下跑标定集
9500-11499 (2000 局), 再在互不重叠的独立测试集 8000-8999 (1000 局) 上复核。

用法::

    python tools/pareto_ci.py
    python tools/pareto_ci.py --ref 0.20 --boot 4000 --jobs 12
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.eval import parse_seeds  # noqa: E402

# 与 pareto_front.py 保持同一网格, 便于两张表对齐
GATE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0]
MODULE = "robotdog.solver.sweeper"


def _task(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """一个 (门限, 种子分片) 工作单元, 返回**逐局**记录。"""
    mod = importlib.import_module(payload["module"])
    from robotdog.solver.world import World

    cfg = mod.build_cfg()
    setattr(cfg, "hunt_mass_min", payload["gate"])
    fn = getattr(mod, "run_candidate", None) or getattr(mod, "run_sweeper")

    rows = []
    for sd in payload["seeds"]:
        w = World(seed=sd)
        fn(w, cfg=cfg, knows_total=False)          # 部署口径 = 正式测试口径
        rows.append({"seed": sd, "cleared": w.cleared_count, "total": w.n_sources,
                     "virtual_s": w.virtual_t, "moved_m": w.moved_m,
                     "measures": w.measures})
    return rows


def collect(seeds: List[int], gates: List[float], jobs: int) -> Dict[float, List[Dict[str, Any]]]:
    import multiprocessing as mp

    chunk = 30
    tasks = []
    for gate in gates:
        for i in range(0, len(seeds), chunk):
            tasks.append(dict(module=MODULE, gate=gate, seeds=seeds[i:i + chunk]))
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with mp.get_context(method).Pool(jobs) as pool:
        parts = pool.map(_task, tasks)

    bucket: Dict[float, List[Dict[str, Any]]] = {g: [] for g in gates}
    for t, rows in zip(tasks, parts):
        bucket[t["gate"]].extend(rows)
    for gate in bucket:
        bucket[gate].sort(key=lambda r: r["seed"])
    return bucket


def _arrays(rows: List[Dict[str, Any]]):
    clr = np.array([r["cleared"] for r in rows], dtype=np.float64)
    tot = np.array([r["total"] for r in rows], dtype=np.float64)
    avg = np.array([r["virtual_s"] / max(r["cleared"], 1) for r in rows], dtype=np.float64)
    return clr, tot, avg


def _pct(x: np.ndarray, lo: float = 2.5, hi: float = 97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def summarize(bucket: Dict[float, List[Dict[str, Any]]], ref_gate: float,
              boot: int, rng: np.random.Generator) -> Dict[str, Any]:
    gates = sorted(bucket)
    if ref_gate not in bucket:
        raise ValueError("参考门限 %s 不在网格内: %s" % (ref_gate, gates))

    n = len(bucket[ref_gate])
    # 同一组重采样下标用于所有门限 -> 真正的配对 bootstrap
    idx = rng.integers(0, n, size=(boot, n)).astype(np.int32)

    data = {}
    for g in gates:
        clr, tot, avg = _arrays(bucket[g])
        data[g] = dict(
            clr=clr, tot=tot, avg=avg,
            ratio=float(clr.sum() / tot.sum()),
            mean_avg=float(avg.mean()),
            cleared=int(clr.sum()), total=int(tot.sum()),
            full=int(sum(1 for r in bucket[g] if r["cleared"] == r["total"])),
            moved=statistics.mean(r["moved_m"] for r in bucket[g]),
            measures=statistics.mean(r["measures"] for r in bucket[g]),
        )

    def boot_stats(g):
        d = data[g]
        r = d["clr"][idx].sum(axis=1) / d["tot"][idx].sum(axis=1)
        a = d["avg"][idx].mean(axis=1)
        return r, a

    rc, ac = boot_stats(ref_gate)
    out = []
    for g in gates:
        d = data[g]
        r_b, a_b = boot_stats(g)
        r_lo, r_hi = _pct(r_b)
        a_lo, a_hi = _pct(a_b)
        dr_lo, dr_hi = _pct(r_b - rc)
        da_lo, da_hi = _pct(a_b - ac)
        worse = int(np.sum(d["clr"] < data[ref_gate]["clr"]))
        better = int(np.sum(d["clr"] > data[ref_gate]["clr"]))
        out.append(dict(
            gate=g, ratio=d["ratio"], ratio_lo=r_lo, ratio_hi=r_hi,
            mean_avg=d["mean_avg"], mean_avg_lo=a_lo, mean_avg_hi=a_hi,
            cleared=d["cleared"], total=d["total"], full=d["full"],
            moved=d["moved"], measures=d["measures"],
            d_ratio=d["ratio"] - data[ref_gate]["ratio"], d_ratio_lo=dr_lo, d_ratio_hi=dr_hi,
            d_avg=d["mean_avg"] - data[ref_gate]["mean_avg"], d_avg_lo=da_lo, d_avg_hi=da_hi,
            # Δratio 的 bootstrap 分布里 <=0 的比例: 越小说明"低于参考点"越不可能是噪声
            p_ratio_not_below_ref=float(np.mean(r_b - rc > 0.0)),
            n_seeds_fewer=worse, n_seeds_more=better,
            n_seeds_tie=int(n - worse - better),
        ))
    return {"n": n, "ref_gate": ref_gate, "boot": boot, "candidates": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="收尾闸门的配对 confidence interval")
    ap.add_argument("--jobs", type=int, default=0, help="0=自动")
    ap.add_argument("--boot", type=int, default=4000, help="bootstrap 重采样次数")
    ap.add_argument("--ref", type=float, default=0.20, help="参考门限 (采纳点)")
    ap.add_argument("--calib", default="9500-11499", help="标定集种子区间")
    ap.add_argument("--test", default="8000-8999", help="独立测试集种子区间")
    args = ap.parse_args()

    cpu = os.cpu_count() or 4
    jobs = args.jobs or max(2, min(cpu, 12))

    result: Dict[str, Any] = {"problem": 3, "deploy": True, "gates": GATE_GRID,
                              "ref_gate": args.ref, "boot": args.boot}
    for tag, seeds in (("", parse_seeds(args.calib)), ("_gen", parse_seeds(args.test))):
        t0 = time.time()
        bucket = collect(seeds, GATE_GRID, jobs)
        rng = np.random.default_rng(20260912)      # 固定种子 -> 结果可复现
        block = summarize(bucket, args.ref, args.boot, rng)
        block["seeds"] = [seeds[0], seeds[-1]]
        block["generality"] = tag == "_gen"
        result["calib" if tag == "" else "test"] = block
        print("=== %s | %d 局 | 墙钟 %.0f s" % (
            "独立测试集" if tag else "标定集", len(seeds), time.time() - t0))
        for c in block["candidates"]:
            star = " <- 参考点" if abs(c["gate"] - args.ref) < 1e-12 else ""
            print("  gate=%.2f  清除率 %.4f [%.4f, %.4f]  平均 %6.1f s [%6.1f, %6.1f]"
                  "  Δ比例 %+.4f [%+.4f, %+.4f]  Δt %+6.1f [%+6.1f, %+6.1f]%s"
                  % (c["gate"], c["ratio"], c["ratio_lo"], c["ratio_hi"],
                     c["mean_avg"], c["mean_avg_lo"], c["mean_avg_hi"],
                     c["d_ratio"], c["d_ratio_lo"], c["d_ratio_hi"],
                     c["d_avg"], c["d_avg_lo"], c["d_avg_hi"], star))

    path = "data/reports/pareto_p3_ci.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print("-> %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
