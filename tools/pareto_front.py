# -*- coding: utf-8 -*-
"""帕累托前沿分析: 双目标 (最大化清除比例, 最小化平均定位清除时间)。

做法: 构造一族**候选运行点** —— 收尾闸门 ``hunt_mass_min`` 网格
(0.0 表示"扫到没有残余质量为止", 1.0 等价于不作收尾)。
全部在**部署口径** (``knows_total=False``, 即正式测试口径) 下跑标定集 9500-11499 (2000 局),
再对**互不重叠**的独立测试集 8000-8999 (1000 局) 复核。区间可用 ``--calib`` / ``--test`` 改。

然后计算非支配集 (Pareto 前沿) 与膝点 (knee point)。

用法::

    python tools/pareto_front.py
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.eval import parse_seeds  # noqa: E402

# 收尾闸门网格: 0.0=扫到无残余; 1.0 等价于不做收尾 (时间最短、清除率最低的一端)
GATE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.0]

# 候选族: (标签, 模块, base, 固定覆盖项)
CANDIDATES = [
    ("问题3 策略", "robotdog.solver.sweeper", "release", {}),
]


def _build_cfg(mod, base: str, overrides: Dict[str, Any]):
    """按发布约定构造配置对象。"""
    if base != "release":
        raise ValueError(base)
    cfg = mod.build_cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _task(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """一个 (候选, 种子分片) 工作单元。"""
    mod = importlib.import_module(payload["module"])
    from robotdog.solver.world import World

    overrides = dict(payload["overrides"])
    overrides.setdefault("hunt_mass_min", payload["gate"])
    cfg = _build_cfg(mod, payload["base"], overrides)
    fn = getattr(mod, "run_candidate", None) or getattr(mod, "run_sweeper")

    rows = []
    for sd in payload["seeds"]:
        w = World(seed=sd)
        fn(w, cfg=cfg, knows_total=False)              # 部署口径 = 正式测试口径
        rows.append({"seed": sd, "cleared": w.cleared_count, "total": w.n_sources,
                     "virtual_s": w.virtual_t, "moved_m": w.moved_m,
                     "measures": w.measures})
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    return {
        "n": len(rows), "ratio": clr / max(tot, 1), "cleared": clr, "total": tot,
        "full": sum(1 for r in rows if r["cleared"] == r["total"]),
        "avg_clear": statistics.mean(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "moved": statistics.mean(r["moved_m"] for r in rows),
        "measures": statistics.mean(r["measures"] for r in rows),
    }


def pareto_front(cands: List[Dict[str, Any]]) -> List[int]:
    """非支配集: 目标为 (ratio 越大越好, avg_clear 越小越好)。返回候选下标, 按 avg_clear 升序。"""
    front = []
    for i, a in enumerate(cands):
        dominated = False
        for j, b in enumerate(cands):
            if i == j:
                continue
            ge = b["ratio"] >= a["ratio"] - 1e-12 and b["avg_clear"] <= a["avg_clear"] + 1e-9
            gt = b["ratio"] > a["ratio"] + 1e-12 or b["avg_clear"] < a["avg_clear"] - 1e-9
            if ge and gt:
                dominated = True
                break
        if not dominated:
            front.append(i)
    front.sort(key=lambda i: cands[i]["avg_clear"])
    return front


def knee_point(cands: List[Dict[str, Any]], front: List[int]) -> int:
    """归一化后到"两端连线"距离最大的前沿点 (经典膝点)。"""
    if len(front) <= 2:
        return front[len(front) // 2]
    ts = [cands[i]["avg_clear"] for i in front]
    rs = [cands[i]["ratio"] for i in front]
    t0, t1 = min(ts), max(ts)
    r0, r1 = min(rs), max(rs)
    span_t, span_r = max(t1 - t0, 1e-9), max(r1 - r0, 1e-9)
    pts = [((cands[i]["avg_clear"] - t0) / span_t, (cands[i]["ratio"] - r0) / span_r)
           for i in front]
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    best, best_d = 0, -1.0
    for k, (x, y) in enumerate(pts):
        d = abs(dy * (x - x1) - dx * (y - y1)) / norm
        if d > best_d:
            best, best_d = k, d
    return front[best]


def run(seeds: List[int], jobs: int) -> Dict[str, Any]:
    import multiprocessing as mp

    tasks = []
    meta = []
    for label, module, base, fixed in CANDIDATES:
        for gate in GATE_GRID:
            meta.append((label, module, base, dict(fixed), gate))

    chunk = 30
    for label, module, base, fixed, gate in meta:
        for i in range(0, len(seeds), chunk):
            tasks.append(dict(module=module, base=base,
                              overrides=dict(fixed), gate=gate,
                              seeds=seeds[i:i + chunk], tag=(label, gate)))
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with mp.get_context(method).Pool(jobs) as pool:
        parts = pool.map(_task, tasks)

    bucket: Dict[Any, List[Dict[str, Any]]] = {}
    for t, rows in zip(tasks, parts):
        bucket.setdefault(t["tag"], []).extend(rows)

    cands = []
    for (label, gate), rows in bucket.items():
        rows.sort(key=lambda r: r["seed"])
        s = summarize(rows)
        s.update(label=label, gate=gate)
        cands.append(s)
    cands.sort(key=lambda c: (c["label"], c["gate"]))
    front = pareto_front(cands)
    knee = knee_point(cands, front)
    return {"problem": 3, "seeds": [seeds[0], seeds[-1]], "n": len(seeds),
            "deploy": True, "candidates": cands, "frontier": front, "knee": knee}


def main() -> int:
    ap = argparse.ArgumentParser(description="帕累托前沿 (清除比例 vs 平均定位清除时间)")
    ap.add_argument("--problem", type=int, default=3, choices=(3,),
                    help="本工程只提供问题3 求解器")
    ap.add_argument("--jobs", type=int, default=0, help="0=自动")
    ap.add_argument("--calib", default="9500-11499", help="标定集种子区间")
    ap.add_argument("--test", default="8000-8999", help="独立测试集种子区间")
    args = ap.parse_args()

    cpu = os.cpu_count() or 4
    jobs = args.jobs or max(2, min(cpu, 12))
    holdout = parse_seeds(args.calib)
    gen = parse_seeds(args.test)
    labels = {"": "标定集 %s" % args.calib, "_gen": "独立测试集 %s" % args.test}

    for tag, seeds in (("", holdout), ("_gen", gen)):
        t0 = time.time()
        out = run(seeds, jobs)
        out["generality"] = tag == "_gen"
        path = "data/reports/pareto_p3%s.json" % tag
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("=== %s | %d 局 | 墙钟 %.0f s -> %s"
              % (labels[tag], len(seeds), time.time() - t0, path))
        print("  前沿 (%d 个非支配点, 按平均定位清除升序):" % len(out["frontier"]))
        for i in out["frontier"]:
            c = out["candidates"][i]
            mark = "  <- 膝点" if i == out["knee"] else ""
            print("    %-24s gate=%.2f  清除率 %.4f  平均定位清除 %6.1f s  移动 %5.0f m%s"
                  % (c["label"], c["gate"], c["ratio"], c["avg_clear"], c["moved"], mark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
