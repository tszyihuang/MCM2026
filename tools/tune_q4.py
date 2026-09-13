# -*- coding: utf-8 -*-
"""问题4 参数扫描 (多核并行), 直接打印"清除率 / s per source"两列指标。

比 ``tools/eval_q4.py --sweep`` 更省事的地方: 一次可以扫多个旋钮的组合,
并自动按 s/源 排序, 便于找"既达标又稳"的配方。

用法::

    python tools/tune_q4.py --seeds 9500-9569 --jobs 20 \
        --grid point_set=ring,hex --grid spacing=1000,1100
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    import importlib
    mod = importlib.import_module(payload["module"])
    from robotdog.solver.world import World
    over = payload["overrides"]
    cfg = mod.build_cfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    wk = mod.world_kwargs() if hasattr(mod, "world_kwargs") else {}
    rows = []
    for sd in payload["seeds"]:
        w = World(seed=sd, problem=payload["problem"], **wk)
        mod.run_sweeper4(w, cfg=cfg, knows_total=not payload["deploy"])
        rows.append({
            "cleared": w.cleared_count, "total": w.n_sources, "virtual_s": w.virtual_t,
            "moved_m": w.moved_m, "measures": w.measures, "clears": w.clears,
            "failed": w.failed_clears, "ndir": w.n_directional,
            "ndir_cleared": sum(1 for s in w.sources if not s.is_omni and not s.alive),
            "stages": getattr(w, "q4_stages", None),
        })
    return {"overrides": over, "rows": rows}


def _summ(rows: List[Dict[str, Any]], n_seeds: int) -> Dict[str, Any]:
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    return {
        "ratio": clr / max(tot, 1),
        "full": sum(1 for r in rows if r["cleared"] == r["total"]),
        "n": n_seeds,
        "persrc": statistics.mean(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "virtual": statistics.mean(r["virtual_s"] for r in rows),
        "moved": statistics.mean(r["moved_m"] for r in rows),
        "measures": statistics.mean(r["measures"] for r in rows),
        "clears": statistics.mean(r["clears"] for r in rows),
        "failed": statistics.mean(r["failed"] for r in rows),
        "ndir": sum(r["ndir"] for r in rows),
        "ndir_cleared": sum(r["ndir_cleared"] for r in rows),
        "scan_s": statistics.mean((r["stages"] or {}).get("scan_s", 0.0) for r in rows),
        "fix_s": statistics.mean((r["stages"] or {}).get("fix_s", 0.0) for r in rows),
        "tail_s": statistics.mean((r["stages"] or {}).get("tail_s", 0.0) for r in rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="robotdog.solver.sweeper4")
    ap.add_argument("--seeds", default="9500-9569")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--grid", action="append", default=[],
                    help="k=v1;v2;... (可多次; 值用 ; 分隔)")
    ap.add_argument("--fix", action="append", default=[], help="固定覆盖 k=v (可多次)")
    ap.add_argument("--problem", type=int, default=4)
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()

    seeds: List[int] = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    base: Dict[str, Any] = {}
    for kv in (a.fix or []):
        k, _, v = kv.partition("=")
        base[k] = _cast(v)

    axes: List[Tuple[str, List[Any]]] = []
    for item in a.grid:
        k, _, vals = item.partition("=")
        axes.append((k, [_cast(v) for v in vals.split(";")]))

    combos: List[Dict[str, Any]] = []
    if not axes:
        combos = [dict(base)]
    else:
        for prod in itertools.product(*[v for _k, v in axes]):
            ov = dict(base)
            ov.update({axes[i][0]: prod[i] for i in range(len(axes))})
            combos.append(ov)

    jobs = max(1, min(a.jobs, len(seeds)))
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    t0 = time.time()
    results: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    with ctx.Pool(jobs) as pool:
        payloads = [dict(module=a.module, seeds=ch, overrides=combos[0],
                         problem=a.problem, deploy=a.deploy) for ch in chunks if ch]
        for ov in combos:
            for p in payloads:
                p["overrides"] = ov
            parts = pool.map(_worker, payloads)
            rows = [r for p in parts for r in p["rows"]]
            results.append((ov, _summ(rows, len(seeds))))

    results.sort(key=lambda t: (-(t[1]["ratio"] >= 0.98), t[1]["persrc"]))
    print("%-52s %7s %10s %8s %8s %7s %7s %7s" %
          ("配方", "清除率", "全清", "s/源", "移动m", "检测", "扫描s", "收尾s"))
    for ov, s in results[:a.top]:
        tag = (",".join("%s=%s" % (kk, vv) for kk, vv in sorted(ov.items())
                        if kk in ("spacing", "edge_ring", "point_set", "ring_spec")))[:52]
        print("%-52s %7.4f %5d/%-4d %8.1f %8.0f %7.1f %7.0f %7.0f"
              % (tag, s["ratio"], s["full"], s["n"], s["persrc"],
                 s["moved"], s["measures"], s["scan_s"], s["tail_s"]))
    print("墙钟 %.0f s | %d 局/配方 | %d 配方" % (time.time() - t0, len(seeds), len(combos)))
    return 0


def _cast(v: str) -> Any:
    """``"a;b;c"`` → ``["a","b","c"]``; 单个值按字面量解析。"""
    if ";" in v:
        return [_cast(x) for x in v.split(";")]
    v = v.strip()
    if v in ("True", "False"):
        return v == "True"
    if v == "None":
        return None
    if v.startswith("(") or v.startswith("["):
        import ast
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


if __name__ == "__main__":
    raise SystemExit(main())
