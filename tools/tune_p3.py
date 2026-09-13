# -*- coding: utf-8 -*-
"""问题3 参数扫描（**部署口径** ``knows_total=False``，被打分的那个口径）。

用途: 在"清除比例不掉"的前提下找更快的配置 —— 即速度/准确度的帕累托前沿。
每个配置只覆盖 ``SweepConfig`` 的若干字段, 其余保持发布值 (:func:`sweeper.build_cfg`)。

输出一行一个配置: 清除比例 / 全清局 / **平均定位清除** / 每局移动 / 每局检测,
并给出相对发布配置的增量。``--stage combo`` 跑若干组合候选。

用法::

    python tools/tune_p3.py --seeds 9500-9999 --jobs 8                 # 单旋钮 OAT
    python tools/tune_p3.py --seeds 9500-9999 --jobs 8 --stage combo   # 组合候选
    python tools/tune_p3.py --seeds 20000-20999 --jobs 8 --only NAME   # 独立集复核单个候选
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.solver.eval import parse_seeds                        # noqa: E402

#: 单旋钮 OAT: (名字, 覆盖字段)
OAT: List[Tuple[str, Dict[str, Any]]] = [
    ("release", {}),
    ("coverage_finish=0", {"coverage_finish": False}),
    ("coverage_max_steps=8", {"coverage_max_steps": 8}),
    ("coverage_max_steps=30", {"coverage_max_steps": 30}),
    ("probe_v_source=1000", {"probe_v_source": 1000.0}),
    ("probe_v_source=2000", {"probe_v_source": 2000.0}),
    ("hunt_max_ch=12", {"hunt_max_ch": 12}),
    ("hunt_max_ch=8", {"hunt_max_ch": 8}),
    ("sweep_max_ch=12", {"sweep_max_ch": 12}),
    ("sweep_max_ch=8", {"sweep_max_ch": 8}),
    ("sweep_pd_min=0.15", {"sweep_pd_min": 0.15}),
    ("sweep_pd_min=0.20", {"sweep_pd_min": 0.20}),
    ("sweep_pi_min=0.05", {"sweep_pi_min": 0.05}),
    ("sweep_skip_std=60", {"sweep_skip_std": 60.0}),
    ("sweep_skip_std=200", {"sweep_skip_std": 200.0}),
    ("probe_max=24", {"probe_max": 24}),
    ("probe_max=56", {"probe_max": 56}),
    ("hunt_fail_max=8", {"hunt_fail_max": 8}),
    ("hunt_fail_max=16", {"hunt_fail_max": 16}),
    ("pi_pursue=0.15", {"pi_pursue": 0.15}),
    ("pi_pursue=0.40", {"pi_pursue": 0.40}),
    ("clear_std=20", {"clear_std": 20.0}),
    ("clear_std=35", {"clear_std": 35.0}),
    ("pursue_max_steps=10", {"pursue_max_steps": 10}),
    ("pursue_max_steps=20", {"pursue_max_steps": 20}),
    ("leg_step_m=400", {"leg_step_m": 400.0}),
    ("leg_step_m=650", {"leg_step_m": 650.0}),
    ("standoff_m=350", {"standoff_m": 350.0}),
    ("standoff_m=500", {"standoff_m": 500.0}),
    ("lateral_m=20", {"lateral_m": 20.0}),
    ("lateral_m=45", {"lateral_m": 45.0}),
    ("fix_step_m=20", {"fix_step_m": 20.0}),
    ("fix_step_m=45", {"fix_step_m": 45.0}),
    ("info_value_s=200", {"info_value_s": 200.0}),
    ("info_value_s=320", {"info_value_s": 320.0}),
    ("leg_stop=0", {"leg_stop": False}),
    ("joint_choice=0", {"joint_choice": False}),
    ("tsp_order=0", {"tsp_order": False}),
]

#: 组合候选（在 OAT 结果上按"单项不掉清除率且不更慢"筛出来后叠加）
_SAFE = {"sweep_pi_min": 0.05, "leg_stop": False, "probe_v_source": 1000.0,
         "standoff_m": 350.0, "clear_std": 35.0, "info_value_s": 320.0,
         "lateral_m": 20.0}
COMBO: List[Tuple[str, Dict[str, Any]]] = [
    ("release", {}),
    ("core3 (pi_min+leg_stop+v_source)",
     {"sweep_pi_min": 0.05, "leg_stop": False, "probe_v_source": 1000.0}),
    ("safe7 (7 项安全叠加)", dict(_SAFE)),
    ("safe7 + coverage_finish=0",
     dict(_SAFE, coverage_finish=False)),
    ("core3 + coverage_finish=0",
     {"sweep_pi_min": 0.05, "leg_stop": False, "probe_v_source": 1000.0,
      "coverage_finish": False}),
    ("leg_stop=0 + coverage_finish=0",
     {"leg_stop": False, "coverage_finish": False}),
    ("safe7 + hunt_max_ch=12",
     dict(_SAFE, hunt_max_ch=12)),
    ("safe7 + sweep_skip_std=60",
     dict(_SAFE, sweep_skip_std=60.0)),
]


#: 速度优先档（关闭覆盖收尾, 只用信念门限 λ 收尾）: 论文 §5.2 的折中曲线
_SPEED = {"coverage_finish": False}
SPEED: List[Tuple[str, Dict[str, Any]]] = [
    ("speed lam=0.00", dict(_SPEED, hunt_mass_min=0.00)),
    ("speed lam=0.05", dict(_SPEED, hunt_mass_min=0.05)),
    ("speed lam=0.10", dict(_SPEED, hunt_mass_min=0.10)),
    ("speed lam=0.20", dict(_SPEED, hunt_mass_min=0.20)),
    ("speed lam=0.30", dict(_SPEED, hunt_mass_min=0.30)),
    ("speed lam=0.40", dict(_SPEED, hunt_mass_min=0.40)),
    ("speed lam=0.60", dict(_SPEED, hunt_mass_min=0.60)),
    ("speed lam=1.00", dict(_SPEED, hunt_mass_min=1.00)),
]


def run_case(payload) -> Dict[str, Any]:
    over, seed = payload
    from robotdog.solver.sweeper import build_cfg, run_sweeper
    from robotdog.solver.world import World
    cfg = build_cfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    w = World(seed=seed)
    run_sweeper(w, cfg=cfg, knows_total=False)
    return {
        "seed": seed, "cleared": w.cleared_count, "total": w.n_sources,
        "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
    }


def _run_chunk(payloads) -> List[Dict[str, Any]]:
    """工作进程入口（必须是模块级函数, 否则 Windows 上无法 pickle）。"""
    return [run_case(p) for p in payloads]


#: 最近一次 ``eval_cfg`` 的逐局明细 (供 ``--rows-out`` 使用; 单进程顺序调用, 无并发写)
_LAST_ROWS: List[Dict[str, Any]] = []


def eval_cfg(over: Dict[str, Any], seeds: List[int], jobs: int) -> Dict[str, Any]:
    global _LAST_ROWS
    chunks = [list(seeds[i::jobs]) for i in range(jobs)]
    payloads = [[(over, sd) for sd in ch] for ch in chunks if ch]
    t0 = time.time()
    if len(payloads) == 1:
        rows = _run_chunk(payloads[0])
    else:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(len(payloads)) as pool:
            parts = pool.map(_run_chunk, payloads)
        rows = [r for p in parts for r in p]
    _LAST_ROWS = rows
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    return {
        "n": len(rows), "ratio": clr / max(tot, 1), "cleared": clr, "total": tot,
        "full_count": sum(1 for r in rows if r["cleared"] == r["total"]),
        "full_rate": sum(1 for r in rows if r["cleared"] == r["total"]) / len(rows),
        "avg_clear": statistics.fmean(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "median_clear": statistics.median(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "worst_clear": max(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
        "virtual": statistics.fmean(r["virtual_s"] for r in rows),
        "moved": statistics.fmean(r["moved_m"] for r in rows),
        "measures": statistics.fmean(r["measures"] for r in rows),
        "missed": [r["seed"] for r in rows if r["cleared"] != r["total"]],
        "wall_s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="问题3 参数扫描（部署口径）")
    ap.add_argument("--seeds", default="9500-9999")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stage", choices=("oat", "combo", "speed"), default="oat")
    ap.add_argument("--rows-out", default="",
                    help="把每局的 (cleared,total) 明细也写进 JSON, 用于评估小样本风险")
    ap.add_argument("--only", default="", help="只跑名字里含该子串的配置")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    jobs = max(1, min(a.jobs, len(seeds)))
    cfgs = {"oat": OAT, "combo": COMBO, "speed": SPEED}[a.stage]
    if a.only:
        cfgs = [(n, o) for (n, o) in cfgs if a.only in n]

    base = None
    out: Dict[str, Any] = {}
    rowdump: Dict[str, Any] = {}
    print("部署口径 | seeds %d-%d (%d 局) | stage=%s" % (seeds[0], seeds[-1], len(seeds), a.stage))
    print("%-24s %8s %9s %11s %9s %9s %7s" %
          ("配置", "清除比例", "全清局", "平均定位清除", "总虚拟", "移动", "检测"))
    for name, over in cfgs:
        s = eval_cfg(over, seeds, jobs)
        if a.rows_out:
            rowdump[name] = {"overrides": over, "rows": _LAST_ROWS}
        out[name] = {"overrides": over, "summary": s}
        if name == "release":
            base = s
        d = ""
        if base is not None and name != "release":
            d = "%+7.1f" % (s["avg_clear"] - base["avg_clear"])
        print("%-24s %8.4f %4d/%-4d %8.1f %s %9.0f %9.0f %7.1f" %
              (name, s["ratio"], s["full_count"], s["n"], s["avg_clear"], d,
               s["virtual"], s["moved"], s["measures"]))
        if s["missed"]:
            print("     漏源局 (%d): %s" % (len(s["missed"]), s["missed"][:12]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": seeds, "stage": a.stage, "results": out}, fh,
                      ensure_ascii=False, indent=1)
        print("明细 ->", a.out)
    if a.rows_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.rows_out)), exist_ok=True)
        with open(a.rows_out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": seeds, "results": rowdump}, fh, ensure_ascii=False)
        print("逐局明细 ->", a.rows_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
