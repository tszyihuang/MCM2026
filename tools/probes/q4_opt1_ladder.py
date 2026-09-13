# -*- coding: utf-8 -*-
"""问题4 调度器阶梯 (A~E) 与单旋钮消融的可复现重跑工具。

输出与 ``data/reports/q4_opt1_ladder500.json`` 同结构 (名字 -> {overrides, summary}),
供 ``docs/问题4论文.md`` 表 8/表 9 直接取数。

用法::

    python tools/probes/q4_opt1_ladder.py --seeds 9500-9999 --jobs 16 \
        --out data/reports/q4_opt1_ladder500.json

注意: 布局 (``ring_spec``) 用的是**发布配置**, 所以启用/放宽布局后再跑,
阶梯与消融的绝对值会跟着变, 结论 (相对差) 才是要读的东西。

2026-09-13 起 ``robotdog.solver.sweeper4`` 已换成 **16 站 + 中继测量版**
（22 站零漏检档只差 ``PARAMS['ring_def']`` 一行, 18 站旧档留档为 ``sweeper4_ring18.py``）,
实测约 422 s/源。本脚本的 A~E 档位与消融旋钮在各代里同名同义, 因此可以直接重跑;
但绝对值会整体变快, 与 ``data/reports/q4_opt1_ladder500.json`` 里 18 站档的历史数字
不可混比。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.eval_q4 import parse_seeds, run_config              # noqa: E402

MODULE = "robotdog.solver.sweeper4"

#: (报告名, 覆盖参数) —— 顺序即论文中的阅读顺序
CONFIGS: List[Tuple[str, Dict[str, Any]]] = [
    ("A  基线: 贪心最近任务", {"planner": "A"}),
    ("B  每步重解 NN + 2-opt", {"planner": "E", "plan_stability": 0, "plan_ls": 1}),
    ("C  B + Or-opt(1..3)", {"planner": "E", "plan_stability": 0, "plan_ls": 2}),
    ("D  C + Or-opt(1..5)", {"planner": "E", "plan_stability": 0, "plan_ls": 3}),
    ("E  计划稳定 + cheapest-ins + 2opt/Or3", {"plan_ls": 2, "plan_stability": 1}),
    ("E-prime  E 换 Or-opt(1..5)", {"plan_ls": 3, "plan_stability": 1}),
    ("E-dprime E 换多起点", {"plan_ls": 4, "plan_stability": 1}),
    ("消融: E 去掉计划稳定性 (=C)", {"plan_ls": 2, "plan_stability": 0}),
    ("消融: E 去掉局部搜索", {"plan_ls": 0, "plan_stability": 1}),
    ("发布配置 (E + skip_far + ray_recheck)", {}),
    ("skip_far=0 (关掉补测剪枝)", {"skip_far": 0.0}),
    ("skip_far_k=0 (不留 σ 余量)", {"skip_far_k": 0.0}),
    ("skip_far=1700 (更保守)", {"skip_far": 1700.0}),
    ("ray_recheck=0", {"ray_recheck": 0}),
    ("min_obs=3 (攒三条示向度)", {"min_obs": 3}),
    ("creep_step=30", {"creep_step": 30.0}),
    ("creep_step=44", {"creep_step": 44.0}),
    ("relay=0 (关掉中继测量)", {"relay": 0}),
    ("relay_cap=2", {"relay_cap": 2}),
    ("fix_sigma_ok=1e9 (无条件跳过已交会)", {"fix_sigma_ok": 1e9}),
    ("hunt_mode=ray", {"hunt_mode": "ray"}),
    ("hunt_probe=300", {"hunt_probe": 300.0}),
    ("zero_ring_stride=2", {"zero_ring_stride": 2}),
    ("zero_inner=0 (内圈不再补测零观测)", {"zero_inner": 0}),
    ("opp_try=1", {"opp_try": 1}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="问题4 阶梯与消融重跑")
    ap.add_argument("--seeds", default="9500-9999")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", nargs="*", default=[], help="只跑名字里含这些子串的档")
    a = ap.parse_args()

    seeds = parse_seeds(a.seeds)
    out: Dict[str, Any] = {}
    t0 = time.time()
    for name, over in CONFIGS:
        if a.only and not any(s in name for s in a.only):
            continue
        r = run_config(MODULE, seeds, over, a.jobs, problem=4, deploy=True)
        out[name] = {"overrides": over, "summary": r["summary"]}
        s = r["summary"]
        print("%-40s 清除率 %.4f  全清 %4d/%-4d  %7.2f s/源  移动 %6.0f m  检测 %5.1f"
              % (name, s["ratio"], s["full"], s["n"], s["avg_clear"],
                 s["moved"], s["measures"]), flush=True)
    print("墙钟 %.0f s | %d 局/档 | %d 档" % (time.time() - t0, len(seeds), len(out)))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("明细 ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
