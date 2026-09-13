# -*- coding: utf-8 -*-
"""问题4 移植保真度对照: 仓库内模块 vs 沙盒单文件策略。

``robotdog/solver/sweeper4.py``（当前发布策略）与 ``sweeper4_ring18.py``
（上一版 18 站档）都是**单文件策略的移植版**。本工具把两者放在**同一批世界**
上逐局比对六个汇总量 —— 只要有一局不一致, 就说明移植过程中动到了决策逻辑
（而不只是外壳）。

沙盒默认带的布局常与仓库发布配置不同。为了检验**决策逻辑**而不是配置差异,
本工具默认把沙盒的布局参数（``ring_spec`` / ``ring_def`` / 相位 / 栅格）
显式注入仓库侧配置后再比对; ``--release`` 则用仓库发布配置, 用于量化
"换布局"带来的差异。

用法::

    # 发布策略 vs 它的单文件来源 (应逐位一致)
    set OPT1_SOLVER=C:\\Users\\ironi\\Desktop\\solver_q4.py
    python tools/probes/q4_opt1_parity.py --seeds 9500-9599

    # 上一版 18 站档 vs 沙盒 opt1 (逐位一致)
    set OPT1_SOLVER=C:\\Users\\ironi\\Desktop\\opt1\\solver.py
    python tools/probes/q4_opt1_parity.py --module sweeper4_ring18 --seeds 9500-9549

沙盒路径用环境变量 ``OPT1_SOLVER`` 指定; 找不到时会明确报错而不是默默跳过。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotdog.solver import opt1_port as ext                      # noqa: E402
from robotdog.solver.world import World                           # noqa: E402

#: 比对的汇总量 (顺序固定, 便于打印)
FIELDS = ("cleared_count", "virtual_t", "moved_m", "measures", "clears",
          "failed_clears")

#: 布局参数 (``--release`` 之外一律从沙盒侧注入, 只比较决策逻辑)
LAYOUT_KEYS = ("ring_spec", "ring_phase", "ring_def", "ring_r", "ring_n",
               "grid_d", "grid_rmax")


def snapshot(w) -> tuple:
    return tuple(getattr(w, f) for f in FIELDS)


def run_one(new, seed: int, release: bool = False) -> tuple:
    """跑同一局: 左边移植版, 右边沙盒版。

    ``release=False`` (默认) 把沙盒的布局参数注入移植版, 只比较决策逻辑;
    ``release=True`` 让移植版用仓库发布配置, 用于量化布局变更的影响。
    """
    wn = World(seed=seed, problem=4, **new.world_kwargs())
    cfg_new = new.build_cfg()
    if not release:
        cfg_ext = ext.build_cfg()
        for k in LAYOUT_KEYS:
            if hasattr(cfg_ext, k) and hasattr(cfg_new, k):
                setattr(cfg_new, k, getattr(cfg_ext, k))
    new.run_sweeper4(wn, cfg_new)
    we = World(seed=seed, problem=4, **ext.world_kwargs())
    ext.run_candidate(we, ext.build_cfg())
    return snapshot(wn), snapshot(we)


def main() -> int:
    ap = argparse.ArgumentParser(description="问题4 移植保真度对照")
    ap.add_argument("--module", default="sweeper4",
                    choices=("sweeper4", "sweeper4_ring18"),
                    help="仓库侧模块 (被对照的移植版)")
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--max-print", type=int, default=5, help="最多打印几局差异")
    ap.add_argument("--release", action="store_true",
                    help="移植版用仓库发布布局, 而不是沙盒布局")
    a = ap.parse_args()
    import importlib
    new = importlib.import_module("robotdog.solver." + a.module)
    seeds = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    same = 0
    diffs = []
    for sd in seeds:
        mine, theirs = run_one(new, sd, a.release)
        if mine == theirs:
            same += 1
        else:
            diffs.append((sd, mine, theirs))

    print("对照口径: %s" % ", ".join(FIELDS))
    print("仓库侧模块: robotdog.solver.%s" % a.module)
    print("沙盒策略: %s" % os.environ.get("OPT1_SOLVER", ext._DEFAULT_OPT1))
    print("站点布局: %s" % ("仓库发布配置" if a.release else "沙盒布局 (逻辑对照)"))
    print("逐位一致 %d/%d" % (same, len(seeds)))
    for sd, mine, theirs in diffs[:max(0, a.max_print)]:
        print("  seed %d\n    移植 %s\n    沙盒 %s" % (sd, mine, theirs))
    return 0 if same == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
