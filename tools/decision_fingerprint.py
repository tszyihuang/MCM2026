# -*- coding: utf-8 -*-
"""重构安全网: 逐动作决策指纹 (behavioral fingerprint).

动机
----
本仓库对**逐位可复现**有硬性要求: 测点坐标差 1 ULP 就足以让 ``is_measured`` 的
去重翻转, 整局轨迹随之分叉 (见 ``sweeper.pursue_fast`` 的注释)。因此"重构成没改
行为"不能只看一批案例的平均分 —— 平均分相同也可能是两条不同的轨迹。

本脚本把一局的**离散决策序列**取出来做指纹: 每一次 /measure 与 /clear 的
(位置取整到 1e-6, 频道, 结果) 依次喂进 SHA1。轨迹只要有一个动作不同, 指纹就变。

用法::

    python tools/decision_fingerprint.py                    # 打印全部口径的指纹
    python tools/decision_fingerprint.py --seeds 9500-9799  # 指定种子
    python tools/decision_fingerprint.py --json out.json    # 存成 JSON 便于 diff

输出里的 ``selftest`` 行必须两次相同 —— 否则说明策略本身不确定, 指纹不可用作判据。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 口径清单: (名字, 模块, 入口函数, knows_total)
CASES: List[Tuple[str, str, str, bool]] = [
    ("q3", "robotdog.solver.sweeper", "run_sweeper", True),
    ("q3_deploy", "robotdog.solver.sweeper", "run_sweeper", False),
]

# 每个口径该配哪个 cfg 工厂
CFG_FACTORY: Dict[str, str] = {
    "q3": "build_cfg",
    "q3_deploy": "build_cfg",
}

#: 需要额外覆盖的配置 (口径名 -> {字段: 值})
CFG_OVERRIDES: Dict[str, Dict[str, Any]] = {}


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


def run_case(name: str, module: str, entry: str, knows_total: bool,
             seeds: List[int]) -> Dict[str, Any]:
    import importlib
    from robotdog.solver.world import World

    mod = importlib.import_module(module)
    cfg = getattr(mod, CFG_FACTORY[name])()
    for k, v in CFG_OVERRIDES.get(name, {}).items():
        setattr(cfg, k, v)
    fn = getattr(mod, entry)
    # 规划器自带世界装配 (信念类因路线而异), 见模块 docstring
    wk = mod.world_kwargs() if hasattr(mod, "world_kwargs") else {}

    # 记录每一次动作: (位置, 频道, 结果)
    trace: List[str] = []
    orig_measure, orig_clear = World.measure, World.clear

    def measure(self, x, y, channel):  # type: ignore[no-untyped-def]
        info = orig_measure(self, x, y, channel)
        trace.append("m|%.6f|%.6f|%d|%s" % (x, y, channel, info.result))
        return info

    def clear(self, x, y, channel):  # type: ignore[no-untyped-def]
        info = orig_clear(self, x, y, channel)
        trace.append("c|%.6f|%.6f|%d|%s" % (x, y, channel, info.result))
        return info

    World.measure, World.clear = measure, clear  # type: ignore[assignment]
    h = hashlib.sha1()
    stats = {"cleared": 0, "total": 0, "virtual_s": 0.0, "moved_m": 0.0,
             "measures": 0, "clears": 0}
    try:
        for sd in seeds:
            w = World(seed=sd, **wk)
            del trace[:]
            if knows_total:
                fn(w, cfg)
            else:
                fn(w, cfg, knows_total=False)
            h.update(("S%d\n" % sd).encode())
            h.update(("\n".join(trace) + "\n").encode())
            stats["cleared"] += w.cleared_count
            stats["total"] += w.n_sources
            stats["virtual_s"] += w.virtual_t
            stats["moved_m"] += w.moved_m
            stats["measures"] += w.measures
            stats["clears"] += w.clears
    finally:
        World.measure, World.clear = orig_measure, orig_clear  # type: ignore[assignment]

    n = len(seeds)
    return {
        "digest": h.hexdigest()[:16],
        "actions": len(trace),
        "ratio": round(stats["cleared"] / max(stats["total"], 1), 6),
        "full": stats["cleared"] == stats["total"],
        "avg_clear_s": round(stats["virtual_s"] / max(stats["cleared"], 1), 2),
        "moved_m": round(stats["moved_m"] / n, 1),
        "measures": round(stats["measures"] / n, 1),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="逐动作决策指纹 (重构安全网)")
    ap.add_argument("--seeds", default="9500-9799")
    ap.add_argument("--json", default="", help="把结果写成 JSON")
    ap.add_argument("--selftest", action="store_true",
                    help="重复跑第一个口径两次, 确认策略本身是确定性的")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        ap.error("--seeds 解析为空")

    cases = list(CASES)

    print("种子 %d 局 (%d-%d)  口径 %d 条" % (len(seeds), seeds[0], seeds[-1], len(cases)))
    print("%-20s %-18s %8s %8s %10s %9s %8s" %
          ("口径", "digest", "清除比例", "全清", "平均清除s", "移动m", "检测"))
    out: Dict[str, Any] = {"seeds": [seeds[0], seeds[-1], len(seeds)], "cases": {}}
    t0 = time.time()
    for name, module, entry, knows_total in cases:
        r = run_case(name, module, entry, knows_total, seeds)
        out["cases"][name] = r
        print("%-20s %-18s %8.4f %8s %10.1f %9.0f %8.1f"
              % (name, r["digest"], r["ratio"], "是" if r["full"] else "否",
                 r["avg_clear_s"], r["moved_m"], r["measures"]), flush=True)
    if args.selftest:
        print("\n[selftest] 同一口径连跑两次 (digest 必须相同):")
        for name, module, entry, knows_total in cases[:2]:
            a = run_case(name, module, entry, knows_total, seeds[:40])
            b = run_case(name, module, entry, knows_total, seeds[:40])
            ok = a["digest"] == b["digest"]
            print("  %-20s %s  %s vs %s" % (name, "确定性 OK" if ok else "**不确定**",
                                            a["digest"], b["digest"]))
            out.setdefault("selftest", {})[name] = ok
    out["wall_s"] = round(time.time() - t0, 1)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("\nJSON ->", args.json)
    print("总耗时 %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
