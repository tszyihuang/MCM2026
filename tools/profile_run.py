"""逐步剖析: 一局里时间与移动都花在哪些决策上.

**这是"时间去哪儿了"的权威口径**。

阶段划分 (与 ``sweeper.run_sweeper`` 的主循环一一对应):
  initial              起点盲扫 (全场的第 1 个动作)
  leg_stop             去清除的路上顺路停车检测
  pursue               逼近/定位/清除过程中的检测与清除
  sweep_after_clear    清除成功后的原地补测 (零移动)
  dedicated_probe      专程信息探测 (best_info_point 选点)
  hunt                 收尾集合覆盖探测 (cover_point 选点)

实现方式 (不改被测代码):
  * 在 ``World`` **类**上包装 ``measure``/``clear``, 逐动作记录 (阶段, 耗时, 移动距离);
    逐动作移动距离之和恒等于 ``world.moved_m`` (脚本自检并打印);
  * 阶段在动作发生的**那一刻**用 ``sys._getframe`` 走调用栈判定 —— 调用链是确定的:
      run_sweeper -> ops.probe      -> World.measure   (起点盲扫 / 收尾 hunt)
      run_sweeper -> 原语 -> probe  -> World.measure   (其它阶段)
    原语名 (pursue_fast / leg_info_stop / best_info_point / cover_point) 唯一确定阶段。
  这样既有明确的作用域, 又不依赖源码行号 (行号方案在缩进/注释上反复出错)。

  **约定**: 被测实现需保留这些函数名 (``pursue_fast`` / ``sweep_here`` / ``probe_at`` /
  ``leg_info_stop`` / ``best_info_point`` / ``cover_point``), 否则该阶段会并入相邻阶段。

用法::

    python tools\profile_run.py --module robotdog.solver.sweeper --seeds 9500-9549
    python tools\profile_run.py --module robotdog.solver.sweeper --seeds 9500-9549 --deploy
    python tools\profile_run.py --module robotdog.solver.sweeper --seed 9542 --trace
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robotdog.consts import SPEED_MPS  # noqa: E402
from robotdog.solver.eval import parse_seeds  # noqa: E402

# 调用栈上出现这些函数名 => 该阶段
BY_FUNC = {
    "pursue_fast": "pursue",
    "pursue": "pursue",
    "leg_info_stop": "leg_stop",
}


# 包装只安装一次 (World 是类, 进程内共享), 因此事件接收器必须可换。
_SINK: Dict[str, Any] = {"events": [], "state": {"n": 0}, "leg": {"active": False},
                         "mark": {"p": ""}, "module": ""}


def _install_leg_mark(module_name: str) -> None:
    """标记 ``run_sweeper`` 主循环里"当前正在处理一条 leg"。

    ``leg_info_stop`` 只在该段内被调用, 而紧跟其后的 ``pursue_fast`` 结束该段。
    于是 "leg 内发生的 probe_at" = 顺路停车 (移动本来就要走), "leg 外" = 真正专程探测。
    """
    sw = importlib.import_module(module_name)
    if getattr(sw, "_leg_mark_installed", False):
        return
    _SINK["mark"]["p"] = ""
    for name, start in (("leg_info_stop", True), ("pursue_fast", False),
                        ("pursue", False)):
        fn = getattr(sw, name, None)
        if fn is None:
            continue

        def make(f, s):
            def inner(*a, **kw):
                old = _SINK["leg"]["active"]
                if s:
                    _SINK["leg"]["active"] = True
                try:
                    return f(*a, **kw)
                finally:
                    if not s:
                        _SINK["leg"]["active"] = old
            return inner
        setattr(sw, name, make(fn, start))
    sw._leg_mark_installed = True  # type: ignore[attr-defined]


_WRAPPER = frozenset(("measure", "clear", "classify", "_stack_names"))
_DEBUG: Optional[List[Any]] = None      # 设为 list 时记录每次分类看到的调用栈 (调试用)


def _stack_names(limit: int = 10) -> List[str]:
    """调用栈上的函数名, **跳过包装帧** (names[0] 即真正发起动作的原语)。"""
    out: List[str] = []
    f = sys._getframe(2)
    while f is not None and len(out) < limit:
        name = f.f_code.co_name
        if name not in _WRAPPER:
            out.append(name)
        f = f.f_back
    return out


def _phase_of(names: List[str]) -> str:
    direct = names[0] if names else ""
    if "pursue_fast" in names or "pursue" in names:
        return "pursue"
    if direct == "sweep_here":
        return "initial" if _SINK["state"]["n"] == 0 else "sweep_after_clear"
    if direct == "probe":
        up1 = names[1] if len(names) > 1 else ""
        if up1 == "sweep_here":
            return "initial" if _SINK["state"]["n"] == 0 else "sweep_after_clear"
        if up1 == "probe_at":
            # ``leg_info_stop`` 选出的停车点**落在去目标的直线段上**, 因此这段移动本来
            # 就要走 (逐段对账: pursue + legstop 的移动减去到真值的直线
            # 距离恰好等于 approach detour)。把它们算成"专程探测移动"会严重高估探测成本
            # (实测量级 2258 m/game vs 真正的专程移动 78~105 m/game), 所以这里单列。
            return "leg_stop" if _SINK["leg"]["active"] or "leg_info_stop" in names \
                else "dedicated_probe"
        if up1 == "run_sweeper":
            return "initial" if _SINK["state"]["n"] == 0 else "hunt"
    if direct == "probe_at":
        return "leg_stop" if _SINK["leg"]["active"] else "dedicated_probe"
    return "other"


def classify() -> str:
    """在**动作发起的那一刻**判定阶段。

    实测的调用链 (见 ``World.measure`` 的直接调用者, 包装帧已剔除):
      ``probe -> sweep_here -> run_sweeper``   原地补测 (起点盲扫 / 清除后)
      ``probe -> probe_at   -> run_sweeper``   专程探测 (或顺路停车, 见 ``_leg_here``)
      ``pursue_fast         -> run_sweeper``   逼近途中
      ``probe -> run_sweeper``                 起点盲扫 (第 1 个动作) / 收尾 hunt
    """
    return _phase_of(_stack_names())


def _install_world() -> None:
    from robotdog.solver.world import World
    if getattr(World, "_profile_patched", False):
        return
    orig_m, orig_c = World.measure, World.clear

    def measure(self, x, y, channel):
        px, py = self.x, self.y
        ph = classify()
        info = orig_m(self, x, y, channel)
        _SINK["state"]["n"] += 1
        _SINK["events"].append((ph, info.dt, math.hypot(x - px, y - py)))
        return info

    def clear(self, x, y, channel):
        px, py = self.x, self.y
        ph = classify()
        info = orig_c(self, x, y, channel)
        _SINK["state"]["n"] += 1
        _SINK["events"].append((ph, info.dt, math.hypot(x - px, y - py)))
        return info

    World.measure = measure        # type: ignore[assignment]
    World.clear = clear            # type: ignore[assignment]
    World._profile_patched = True  # type: ignore[attr-defined]


def _load_entry(module_name: str):
    """按 ``tools/candidates.py`` 的同一约定解析入口与配置 (保证两边跑同一策略)。

    **关键**: 模块没有 ``build_cfg()`` 时必须回退到 ``sweeper.build_cfg()``, 不能传
    ``cfg=None`` —— 那会退化成默认 ``SweepConfig()``, 静默换成另一套参数。
    """
    mod = importlib.import_module(module_name)
    from robotdog.solver.sweeper import build_cfg as release_cfg, run_sweeper
    cfg = mod.build_cfg() if hasattr(mod, "build_cfg") else release_cfg()
    fn = getattr(mod, "run_candidate", None) or run_sweeper
    takes_cfg = len(inspect.signature(fn).parameters) >= 2
    return fn, cfg, takes_cfg


def profile_one(module_name: str, seed: int, trace: bool = False,
                knows_total: bool = True) -> Dict[str, Any]:
    from robotdog.solver.world import World

    fn, cfg, takes_cfg = _load_entry(module_name)
    events: List[Tuple[str, float, float]] = []
    _SINK["events"] = events
    _SINK["state"] = {"n": 0}
    _SINK["leg"] = {"active": False}
    _install_world()
    _install_leg_mark(module_name)

    w = World(seed=seed)
    kwargs: Dict[str, Any] = {"trace": trace, "knows_total": knows_total}
    if takes_cfg:
        fn(w, cfg, **kwargs)
    else:
        fn(w, **kwargs)

    agg: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0, "dt": 0.0, "moved": 0.0})
    for ph, dt, moved in events:
        agg[ph]["n"] += 1
        agg[ph]["dt"] += dt
        agg[ph]["moved"] += moved
    return {
        "seed": seed, "cleared": w.cleared_count, "total": w.n_sources,
        "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
        "clears": w.clears, "failed_clears": w.failed_clears,
        "sum_moved": sum(m for _p, _d, m in events),
        "phases": {k: dict(v) for k, v in agg.items()},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="逐步时间/移动剖析")
    ap.add_argument("--module", default="robotdog.solver.sweeper")
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--seed", type=int, default=0, help="单局剖析 (优先于 --seeds)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--deploy", action="store_true",
                    help="用部署口径 (knows_total=False) 剖析; 正式测试就是这一档")
    args = ap.parse_args(argv)

    seeds = [args.seed] if args.seed else parse_seeds(args.seeds)
    tot: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "dt": 0.0, "moved": 0.0})
    virt = mv = smv = 0.0
    for sd in seeds:
        r = profile_one(args.module, sd, trace=args.trace and len(seeds) == 1,
                        knows_total=not args.deploy)
        virt += r["virtual_s"]
        mv += r["moved_m"]
        smv += r["sum_moved"]
        for k, v in r["phases"].items():
            for f in ("n", "dt", "moved"):
                tot[k][f] += v[f]
    n = len(seeds)
    print("=== %s | %d 局 | %s ===" % (args.module, n,
                                       "部署口径 knows_total=False" if args.deploy
                                       else "进程内口径 knows_total=True"))
    print("每局: 虚拟 %.0f s | 移动 %.0f m (%.0f s) | 检测+清除+切换 %.0f s | 移动自检 %s"
          % (virt / n, mv / n, mv / SPEED_MPS / n, (virt - mv / SPEED_MPS) / n,
             "OK" if abs(smv - mv) < 1e-6 else "不一致 %.1f vs %.1f" % (smv / n, mv / n)))
    print()
    print("%-22s %9s %9s %9s %7s" % ("阶段", "动作数/局", "秒/局", "移动m/局", "占比"))
    for k in sorted(tot, key=lambda k: -tot[k]["dt"]):
        v = tot[k]
        print("%-22s %9.1f %9.1f %9.0f %6.1f%%"
              % (k, v["n"] / n, v["dt"] / n, v["moved"] / n, 100.0 * v["dt"] / max(virt, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
