"""单文件策略（``strategy(ctx, scenario=None)`` 形态）到进程内 ``World`` 的适配层。

单文件策略的入口是 ``strategy(ctx, scenario=None)``，只使用
``ctx.enter`` / ``ctx.measure`` / ``ctx.clear`` / ``ctx.exit`` 四个动作，
返回字段为 ``measure_result`` / ``svd_deg`` / ``clear_result``。
本模块把主工程的 :class:`robotdog.solver.world.World` 包装成同样的 ``ctx``，
从而可以直接用 ``tools/eval_q4.py`` 评测**任意**这种形态的单文件策略 ——
发布策略 ``robotdog/solver/sweeper4.py`` 是它的第一个使用方。

默认加载**本仓库的发布策略** ``robotdog/solver/sweeper4.py``（因此
``--modules robotdog.solver.opt1_port`` 与 ``--planners sweeper4`` 等价，
可用于交叉校验）；设置环境变量 ``OPT1_SOLVER`` 可换成任意外部单文件::

    # 沙盒 18 站版（上一版）
    set OPT1_SOLVER=C:\\Users\\ironi\\Desktop\\opt1\\solver.py
    python tools/eval_q4.py --modules robotdog.solver.opt1_port --seeds 9500-9519

    # 发布策略的单文件来源（应与 sweeper4 逐位一致）
    set OPT1_SOLVER=C:\\Users\\ironi\\Desktop\\solver_q4.py
    python tools/probes/q4_opt1_parity.py --seeds 9500-9599
"""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

from .world import World


_DEFAULT_OPT1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "sweeper4.py")
_opt1_mod = None


class _NullBelief:
    """opt1 不做贝叶斯推断；只需要满足 ``World`` 的事件回调接口。"""

    def on_measure(self, mx, my, channel, kind, svd, t) -> None:
        return None

    def on_clear(self, mx, my, channel, hit) -> None:
        return None


def _load_opt1():
    """加载 opt1 单文件策略（每个进程只加载一次）。"""
    global _opt1_mod
    if _opt1_mod is not None:
        return _opt1_mod
    path = os.path.abspath(os.path.expanduser(
        os.environ.get("OPT1_SOLVER", _DEFAULT_OPT1)))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "找不到 opt1 策略文件: %s\n"
            "请设置环境变量 OPT1_SOLVER 指向 opt1/solver.py" % path)
    spec = importlib.util.spec_from_file_location("opt1_stream_solver", path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 opt1 策略文件: %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _opt1_mod = mod
    return mod


class _Opt1Ctx:
    """把主工程 ``World`` 包装成 opt1 期望的 ``ctx`` 接口。"""

    def __init__(self, world: World) -> None:
        self.world = world

    def enter(self) -> Dict[str, Any]:
        # 进程内 World 从 (0,0)、频道 1、虚拟时间 0 开始，无需额外进入动作。
        return {"accepted": True}

    def exit(self) -> Dict[str, Any]:
        return {"accepted": True}

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        info = self.world.measure(float(x), float(y), int(channel))
        out: Dict[str, Any] = {"measure_result": info.result}
        if info.result == "direction" and info.svd_deg is not None:
            out["svd_deg"] = float(info.svd_deg)
        return out

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        info = self.world.clear(float(x), float(y), int(channel))
        return {"clear_result": info.result}


def build_cfg() -> SimpleNamespace:
    """把 opt1 的 ``PARAMS`` 复制成 ``eval_q4.py --set`` 可覆盖的配置对象。"""
    mod = _load_opt1()
    cfg = SimpleNamespace()
    for k, v in mod.PARAMS.items():
        setattr(cfg, k, v)
    return cfg


def world_kwargs() -> Dict[str, Any]:
    # opt1 不做贝叶斯推断，用空信念只保留 World 的物理/计时账本。
    return {"belief_cls": _NullBelief}


def run_candidate(world: World, cfg: Optional[Any] = None,
                  knows_total: bool = True, trace: bool = False,
                  **kwargs: Any) -> Dict[str, Any]:
    """在给定 ``World`` 上执行 opt1 策略。"""
    mod = _load_opt1()
    if cfg is not None:
        for k, v in vars(cfg).items():
            if k in mod.PARAMS:
                mod.PARAMS[k] = v
    mod.strategy(_Opt1Ctx(world), None)
    return {"cleared": len(world.cleared), "virtual_s": world.virtual_t}


# ``eval_q4.py`` 会优先找 ``run_sweeper4``，这里显式暴露同一入口。
run_sweeper4 = run_candidate
