"""问题3 / 问题4 的进程内精确复刻环境 (机器狗可见信息 与 真值 严格分离)。

物理规则与 simulator.core 逐位一致 (已由 tools/tests/test_solver_equiv.py 逐动作验证):
移动 5 m/s, 检测 5 s, 切换频道 1 s (仅 /measure 计), 清除命中 5 s / 未发现 3 s,
近距阈值 5 m, 清除半径 20 m, 有效接收半径 U(1000,1500), 示向度误差 ∈ [-1°,1°]
且同一检测点重复检测读数不变。

问题4 追加: 每个源以 1/2 概率为**定向源**, 定向方向 U(0,360);
定向源只在"定向方向两侧各 90 度"内辐射, 覆盖角外**收不到信号** (附件1 第3条)。
注意两条容易搞反的规则:

* ``/measure`` 的三个分支 (**near / direction / no_signal**) 都要过覆盖角;
  覆盖角外是 ``no_signal``, 而不是 ``near``。
* ``/clear`` 的成功判据**与覆盖角无关** (附件2 第8条只说距离)。
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .belief import Belief
from ..consts import (ARENA_RADIUS_M,
                     CHANNEL_SWITCH_S,
                     CLEAR_FIRE_S,
                     CLEAR_LOCATE_S,
                     CLEAR_RADIUS_M,
                     DETECT_DURATION_S,
                     NEAR_RADIUS_M,
                     N_CHANNELS,
                     RADIUS_MAX_M,
                     RADIUS_MIN_M,
                     SPEED_MPS,
                     SVD_ERROR_DEG)


@dataclass
class Source:
    channel: int
    x: float
    y: float
    radius_m: float
    alive: bool = True
    kind: str = "omni"                      # "omni" | "directional" (问题4)
    direction_deg: Optional[float] = None   # 定向方向 psi, 全向源为 None

    @property
    def is_omni(self) -> bool:
        return self.kind == "omni"

    def in_coverage(self, px: float, py: float) -> bool:
        """(px,py) 是否在信号有效覆盖角度范围内 (定向源为定向方向两侧各 90 度, 含边界)。

        与 ``simulator.core.InterferenceSource.in_coverage`` 逐位一致。
        """
        if self.direction_deg is None:
            return True
        bearing = math.degrees(math.atan2(py - self.y, px - self.x)) % 360.0
        delta = abs(((bearing - float(self.direction_deg) + 180.0) % 360.0) - 180.0)
        return delta <= 90.0 + 1e-9

    def true_bearing(self, px: float, py: float) -> float:
        return math.degrees(math.atan2(self.y - py, self.x - px)) % 360.0


def _draw_case(seed: int, problem: int) -> List[Source]:
    """核心生成循环: 问题3 与问题4 共用, 且与 ``CaseGenerator.generate`` 逐位一致。

    随机数消耗顺序必须与模拟器完全相同 (``simulator/core.py::CaseGenerator``):
    1) ``randint(10,16)`` 个数;  2) ``sample(1..20, n)`` 频道 (升序遍历);
    3) 每个源 ``sqrt(random())`` 半径、``uniform(0,2pi)`` 角度、``uniform(1000,1500)`` 接收半径;
    4) **仅问题4**: 再消耗一个 ``random()`` 判定是否定向, 定向时再消耗一个 ``uniform(0,360)``。
    """
    rng = random.Random(int(seed))
    n = rng.randint(10, 16)
    channels = rng.sample(range(1, N_CHANNELS + 1), n)
    out: List[Source] = []
    for ch in sorted(channels):
        r = ARENA_RADIUS_M * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x, y = r * math.cos(theta), r * math.sin(theta)
        radius = rng.uniform(RADIUS_MIN_M, RADIUS_MAX_M)
        if problem == 4 and rng.random() < 0.5:
            out.append(Source(ch, x, y, radius, kind="directional",
                              direction_deg=rng.uniform(0.0, 360.0)))
        else:
            out.append(Source(ch, x, y, radius))
    return out


def generate_case(seed: int) -> List[Source]:
    """问题3 案例 (全向源); 与 ``CaseGenerator.generate(problem=3)`` 逐位一致。"""
    return _draw_case(seed, 3)


def generate_case_q4(seed: int) -> List[Source]:
    """问题4 案例 (含定向源); 与 ``CaseGenerator.generate(problem=4)`` 逐位一致。"""
    return _draw_case(seed, 4)


@dataclass
class StepInfo:
    kind: str                     # measure / clear
    result: str = ""              # direction / near / no_signal / success / no_target_in_range
    svd_deg: Optional[float] = None
    dt: float = 0.0
    channel: int = 0
    x: float = 0.0
    y: float = 0.0


class World:
    """一局问题3/问题4。对外只暴露机器狗可见信息 (world.belief), 真值仅供评测。"""

    def __init__(self, seed: Optional[int] = None, run_id: str = "RL",
                 deterministic_error: bool = True, problem: int = 3,
                 belief_cls: Optional[type] = None) -> None:
        if seed is None:
            seed = random.randrange(1 << 30)
        self.seed = int(seed)
        self.run_id = run_id
        self.problem = int(problem)
        self.sources: List[Source] = (_draw_case(self.seed, self.problem)
                                      if self.problem == 4 else generate_case(self.seed))
        self.by_channel: Dict[int, Source] = {s.channel: s for s in self.sources}
        self.n_sources = len(self.sources)
        self.n_directional = sum(1 for s in self.sources if not s.is_omni)
        self.x = 0.0
        self.y = 0.0
        self.channel = 1
        self.virtual_t = 0.0
        self.belief = belief_cls() if belief_cls is not None else Belief()
        self.cleared: List[int] = []
        self.actions = 0
        self.measures = 0
        self.clears = 0
        self.failed_clears = 0
        self.moved_m = 0.0
        #: 分项虚拟时间 (仅问题4 的剖析工具读取; 策略不得使用)
        self.time_travel = 0.0
        self.time_switch = 0.0
        self.time_detect = 0.0
        self.time_clear = 0.0
        self.deterministic_error = deterministic_error
        self._err_cache: Dict[Tuple[float, float, int], float] = {}
        self._rng = random.Random((self.seed << 1) ^ 0x9E3779B9)

    # ---------------- 真值 (评测/调试) ----------------
    def truth(self) -> List[Tuple[int, float, float, float, bool]]:
        return [(s.channel, s.x, s.y, s.radius_m, s.alive) for s in self.sources]

    def truth_q4(self) -> List[Dict[str, object]]:
        """带源型与定向方向的真值明细 (仅用于探针/报告, 策略不得读取)。"""
        return [{"channel": s.channel, "x": s.x, "y": s.y, "radius_m": s.radius_m,
                 "kind": s.kind, "direction_deg": s.direction_deg, "alive": s.alive}
                for s in self.sources]

    @property
    def cleared_count(self) -> int:
        return len(self.cleared)

    @property
    def all_cleared(self) -> bool:
        return len(self.cleared) == self.n_sources

    @property
    def pos(self) -> Tuple[float, float]:
        return (self.x, self.y)

    # ---------------- 误差模型 ----------------
    def _error(self, x: float, y: float, channel: int) -> float:
        if not self.deterministic_error:
            return self._rng.uniform(-SVD_ERROR_DEG, SVD_ERROR_DEG)
        key = (round(x, 6), round(y, 6), channel)
        v = self._err_cache.get(key)
        if v is None:
            h = hashlib.sha256(("%s|%.6f|%.6f|%d" % (self.run_id, x, y, channel)).encode()).digest()
            unit = int.from_bytes(h[:8], "big") / float(1 << 64)
            v = (unit * 2.0 - 1.0) * SVD_ERROR_DEG
            self._err_cache[key] = v
        return v

    # ---------------- 动作 ----------------
    def measure(self, x: float, y: float, channel: int) -> StepInfo:
        move_s = math.hypot(x - self.x, y - self.y) / SPEED_MPS
        switch_s = CHANNEL_SWITCH_S if channel != self.channel else 0.0
        dt = move_s + switch_s + DETECT_DURATION_S
        self.virtual_t += dt
        self.time_travel += move_s
        self.time_switch += switch_s
        self.time_detect += DETECT_DURATION_S
        self.x, self.y, self.channel = float(x), float(y), int(channel)
        self.actions += 1
        self.measures += 1
        self.moved_m += move_s * SPEED_MPS
        src = self.by_channel.get(channel)
        if src is not None and src.alive:
            d = math.hypot(x - src.x, y - src.y)
            if d <= NEAR_RADIUS_M and src.in_coverage(x, y):
                self.belief.on_measure(x, y, channel, "near", None, self.virtual_t)
                return StepInfo("measure", "near", None, dt, channel, x, y)
            if d <= src.radius_m and src.in_coverage(x, y):
                svd = (math.degrees(math.atan2(src.y - y, src.x - x)) + self._error(x, y, channel)) % 360.0
                svd = round(svd, 2) + 0.0
                self.belief.on_measure(x, y, channel, "direction", svd, self.virtual_t)
                return StepInfo("measure", "direction", svd, dt, channel, x, y)
        self.belief.on_measure(x, y, channel, "no_signal", None, self.virtual_t)
        return StepInfo("measure", "no_signal", None, dt, channel, x, y)

    def clear(self, x: float, y: float, channel: int) -> StepInfo:
        move_s = math.hypot(x - self.x, y - self.y) / SPEED_MPS
        src = self.by_channel.get(channel)
        hit = src is not None and src.alive and math.hypot(x - src.x, y - src.y) <= CLEAR_RADIUS_M
        dt = move_s + CLEAR_LOCATE_S + (CLEAR_FIRE_S if hit else 0.0)
        self.virtual_t += dt
        self.time_travel += move_s
        self.time_clear += CLEAR_LOCATE_S + (CLEAR_FIRE_S if hit else 0.0)
        self.x, self.y = float(x), float(y)
        self.actions += 1
        self.clears += 1
        self.moved_m += move_s * SPEED_MPS
        if hit:
            src.alive = False
            self.cleared.append(channel)
            self.belief.on_clear(x, y, channel, True)
            return StepInfo("clear", "success", None, dt, channel, x, y)
        self.failed_clears += 1
        self.belief.on_clear(x, y, channel, False)
        return StepInfo("clear", "no_target_in_range", None, dt, channel, x, y)
