"""无线电干扰源环境模拟器 —— 核心引擎.

严格依据《模拟器使用说明》(附件1) 与《模拟器通信接口说明及编程指南》(附件2) 实现:

  * 几何与目标区域: 半径 1800 m 的圆域, 原点为圆心, x 正向为东, y 正向为北。
  * 干扰源: 10~16 个, 频道互不相同且属于 1..20; 有效接收半径 1000~1500 m;
    全向 / 定向两类, 定向源有效覆盖角为定向方向两侧各 90 度(含边界)。
  * 示向度误差: 单次检测误差在 [-1, 1] 度内, 输出 [0,360) 并保留 2 位小数。
  * 低速检测: 距离 <= 5 m 且在覆盖角内 -> measure_result="near"。
  * 清除半径: 20 m; 成功 5 s, 未发现 3 s; /clear 不切换频道。
  * 虚拟时间: 移动耗时 = 直线距离 / 5 m/s; /measure 频道切换 1 s、检测 5 s。
  * 限时: 虚拟世界活动时长上限 100 小时 = 360000 s (仅用于防止程序死循环);
    现实时间上限才是 20 分钟 = 1200 s; 另有 25 分钟测试窗口, 三者取最早到达者。

本模块不依赖第三方库, 可独立用于单元测试。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# 常量 (严格取自题面与附件2)
# --------------------------------------------------------------------------------------

ARENA_ID = "default"
ARENA_RADIUS_M = 1800.0
ROBOT_SPEED_MPS = 5.0
MAX_COORD_ABS = 2_000_000.0
CHANNEL_MIN, CHANNEL_MAX = 1, 20
SVD_ERROR_DEG = 1.0
NEAR_RADIUS_M = 5.0
CLEAR_RADIUS_M = 20.0
DETECT_DURATION_S = 5.0
CHANNEL_SWITCH_S = 1.0
CLEAR_LOCATE_S = 3.0
CLEAR_FIRE_S = 2.0
MAX_VIRTUAL_DURATION_S = 360_000.0
MAX_REAL_DURATION_S = 1200.0
WINDOW_DURATION_S = 25 * 60.0
COUNTDOWN_S = 5.0
BODY_LIMIT = 65_536
MAX_JSON_DEPTH = 16
ID_MAX_BYTES = {"robot_id": 64, "request_id": 128}
IDEMPOTENCY_LIMIT = 20_000
LOG_ENCRYPT_MAX_BYTES = 2 * 1024 * 1024

CONCURRENT_CONFLICT_MSG = "已有动作正在处理中: 不同新动作不得并发发送 (附件2 表2)"

ENTER_FIELDS = {"arena_id", "robot_id", "request_id"}
ACTION_FIELDS = {"arena_id", "robot_id", "request_id", "position", "channel"}
POSITION_FIELDS = {"x", "y"}

END_REASON_TEXT = {
    "user_exit": "机器狗主动调用 /exit 退出",
    "window_timeout": "25 分钟测试窗口超时",
    "program_timeout": "/enter 后程序运行超时 (最长 20 分钟)",
    "virtual_timeout": "虚拟世界活动时长超时 (100 小时 = 360000 s)",
    "manual_abort": "手工中止测试",
    "technical_error": "技术异常",
    "none": "测试进行中",
}


# --------------------------------------------------------------------------------------
# 协议层异常
# --------------------------------------------------------------------------------------


class ProtocolError(Exception):
    """需要以 HTTP 4xx/5xx 返回的错误。"""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.status = status
        self.code = code
        self.message = message


class RequestRejected(Exception):
    """HTTP 200 + accepted=false (业务拒绝)。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message


class _ReplayResponse(Exception):
    """内部信号: request_id 幂等重放。"""

    def __init__(self, response: Dict[str, Any]) -> None:
        super().__init__("replay")
        self.response = response


# --------------------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------------------


@dataclass
class InterferenceSource:
    channel: int
    x: float
    y: float
    radius_m: float
    kind: str = "omni"  # "omni" | "directional"
    direction_deg: Optional[float] = None
    alive: bool = True

    @property
    def is_omni(self) -> bool:
        return self.kind == "omni"

    def distance_to(self, px: float, py: float) -> float:
        return math.hypot(px - self.x, py - self.y)

    def in_coverage(self, px: float, py: float) -> bool:
        """位置是否在信号有效覆盖角度范围内。定向源为定向方向两侧各 90 度(含边界)。"""
        if self.is_omni:
            return True
        bearing = math.degrees(math.atan2(py - self.y, px - self.x)) % 360.0
        delta = abs(((bearing - float(self.direction_deg) + 180.0) % 360.0) - 180.0)
        return delta <= 90.0 + 1e-9

    def true_bearing(self, px: float, py: float) -> float:
        return math.degrees(math.atan2(self.y - py, self.x - px)) % 360.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "position": {"x": round(self.x, 6), "y": round(self.y, 6)},
            "valid_radius_m": round(self.radius_m, 6),
            "kind": "omni" if self.is_omni else "directional",
            "direction_deg": None if self.is_omni else round(float(self.direction_deg), 6),
            "cleared": not self.alive,
        }


@dataclass
class Case:
    """一局测试案例的真值数据 (机器狗不可直接读取)。"""

    code: str
    problem: int
    seed: int
    sources: List[InterferenceSource]
    created_at: str = ""

    @property
    def total(self) -> int:
        return len(self.sources)

    @property
    def omni_count(self) -> int:
        return sum(1 for s in self.sources if s.is_omni)

    @property
    def directional_count(self) -> int:
        return self.total - self.omni_count

    def by_channel(self, channel: int) -> Optional[InterferenceSource]:
        for s in self.sources:
            if s.channel == channel:
                return s
        return None

    def truth(self) -> List[Dict[str, Any]]:
        return [s.snapshot() for s in self.sources]


class CaseGenerator:
    """随机案例生成器 (支持固定随机种子以便复现)。"""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = int(seed) if seed is not None else int.from_bytes(os.urandom(8), "little")
        self.rng = random.Random(self.seed)

    def generate(self, problem: int) -> Case:
        rng = self.rng
        n = rng.randint(10, 16)
        channels = rng.sample(range(CHANNEL_MIN, CHANNEL_MAX + 1), n)
        sources: List[InterferenceSource] = []
        for ch in sorted(channels):
            r = ARENA_RADIUS_M * math.sqrt(rng.random())  # 圆域内均匀分布
            theta = rng.uniform(0.0, 2.0 * math.pi)
            x, y = r * math.cos(theta), r * math.sin(theta)
            radius = rng.uniform(1000.0, 1500.0)
            if problem == 4 and rng.random() < 0.5:
                sources.append(
                    InterferenceSource(
                        channel=ch,
                        x=x,
                        y=y,
                        radius_m=radius,
                        kind="directional",
                        direction_deg=rng.uniform(0.0, 360.0),
                    )
                )
            else:
                sources.append(
                    InterferenceSource(channel=ch, x=x, y=y, radius_m=radius, kind="omni")
                )
        code = "C%s-P%d-%s" % (time.strftime("%Y%m%d"), problem, uuid.uuid4().hex[:8].upper())
        return Case(
            code=code,
            problem=problem,
            seed=self.seed,
            sources=sources,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


class Clock:
    """单调现实时钟 (可注入假时钟以便确定性测试)。"""

    def __init__(self, fn: Optional[Callable[[], float]] = None) -> None:
        self._fn = fn or time.monotonic

    def now(self) -> float:
        return self._fn()


# --------------------------------------------------------------------------------------
# 运行时状态
# --------------------------------------------------------------------------------------


@dataclass
class RunStats:
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    measures: int = 0
    clears: int = 0
    clear_success: int = 0
    channel_switches: int = 0
    moved_distance_m: float = 0.0
    travel_time_s: float = 0.0
    detect_time_s: float = 0.0
    clear_time_s: float = 0.0
    switch_time_s: float = 0.0
    timeouts: int = 0
    end_reason: Optional[str] = None
    enter_wall: Optional[float] = None
    enter_monotonic: Optional[float] = None
    virtual_timeout_flagged: bool = False
    program_runtime_s: float = 0.0


@dataclass
class RunSession:
    """一局测试的可变状态。"""

    case: Case
    run_id: str
    phase: str = "preparing"  # preparing|countdown|armed|running|ended
    countdown_start: float = 0.0
    countdown_ends: float = 0.0
    window_start: Optional[float] = None
    window_deadline: Optional[float] = None
    real_deadline: Optional[float] = None
    remaining_real_duration_s: int = int(MAX_REAL_DURATION_S)
    virtual_us: int = 0
    pos: Tuple[float, float] = (0.0, 0.0)
    channel: int = 1
    entered: bool = False
    exited: bool = False
    actions: List[Dict[str, Any]] = field(default_factory=list)
    rejected_log: List[Dict[str, Any]] = field(default_factory=list)
    idempotency: Dict[str, Tuple[str, str, Dict[str, Any]]] = field(default_factory=dict)
    stats: RunStats = field(default_factory=RunStats)

    @property
    def cleared_count(self) -> int:
        return sum(1 for s in self.case.sources if not s.alive)

    def virtual_seconds(self) -> float:
        return self.virtual_us / 1_000_000.0


# --------------------------------------------------------------------------------------
# 核心引擎
# --------------------------------------------------------------------------------------


class SimulatorEngine:
    """模拟器核心: 动作校验、物理计算、计时与拒绝语义。"""

    def __init__(
        self,
        team_id: str = "MCM2026",
        clock: Optional[Clock] = None,
        countdown_s: float = COUNTDOWN_S,
        window_s: float = WINDOW_DURATION_S,
        max_real_s: float = MAX_REAL_DURATION_S,
        max_virtual_s: float = MAX_VIRTUAL_DURATION_S,
        svd_error_deg: float = SVD_ERROR_DEG,
        source_error_fn: Optional[Callable[[float, float, int], float]] = None,
    ) -> None:
        self.team_id = team_id
        self.clock = clock or Clock()
        self.countdown_s = float(countdown_s)
        self.window_s = float(window_s)
        self.max_real_s = float(max_real_s)
        self.max_virtual_s = float(max_virtual_s)
        self.svd_error_deg = float(svd_error_deg)
        self.run: Optional[RunSession] = None
        self.current_official = False
        self._rng = random.Random()
        self._error_fn = source_error_fn or self._default_error
        # 测试结束回调: 在串行闸门之外触发, 避免与动作执行互相等待
        self.on_run_end: Optional[Callable[[Dict[str, Any]], None]] = None
        self.end_report: Optional[Dict[str, Any]] = None
        # 串行闸门 (附件2 表2: 并发发送不同动作 -> 409)
        self._gate_lock = threading.Lock()
        self._gate_active: Optional[Dict[str, Any]] = None
        self.concurrent_wait_s = 30.0
        # 引擎内并发观测: 同时处于执行中的动作数 (串行闸门保证不超过 1)
        self.gate_active_count = 0
        self.gate_max_concurrent = 0
        # 测试钩子: 人为延长动作处理时间, 用于稳定复现"并发发送"的场景
        self.gate_hold_s = 0.0
        # 业务拒绝是否附带 reject_code / reject_message (默认关闭, 与附件2 §5.2 严格一致)
        self.verbose_reject = False

    def engine_concurrency_stats(self) -> Dict[str, int]:
        with self._gate_lock:
            return {
                "active": self.gate_active_count,
                "max_concurrent": self.gate_max_concurrent,
            }

    # ---------------------------------------------------------------- 会话生命周期

    def _default_error(self, px: float, py: float, channel: int) -> float:
        """示向度误差: 位于 [-1,1] 度内; 同一检测点电磁环境固定, 因此读数可复现。"""
        key = "%s|%.6f|%.6f|%d" % (self.run.run_id if self.run else "-", round(px, 6), round(py, 6), channel)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return (unit * 2.0 - 1.0) * self.svd_error_deg

    def new_run(self, problem: int, seed: Optional[int] = None, code: Optional[str] = None) -> RunSession:
        gen = CaseGenerator(seed)
        case = gen.generate(problem)
        if code:
            case.code = code
        run = RunSession(case=case, run_id=uuid.uuid4().hex, phase="preparing")
        self.run = run
        return run

    def arm(self) -> None:
        """案例数据准备就绪, 显示 5 秒倒计时。"""
        run = self._require_run()
        now = self.clock.now()
        run.phase = "countdown"
        run.countdown_start = now
        run.countdown_ends = now + self.countdown_s

    def tick(self) -> None:
        """按现实时间推进阶段; countdown -> armed(接口开放) -> running -> ended。"""
        run = self.run
        if run is None:
            return
        now = self.clock.now()
        if run.phase == "countdown":
            if now >= run.countdown_ends:
                run.phase = "armed"
                run.window_start = run.countdown_ends
                run.window_deadline = run.countdown_ends + self.window_s
            return
        if run.phase in ("armed", "running"):
            if run.stats.virtual_timeout_flagged or run.virtual_seconds() >= self.max_virtual_s:
                self.end_run("virtual_timeout")
                return
            if run.window_deadline is not None and now >= run.window_deadline:
                self.end_run("window_timeout")
                return
            if run.entered and run.real_deadline is not None and now >= run.real_deadline:
                self.end_run("program_timeout")
                return

    def end_run(self, reason: str) -> Optional[Dict[str, Any]]:
        """结束本局测试: 只做状态与报告构建, 回调由 release_ended_run() 在闸门外触发。

        幂等: 已经结束时直接返回 None, 不再重建报告 —— 报告要遍历整份动作日志,
        而并发超时路径 (tick / _check_budget) 可能重复调用本方法。
        """
        run = self._require_run()
        if run.phase == "ended":
            return None
        run.phase = "ended"
        run.stats.end_reason = reason
        if reason in ("window_timeout", "program_timeout", "virtual_timeout", "technical_error"):
            run.stats.timeouts += 1
        if run.stats.enter_monotonic is not None:
            run.stats.program_runtime_s = max(0.0, self.clock.now() - run.stats.enter_monotonic)
        self.end_report = self.build_report()
        return self.end_report

    def release_ended_run(self) -> None:
        """在串行闸门之外触发结束回调 (生成日志、发布事件)。必须幂等。"""
        report, self.end_report = self.end_report, None
        if report is None or self.on_run_end is None:
            return
        try:
            self.on_run_end(report)
        except Exception:
            pass

    def _require_run(self) -> RunSession:
        if self.run is None:
            raise RuntimeError("当前没有测试会话")
        return self.run

    # ---------------------------------------------------------------- 状态查询

    def interface_open(self) -> bool:
        run = self.run
        return run is not None and run.phase in ("armed", "running")

    def status(self) -> Dict[str, Any]:
        run = self.run
        now = self.clock.now()
        if run is None:
            return {"phase": "idle", "interface_open": False}
        info: Dict[str, Any] = {
            "phase": run.phase,
            "interface_open": self.interface_open(),
            "case_code": run.case.code,
            "problem": run.case.problem,
            "official": self.current_official,
            "virtual_time_s": round(run.virtual_seconds(), 6),
            "cleared": run.cleared_count,
            # 演练测试可显示案例真值, 正式测试不显示 (附件1 4.6)
            "source_total": None if self.current_official else run.case.total,
            "entered": run.entered,
            "exited": run.exited,
            "end_reason": run.stats.end_reason,
            "end_reason_text": END_REASON_TEXT.get(run.stats.end_reason or "none", ""),
            "action_count": len(run.actions),
        }
        if run.phase == "countdown":
            info["countdown_left_s"] = max(0.0, round(run.countdown_ends - now, 3))
        if run.window_deadline is not None:
            info["window_left_s"] = max(0.0, round(run.window_deadline - now, 3))
        if run.entered and run.real_deadline is not None:
            info["program_left_s"] = max(0.0, round(run.real_deadline - now, 3))
        info["virtual_left_s"] = max(0.0, round(self.max_virtual_s - run.virtual_seconds(), 6))
        return info

    # ---------------------------------------------------------------- 响应构造

    @staticmethod
    def _envelope(accepted: bool) -> Dict[str, Any]:
        return {
            "accepted": accepted,
            "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": 0,
        }

    def _ok(self, run: RunSession) -> Dict[str, Any]:
        # 热路径: 直接拼字典, 省掉 _envelope 的一次调用与一次键改写。
        return {
            "accepted": True,
            "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": round(run.virtual_us / 1_000_000.0, 6),
        }

    def _reject(self, run: Optional[RunSession], code: str, message: str) -> Dict[str, Any]:
        """业务拒绝响应: HTTP 200 + accepted=false。

        附件2 §5.2 规定此时"只包含 accepted、real_timestamp_ms、virtual_time_s" 三个字段,
        因此默认不附加任何自定义字段。调试需要时可设置 verbose_reject=True,
        额外返回 reject_code / reject_message (机器狗程序只应以 accepted 为判据)。
        """
        resp = self._envelope(False)
        if self.verbose_reject:
            resp["reject_code"] = code
            resp["reject_message"] = message
        if run is not None:
            run.rejected_log.append(
                {
                    "wall_monotonic": round(self.clock.now(), 3),
                    "code": code,
                    "message": message,
                    "virtual_time_s": round(run.virtual_seconds(), 6),
                }
            )
        return resp

    # ---------------------------------------------------------------- 校验

    def _check_identifier(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str):
            raise ProtocolError(400, "invalid_%s" % field_name, "%s 必须是字符串" % field_name)
        raw = value.encode("utf-8")
        limit = ID_MAX_BYTES[field_name]
        if len(raw) < 1 or len(raw) > limit:
            raise ProtocolError(
                400,
                "invalid_%s" % field_name,
                "%s 的 UTF-8 长度必须在 1~%d 字节之间" % (field_name, limit),
            )
        for ch in value:
            o = ord(ch)
            if o < 0x20 or o == 0x7F or 0x200B <= o <= 0x200F or 0x202A <= o <= 0x202E or o in (0xFEFF, 0x2060):
                raise ProtocolError(
                    400, "invalid_%s" % field_name, "%s 不能包含控制字符或不可见格式字符" % field_name
                )
        return value

    def _check_position(self, payload: Dict[str, Any]) -> Tuple[float, float]:
        if "position" not in payload:
            raise ProtocolError(400, "missing_field", "缺少 position")
        position = payload["position"]
        if not isinstance(position, dict):
            raise ProtocolError(400, "invalid_position", "position 必须是对象")
        for key in position:
            if key not in POSITION_FIELDS:
                raise RequestRejected("unknown_field", "position 下存在未知字段: %s" % key)
        for axis in ("x", "y"):
            if axis not in position:
                raise ProtocolError(400, "missing_field", "缺少 position.%s" % axis)
            value = position[axis]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProtocolError(400, "invalid_position", "position.%s 必须是有限数值" % axis)
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ProtocolError(400, "invalid_position", "position.%s 必须是有限数值" % axis)
            if abs(float(value)) > MAX_COORD_ABS:
                raise ProtocolError(
                    400, "invalid_position", "position.%s 绝对值不得超过 2000000" % axis
                )
        return float(position["x"]), float(position["y"])

    def _check_channel(self, payload: Dict[str, Any]) -> int:
        if "channel" not in payload:
            raise ProtocolError(400, "missing_field", "缺少 channel")
        channel = payload["channel"]
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ProtocolError(400, "invalid_channel", "channel 必须是 1..20 的整数")
        if isinstance(channel, float):
            if math.isnan(channel) or math.isinf(channel) or not channel.is_integer():
                raise ProtocolError(400, "invalid_channel", "channel 必须是 1..20 的整数")
        channel = int(channel)
        if channel < CHANNEL_MIN or channel > CHANNEL_MAX:
            raise ProtocolError(400, "invalid_channel", "channel 必须在 1..20 范围内")
        return channel

    def _check_declared_types(self, payload: Dict[str, Any]) -> None:
        """信封级类型校验 (无副作用), 先于任何业务判定执行。"""
        self._check_identifier(payload["robot_id"], "robot_id")
        self._check_identifier(payload["request_id"], "request_id")
        if not isinstance(payload["arena_id"], str):
            raise ProtocolError(400, "invalid_arena_id", "arena_id 必须是字符串")

    def _declared_ok(self, action: str, payload: Dict[str, Any]) -> None:
        allowed = ENTER_FIELDS if action in ("enter", "exit") else ACTION_FIELDS
        extra = [k for k in payload if k not in allowed]
        if extra:
            raise RequestRejected("unknown_field", "存在未声明字段: %s" % ", ".join(sorted(extra)))

    def _check_envelope(self, action: str, payload: Dict[str, Any]) -> Tuple[str, str]:
        for required in ("arena_id", "robot_id", "request_id"):
            if required not in payload:
                raise ProtocolError(400, "missing_field", "缺少 %s" % required)
        self._declared_ok(action, payload)
        # 类型错误 (HTTP 400) 优先于业务拒绝 (HTTP 200 + accepted=false)
        self._check_declared_types(payload)
        if payload["arena_id"] != ARENA_ID:
            raise RequestRejected("arena_id_mismatch", "arena_id 必须是 %r" % ARENA_ID)
        robot_id = payload["robot_id"]
        if robot_id != self.team_id:
            raise RequestRejected("robot_id_mismatch", "robot_id 必须逐字节等于当前登录参赛队号")
        return robot_id, payload["request_id"]

    def _require_open(self) -> RunSession:
        run = self.run
        if run is None:
            raise RequestRejected("no_test", "当前没有测试")
        if run.phase == "preparing":
            raise RequestRejected("not_open", "案例数据准备中, 机器狗接口未开放")
        if run.phase == "countdown":
            raise RequestRejected("not_open", "5 秒倒计时尚未结束, 机器狗接口未开放")
        if run.phase == "ended":
            raise RequestRejected("test_ended", "测试已经结束")
        return run

    def _require_entered(self, run: RunSession) -> None:
        if not run.entered:
            raise RequestRejected("not_entered", "尚未成功调用 /enter")
        if run.exited:
            raise RequestRejected("already_exited", "机器狗已经退出")

    def _check_budget(self, run: RunSession) -> None:
        """到达截止时刻的请求不再执行。"""
        now = self.clock.now()
        if run.stats.virtual_timeout_flagged or run.virtual_seconds() >= self.max_virtual_s:
            self.end_run("virtual_timeout")
            raise RequestRejected("virtual_timeout", "虚拟世界时长已达上限")
        if run.window_deadline is not None and now >= run.window_deadline:
            self.end_run("window_timeout")
            raise RequestRejected("window_timeout", "25 分钟测试窗口已到")
        if run.entered and run.real_deadline is not None and now >= run.real_deadline:
            self.end_run("program_timeout")
            raise RequestRejected("program_timeout", "程序运行时间已到")

    # ---------------------------------------------------------------- 物理

    @staticmethod
    def travel_seconds(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / ROBOT_SPEED_MPS

    def _advance(self, run: RunSession, delta_s: float) -> None:
        if delta_s <= 0.0:
            return
        if run.virtual_seconds() + delta_s > self.max_virtual_s:
            run.stats.virtual_timeout_flagged = True
            run.virtual_us = int(round(self.max_virtual_s * 1_000_000))
            return
        run.virtual_us += int(round(delta_s * 1_000_000))

    # ---------------------------------------------------------------- 串行闸门
    #
    # 附件2 表2: "并发发送了不同动作 -> 409"。模拟器必须保证同一时刻只有一个动作在
    # 引擎内执行: 若已有动作在执行, 并发到达的新动作直接返回 409; 若并发到达的是
    # "同一个动作的重试"(同 request_id + 同内容 + 同路径, 常见于网络重试), 则等待在途
    # 动作结束后按幂等规则返回其首次响应, 不重复执行。

    @staticmethod
    def _error_envelope(code: str, message: str, status: int = 409) -> Dict[str, Any]:
        """错误响应体 (accepted=false, virtual_time_s=0)。"""
        return {
            "accepted": False,
            "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": 0,
            "error": code,
            "message": message,
            "http_status": status,
        }

    def _wait_for_in_flight(self) -> None:
        """等待在途动作结束 (仅同 request_id 的网络重试会走到这里)。"""
        deadline = time.monotonic() + self.concurrent_wait_s
        while True:
            with self._gate_lock:
                active = self._gate_active
                if active is None:
                    return
                event = active["event"]
            if event.wait(0.05) or time.monotonic() >= deadline:
                return

    def _acquire_gate(self, request_id: Optional[str], fingerprint: Optional[str]) -> Optional[Dict[str, Any]]:
        """尝试占用引擎执行权。返回 None 表示已获得; 返回响应体表示并发冲突 (409)。"""
        deadline = time.monotonic() + self.concurrent_wait_s
        while True:
            with self._gate_lock:
                active = self._gate_active
                if active is None:
                    self._gate_active = {
                        "request_id": request_id,
                        "fingerprint": fingerprint,
                        "event": threading.Event(),
                    }
                    # 占用成功后才计数: 这样 "engine_active/engine_max_concurrent"
                    # 统计的是真正在执行的动作, 而不是被 409 打回的尝试。
                    # 串行闸门保证该计数只在 0/1 之间取值。
                    self.gate_active_count = 1
                    if self.gate_max_concurrent < 1:
                        self.gate_max_concurrent = 1
                    return None
                same_in_flight = (
                    isinstance(request_id, str)
                    and active.get("request_id") == request_id
                    and active.get("fingerprint") == fingerprint
                )
            if not same_in_flight:
                # 不同新动作并发发送 -> 409 (附件2 表2)
                return self._error_envelope("concurrent_request", CONCURRENT_CONFLICT_MSG)
            if time.monotonic() >= deadline:
                return self._error_envelope("concurrent_request", CONCURRENT_CONFLICT_MSG)
            self._wait_for_in_flight()
            if time.monotonic() >= deadline:
                return self._error_envelope("concurrent_request", CONCURRENT_CONFLICT_MSG)

    def _release_gate(self) -> None:
        with self._gate_lock:
            active = self._gate_active
            self._gate_active = None
            self.gate_active_count = 0
        if active is not None:
            active["event"].set()

    def handle(self, action: str, payload: Any) -> Dict[str, Any]:
        """动作入口 (含串行闸门): 返回标准化的 JSON 业务响应; 结构错误抛 ProtocolError。

        同一时刻只允许一个动作在引擎内执行 (附件2 表2: 并发发送不同动作 -> 409);
        同一 request_id 且内容相同的网络重试会等待在途动作结束, 然后按幂等规则返回首次响应。
        """
        is_dict = isinstance(payload, dict)
        request_id = payload.get("request_id") if is_dict else None
        # 指纹在此计算一次并向下传递: 原先 handle/_handle_locked/_check_replay
        # 各自重算一次, 单次动作要序列化 3 遍请求体。
        fingerprint = self._fingerprint(action, payload) if is_dict else None
        request_id = request_id if isinstance(request_id, str) else None
        conflict = self._acquire_gate(request_id, fingerprint)
        if conflict is not None:
            return conflict
        try:
            if self.gate_hold_s > 0.0:
                time.sleep(self.gate_hold_s)  # 仅用于并发测试
            return self._handle_locked(action, payload, request_id, fingerprint)
        finally:
            self._release_gate()
            # 若本次动作结束了测试, 在闸门之外触发结束回调
            self.release_ended_run()

    def _handle_locked(
        self,
        action: str,
        payload: Any,
        request_id: Optional[Any],
        fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if not isinstance(payload, dict):
                raise ProtocolError(400, "invalid_body", "请求体必须是 JSON 对象")
            self._check_replay(action, payload, fingerprint)
            # 幂等记录上限必须先于 _dispatch 判定: 否则动作已经执行并改变了模拟器
            # 状态, 429 却把真实响应丢掉, 机器狗看到的是"未执行"。
            run = self.run
            if run is not None and isinstance(request_id, str) and len(run.idempotency) >= IDEMPOTENCY_LIMIT:
                raise ProtocolError(429, "idempotency_limit", "本局幂等记录达到上限")
            response = self._dispatch(action, payload)
        except _ReplayResponse as replay:
            return replay.response
        except RequestRejected as exc:
            run = self.run
            return self._reject(run, exc.code, exc.message)
        except ProtocolError as exc:
            run = self.run
            if run is not None:
                run.stats.errors += 1
                run.rejected_log.append(
                    {
                        "wall_monotonic": round(self.clock.now(), 3),
                        "code": exc.code,
                        "message": exc.message,
                        "http_status": exc.status,
                    }
                )
            raise
        run = self.run
        if run is not None and isinstance(request_id, str):
            if fingerprint is None:
                fingerprint = self._fingerprint(action, payload)
            run.idempotency[request_id] = (action, fingerprint, response)
        if run is not None:
            if response.get("accepted") is True:
                run.stats.accepted += 1
            else:
                run.stats.rejected += 1
        return response

    def _fingerprint(self, action: str, payload: Dict[str, Any]) -> str:
        try:
            body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            body = repr(payload)
        return "%s|%s" % (action, body)

    def _check_replay(
        self, action: str, payload: Dict[str, Any], fingerprint: Optional[str] = None
    ) -> None:
        run = self.run
        if run is None:
            return
        rid = payload.get("request_id")
        if not isinstance(rid, str):
            return
        prev = run.idempotency.get(rid)
        if prev is None:
            return
        prev_action, prev_fp, prev_resp = prev
        if fingerprint is None:
            fingerprint = self._fingerprint(action, payload)
        if prev_action == action and prev_fp == fingerprint:
            raise _ReplayResponse(dict(prev_resp))
        raise ProtocolError(409, "request_id_conflict", "同一 request_id 对应了不同动作或不同内容")

    def _dispatch(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if action == "enter":
            return self._do_enter(payload)
        if action == "exit":
            return self._do_exit(payload)
        if action == "measure":
            return self._do_measure(payload)
        if action == "clear":
            return self._do_clear(payload)
        raise ProtocolError(500, "unknown_action", "未知动作 %s" % action)

    # ---------------------------------------------------------------- /enter

    def _do_enter(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._check_envelope("enter", payload)
        run = self._require_open()
        if run.entered:
            return self._reject(run, "duplicate_enter", "重复调用 /enter")
        now = self.clock.now()
        run.entered = True
        run.phase = "running"
        run.stats.enter_wall = time.time()
        run.stats.enter_monotonic = now
        remaining_window = 0.0
        if run.window_deadline is not None:
            remaining_window = run.window_deadline - now
        remaining = int(math.floor(max(0.0, min(self.max_real_s, remaining_window))))
        run.remaining_real_duration_s = remaining
        run.real_deadline = now + remaining
        resp = self._ok(run)
        resp["max_virtual_duration_s"] = self.max_virtual_s
        resp["max_real_duration_s"] = self.max_real_s
        resp["remaining_real_duration_s"] = remaining
        self._record(run, "enter", None, None, 0.0, 0.0, 0.0, resp)
        return resp

    # ---------------------------------------------------------------- /exit

    def _do_exit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._check_envelope("exit", payload)
        run = self._require_open()
        self._require_entered(run)
        resp = self._ok(run)
        resp["exit_reason"] = "user_exit"
        run.exited = True
        self._record(run, "exit", None, None, 0.0, 0.0, 0.0, resp)
        self.end_run("user_exit")
        return resp

    # ---------------------------------------------------------------- /measure

    def _do_measure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._check_envelope("measure", payload)
        # 坐标/频道类型校验 (HTTP 400) 必须先于业务判定, 且只做一次:
        # 原先 _check_declared_types 与这里各校验一遍, 每个 /measure 白花约 0.9 µs。
        x, y = self._check_position(payload)
        channel = self._check_channel(payload)
        run = self._require_open()
        self._require_entered(run)
        self._check_budget(run)

        move_s = self.travel_seconds(run.pos, (x, y))
        switch_s = CHANNEL_SWITCH_S if channel != run.channel else 0.0
        self._advance(run, move_s + switch_s + DETECT_DURATION_S)

        run.stats.measures += 1
        run.stats.moved_distance_m += move_s * ROBOT_SPEED_MPS
        run.stats.travel_time_s += move_s
        run.stats.detect_time_s += DETECT_DURATION_S
        run.stats.switch_time_s += switch_s
        if switch_s:
            run.stats.channel_switches += 1

        source = run.case.by_channel(channel)
        result: Dict[str, Any]
        if source is not None and source.alive:
            distance = source.distance_to(x, y)
            if distance <= NEAR_RADIUS_M and source.in_coverage(x, y):
                # 附件2 7.3: near 响应只含 4 个标准字段, 不返回 svd_deg
                result = {"measure_result": "near"}
            elif distance <= source.radius_m and source.in_coverage(x, y):
                svd = (source.true_bearing(x, y) + self._error_fn(x, y, channel)) % 360.0
                result = {"measure_result": "direction", "svd_deg": round(svd, 2) + 0.0}
            else:
                result = {"measure_result": "no_signal"}
        else:
            result = {"measure_result": "no_signal"}

        run.pos = (x, y)
        run.channel = channel
        resp = self._ok(run)
        resp.update(result)
        self._record(run, "measure", {"x": x, "y": y}, channel, move_s, switch_s, DETECT_DURATION_S, resp)
        return resp

    # ---------------------------------------------------------------- /clear

    def _do_clear(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._check_envelope("clear", payload)
        x, y = self._check_position(payload)
        channel = self._check_channel(payload)
        run = self._require_open()
        self._require_entered(run)
        self._check_budget(run)

        move_s = self.travel_seconds(run.pos, (x, y))
        source = run.case.by_channel(channel)
        hit = source is not None and source.alive and source.distance_to(x, y) <= CLEAR_RADIUS_M
        # /clear 先光学精确定位(3 s), 命中后再激光清除(2 s); 未命中只有定位耗时
        action_s = CLEAR_LOCATE_S + (CLEAR_FIRE_S if hit else 0.0)
        self._advance(run, move_s + action_s)

        run.stats.clears += 1
        run.stats.moved_distance_m += move_s * ROBOT_SPEED_MPS
        run.stats.travel_time_s += move_s
        run.stats.clear_time_s += action_s
        if hit:
            source.alive = False
            run.stats.clear_success += 1

        run.pos = (x, y)
        resp = self._ok(run)
        resp["clear_result"] = "success" if hit else "no_target_in_range"
        self._record(run, "clear", {"x": x, "y": y}, channel, move_s, 0.0, action_s, resp)
        return resp

    # ---------------------------------------------------------------- 记录

    def _record(
        self,
        run: RunSession,
        action: str,
        position: Optional[Dict[str, float]],
        channel: Optional[int],
        move_s: float,
        switch_s: float,
        action_s: float,
        response: Dict[str, Any],
    ) -> None:
        entry = {
            "seq": len(run.actions) + 1,
            "action": action,
            "position": None if position is None else {"x": round(position["x"], 6), "y": round(position["y"], 6)},
            "channel": channel,
            "move_distance_m": round(move_s * ROBOT_SPEED_MPS, 6),
            "move_duration_s": round(move_s, 6),
            "switch_duration_s": round(switch_s, 6),
            "action_duration_s": round(action_s, 6),
            "total_duration_s": round(move_s + switch_s + action_s, 6),
            "virtual_time_s": round(run.virtual_seconds(), 6),
            "response": {k: v for k, v in response.items() if k != "real_timestamp_ms"},
        }
        run.actions.append(entry)

    # ---------------------------------------------------------------- 报告

    def build_report(self, reason: Optional[str] = None) -> Dict[str, Any]:
        run = self._require_run()
        case = run.case
        run_stats = run.stats
        # 虚拟时间只取一次: total_duration_s 与 virtual_time_s 必须是同一个值。
        virtual_s = run.virtual_seconds()
        cleared = run.cleared_count
        total = case.total
        omni = case.omni_count
        end_reason = reason or run_stats.end_reason or "none"
        runtime = run_stats.program_runtime_s
        if run.entered and run_stats.enter_monotonic is not None and run.phase != "ended":
            runtime = max(0.0, self.clock.now() - run_stats.enter_monotonic)
        return {
            "simulator": "无线电干扰源环境模拟器 (2026 高教社杯 B 题 · 自建复现版)",
            "protocol": "附件2《模拟器通信接口说明及编程指南》v1.0",
            "case_code": case.code,
            "problem": case.problem,
            "seed": case.seed,
            "case_created_at": case.created_at,
            "end_reason": end_reason,
            "end_reason_text": END_REASON_TEXT.get(end_reason, end_reason),
            "stats": {
                "source_total": total,
                "cleared_count": cleared,
                "cleared_ratio": round(cleared / total, 6) if total else 0.0,
                "total_duration_s": round(virtual_s, 6),
                # 注意: 平均值必须用未取整的 virtual_s 计算, 否则会与 4.x 位上的历史值不符
                "avg_clear_duration_s": round(virtual_s / cleared, 6) if cleared else None,
                "program_runtime_s": round(runtime, 3),
                "virtual_time_s": round(virtual_s, 6),
                "moved_distance_m": round(run_stats.moved_distance_m, 6),
                "travel_duration_s": round(run_stats.travel_time_s, 6),
                "switch_duration_s": round(run_stats.switch_time_s, 6),
                "detect_duration_s": round(run_stats.detect_time_s, 6),
                "clear_duration_s": round(run_stats.clear_time_s, 6),
                "measure_count": run_stats.measures,
                "clear_attempts": run_stats.clears,
                "clear_success": run_stats.clear_success,
                "channel_switches": run_stats.channel_switches,
                "rejected_requests": run_stats.rejected,
                "error_requests": run_stats.errors,
                "timeouts": run_stats.timeouts,
                "final_position": {"x": round(run.pos[0], 6), "y": round(run.pos[1], 6)},
                "final_channel": run.channel,
            },
            "truth": {
                "source_total": total,
                "omni_count": omni,
                "directional_count": total - omni,
                "sources": case.truth(),
            },
            "uncleared_sources": [s.snapshot() for s in case.sources if s.alive],
            "actions": list(run.actions),
            "rejected": list(run.rejected_log),
        }


# --------------------------------------------------------------------------------------
# 行为日志 (加密) 与导出
# --------------------------------------------------------------------------------------

LOG_KEY = b"MCM2026-B-SIMULATOR-LOG-KEY-v1"


def _keystream(key: bytes, counter: int) -> bytes:
    return hashlib.sha256(key + counter.to_bytes(8, "big")).digest()


def _xor_bytes(data: bytes, key: bytes, counter: int = 0) -> bytes:
    """用 SHA256 密钥流对 data 做 XOR。

    原先写法是 ``bytes(b ^ stream[i] for i, b in enumerate(payload))`` —— 每个字节
    都要走一次 Python 层循环, 900 KB 的日志因此要花上百毫秒。这里改成大整数
    异或: 把密钥流与数据各当作一个正整数, 一次 ``^`` 就由 C 完成全部字节异或,
    结果与逐字节异或逐位相同。
    """
    n = len(data)
    if n == 0:
        return b""
    stream = bytearray()
    while len(stream) < n:
        stream += _keystream(key, counter)
        counter += 1
    return (int.from_bytes(data, "big") ^ int.from_bytes(stream[:n], "big")).to_bytes(n, "big")


def encrypt_log(plaintext: bytes, key: bytes = LOG_KEY) -> bytes:
    """轻量加密行为日志: 魔数 + 长度 + 流式异或 + CRC32。"""
    payload = zlib.compress(plaintext, 9)
    body = _xor_bytes(payload, key)
    return b"ENC1" + len(payload).to_bytes(8, "big") + body + zlib.crc32(plaintext).to_bytes(4, "big")


def decrypt_log(blob: bytes, key: bytes = LOG_KEY) -> bytes:
    if blob[:4] != b"ENC1":
        raise ValueError("不是加密日志文件")
    n = int.from_bytes(blob[4:12], "big")
    body = blob[12 : 12 + n]
    crc = int.from_bytes(blob[12 + n : 16 + n], "big")
    plain = zlib.decompress(_xor_bytes(body, key))
    if zlib.crc32(plain) != crc:
        raise ValueError("日志校验失败")
    return plain


def export_log(
    report: Dict[str, Any], out_dir: str, official: bool = True
) -> Dict[str, Any]:
    """导出行为日志。正式测试生成加密 .log, 演练测试生成可读 .json。"""
    os.makedirs(out_dir, exist_ok=True)
    kind = "formal" if official else "practice"
    stem = "%s_p%d_%s" % (kind, int(report.get("problem", 3)), str(report["case_code"]))
    plain = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    paths: Dict[str, Any] = {}
    if official:
        blob = encrypt_log(plain)
        enc_path = os.path.join(out_dir, stem + ".log")
        with open(enc_path, "wb") as fh:
            fh.write(blob)
        paths["encrypted"] = enc_path
        paths["encrypted_size"] = len(blob)
        paths["over_upload_limit"] = len(blob) > LOG_ENCRYPT_MAX_BYTES
    text_path = os.path.join(out_dir, stem + ".json")
    # 直接写字节: plain 已经是 UTF-8, 文本模式还要再 decode 出一份同样大的字符串。
    with open(text_path, "wb") as fh:
        fh.write(plain)
    paths["readable"] = text_path
    paths["readable_size"] = len(plain)
    paths["stem"] = stem
    return paths
