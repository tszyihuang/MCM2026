"""测试工具: 可注入假时钟的模拟器实例 + HTTP 客户端封装。"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.core import Clock, InterferenceSource, SimulatorEngine  # noqa: E402
from simulator.server import SimulatorServer  # noqa: E402

TEAM = "T2026TESTS"


class FakeClock:
    """可手动推进的单调时钟。"""

    def __init__(self, t0: float = 10_000.0) -> None:
        self.t = float(t0)

    def __call__(self) -> float:
        return self.t

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


class SimHarness:
    """在临时端口上启动模拟器, 提供 raw HTTP 请求能力。"""

    def __init__(
        self,
        sources: Optional[list] = None,
        team_id: str = TEAM,
        countdown_s: float = 0.0,
        window_s: float = 1500.0,
        max_real_s: float = 1200.0,
        max_virtual_s: float = 360_000.0,
        fake_clock: Optional[FakeClock] = None,
        error_enabled: bool = False,
        verbose_reject: bool = True,
    ) -> None:
        self.fake_clock = fake_clock if fake_clock is not None else FakeClock()
        self.engine = SimulatorEngine(
            team_id=team_id,
            clock=Clock(self.fake_clock),
            countdown_s=countdown_s,
            window_s=window_s,
            max_real_s=max_real_s,
            max_virtual_s=max_virtual_s,
            source_error_fn=None if error_enabled else (lambda px, py, ch: 0.0),
        )
        # 单元测试默认打开调试字段 (reject_code/reject_message) 以便断言拒绝原因;
        # 严格字段集由 test_protocol.RejectFieldSetTest 单独验证。
        self.engine.verbose_reject = verbose_reject
        self.team_id = team_id
        self.server = SimulatorServer(self.engine, port=0)
        self.port = self.server.start()
        self._sources = sources
        self._seq = 0
        self._seq_lock = threading.Lock()

    # ---- 生命周期 ----
    def start_run(self, sources: Optional[list] = None, problem: int = 3, code: str = "TESTCASE") -> None:
        run = self.engine.new_run(problem, seed=1, code=code)
        if sources is not None:
            run.case.sources = list(sources)
        elif self._sources is not None:
            run.case.sources = list(self._sources)
        if self.engine.countdown_s <= 0.0:
            # 免倒计时模式: 直接开放接口
            now = self.fake_clock.now()
            run.phase = "armed"
            run.window_start = now
            run.window_deadline = now + self.engine.window_s
        else:
            self.engine.arm()
            self.fake_clock.advance(self.engine.countdown_s + 1e-6)
        self.engine.tick()

    def close(self) -> None:
        self.server.stop()

    # ---- 原始 HTTP ----
    def raw(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Any, Dict[str, str]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            hdrs = {"Content-Type": "application/json"}
            if headers:
                hdrs.update(headers)
            conn.request(method, path, body=body, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            parsed: Any
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else None
            except Exception:
                parsed = raw
            return resp.status, parsed, dict(resp.getheaders())
        finally:
            conn.close()

    # ---- 便捷动作 ----
    def rid(self, prefix: str = "r") -> str:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        return "%s-%d" % (prefix, seq)

    def base(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return {"arena_id": "default", "robot_id": self.team_id, "request_id": request_id or self.rid()}

    def action(self, x: float, y: float, channel: int, request_id: Optional[str] = None) -> Dict[str, Any]:
        payload = self.base(request_id)
        payload["position"] = {"x": x, "y": y}
        payload["channel"] = channel
        return payload

    def post(self, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        status, body, _ = self.raw("POST", path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return status, body

    def enter(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self.post("/enter", self.base(request_id))[1]

    def measure(self, x: float, y: float, ch: int, request_id: Optional[str] = None) -> Dict[str, Any]:
        """返回响应体 (单次检测)。"""
        return self.post("/measure", self.action(x, y, ch, request_id))[1]

    def measure_status(self, x: float, y: float, ch: int, request_id: Optional[str] = None):
        return self.post("/measure", self.action(x, y, ch, request_id))

    def clear(self, x: float, y: float, ch: int, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self.post("/clear", self.action(x, y, ch, request_id))[1]

    def clear_status(self, x: float, y: float, ch: int, request_id: Optional[str] = None):
        return self.post("/clear", self.action(x, y, ch, request_id))

    def exit(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        return self.post("/exit", self.base(request_id))[1]


def src(channel: int, x: float, y: float, radius: float = 1500.0, kind: str = "omni", direction: float = 0.0):
    return InterferenceSource(
        channel=channel, x=x, y=y, radius_m=radius, kind=kind, direction_deg=direction
    )
