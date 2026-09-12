"""机器狗程序的 HTTP+JSON 客户端 (纯标准库).

通信严格遵循附件2: 只使用 /enter、/measure、/clear、/exit 四条指令;
每个新动作使用新的 request_id, 逐次等待响应; 网络重试复用原 request_id 与请求内容;
同时检查 HTTP 状态与 accepted 字段; 现实预算取自 /enter 的 remaining_real_duration_s。

这里的预算是**唯一**的时间口径来源: 其它策略 (含 ``solver``) 一律通过本客户端
读取 ``remaining_real_s``, 不自己另算一套 —— 本项目历史上正因"另算一套时间口径"
出过把虚拟上限当现实预算的严重缺陷。
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .consts import MAX_VIRTUAL_S
from .geometry import clamp_coord


class RobotClient:
    """与模拟器通信的客户端: 负责重试、幂等、时间预算与请求日志。"""

    def __init__(
        self,
        base_url: str,
        team_id: str,
        log_path: Optional[str] = None,
        request_timeout: float = 10.0,
        max_retry: int = 4,
        transport: Optional[Callable[[str, Dict[str, Any]], Tuple[int, Dict[str, Any]]]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.team_id = team_id
        self.log_path = log_path
        self.request_timeout = request_timeout
        self.max_retry = max_retry
        # 现实耗时来源。真实运行时是墙钟; 进程内回归工具可注入自推进时钟,
        # 使 1200 s 预算闸门可复现 —— 否则同一 seed 每跑一次结果都不同
        # (每局的预算判断取决于当次运行的机器负载), 参数对照就失去意义。
        self.clock = clock or time.time
        self.counter = 0
        self.virtual_time_s = 0.0
        self.remaining_real_s: Optional[float] = None
        self.max_real_s: float = 1200.0
        # 虚拟世界活动时长上限, /enter 成功后由模拟器报告值覆盖。
        # 仅作兜底安全网, 不是约束 (附件1 §2.5)。
        self.max_virtual_s: float = MAX_VIRTUAL_S
        self.entered_wall: Optional[float] = None
        self._transport = transport
        self._fh = None
        # 串行发送保证与并发冲突统计 (附件2 5.3: 不得并发发送不同动作)
        self._send_lock = threading.Lock()
        self._in_flight: Optional[str] = None
        self.serial_violations = 0
        self.conflict_retries = 0
        self._conn: Optional["http.client.HTTPConnection"] = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    # ---- 内部 ----
    def _next_id(self, prefix: str) -> str:
        self.counter += 1
        return "%s-%d" % (prefix, self.counter)

    def _close_conn(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _conn_for(self, timeout: float) -> "http.client.HTTPConnection":
        """取得复用连接 (HTTP/1.1 keep-alive)。

        逐次新建 TCP 连接是本程序最大的时间开销: 在本机上单次往返约 4.6 ms,
        其中 3.2 ms 都花在 ``socket.connect``。整局测试要发数百个动作,
        连接复用后同一动作的往返降到约 0.2 ms, 同样的现实预算能多做十几倍的
        检测/清除动作。
        """
        conn = self._conn
        if conn is not None and conn.sock is not None:
            try:
                conn.sock.settimeout(timeout)
            except OSError:
                pass
            return conn
        host, _, port = self.base_url.partition("://")[2].partition(":")
        conn = http.client.HTTPConnection(host or "127.0.0.1", int(port or 80), timeout=timeout)
        self._conn = conn
        return conn

    def _post_once(self, path: str, body: bytes) -> Tuple[int, Any]:
        """发送一次请求 (复用连接); 连接已失效时重建一次。"""
        conn = self._conn_for(self.request_timeout)
        try:
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json; charset=utf-8"})
            resp = conn.getresponse()
            raw = resp.read()
        except (http.client.HTTPException, OSError):
            # 半关连接/服务端主动关闭: 丢弃旧连接重建一次 (body 完全不变, 幂等)
            self._close_conn()
            conn = self._conn_for(self.request_timeout)
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json; charset=utf-8"})
            resp = conn.getresponse()
            raw = resp.read()
        return resp.status, raw

    def _transport_http(self, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retry + 1):
            try:
                status, raw = self._post_once(path, body)
            except Exception as exc:  # noqa: BLE001 - 连接失败/超时: 复用原请求重试
                last_error = exc
                self._close_conn()
                time.sleep(min(0.4 * attempt, 2.0))
                continue
            try:
                return status, json.loads(raw.decode("utf-8"))
            except Exception:  # noqa: BLE001 - 响应不是 JSON
                return status, {"accepted": False, "error": "http_%d" % status}
        raise RuntimeError("请求 %s 连续失败: %s" % (path, last_error))

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._transport or self._transport_http
        # 串行保证: 程序内部不允许出现两个动作并发 (附件2 5.3)
        with self._send_lock:
            if self._in_flight:
                self.serial_violations += 1
                print(
                    "[robot] 内部错误: 检测到并发发送动作 (in_flight=%r), 已串行等待" % self._in_flight,
                    file=sys.stderr,
                )
            self._in_flight = path
            try:
                return self._post_locked(fn, path, payload)
            finally:
                self._in_flight = None

    def _post_locked(self, fn, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(3):
            status, data = fn(path, payload)
            conflict = (
                status == 409
                or data.get("error") == "concurrent_request"
                or data.get("reject_code") == "concurrent_request"
            )
            if conflict and attempt < 2:
                # 模拟器判定存在并发动作: 该请求未被执行, 稍后复用同一 request_id 与原内容重试
                self.conflict_retries += 1
                time.sleep(0.25 * (attempt + 1))
                continue
            self._record(path, payload, status, data)
            if status == 200 and data.get("accepted") is True:
                self.virtual_time_s = float(data.get("virtual_time_s", self.virtual_time_s))
            return data
        raise RuntimeError("请求 %s 因并发冲突连续被拒绝" % path)

    def _record(self, path: str, payload: Dict[str, Any], status: int, data: Dict[str, Any]) -> None:
        """写一行机器狗自身的行为日志 (JSONL)。

        这里刻意只写文件不留在内存: 整局有上千个动作, 把每条的完整请求与响应都
        累积在 ``self.records`` 里只会白白占用内存, 而没有任何读取方。
        """
        if self._fh is None:
            return
        entry = {
            "wall": round(time.time(), 3),
            "path": path,
            "request": payload,
            "http_status": status,
            "response": data,
        }
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ---- 四条指令 ----
    def _base(self, request_id: str) -> Dict[str, Any]:
        return {"arena_id": "default", "robot_id": self.team_id, "request_id": request_id}

    def enter(self) -> Dict[str, Any]:
        data = self._post("/enter", self._base(self._next_id("enter")))
        if data.get("accepted") is True:
            self.remaining_real_s = float(data.get("remaining_real_duration_s", 1200))
            self.max_real_s = float(data.get("max_real_duration_s") or 1200.0)
            reported_virtual = data.get("max_virtual_duration_s")
            if reported_virtual:
                self.max_virtual_s = float(reported_virtual)
            self.virtual_time_s = float(data.get("virtual_time_s") or 0.0)
            self.entered_wall = self.clock()
        return data

    def exit(self) -> Dict[str, Any]:
        return self._post("/exit", self._base(self._next_id("exit")))

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        payload = self._base(self._next_id("measure"))
        payload["position"] = {"x": clamp_coord(x), "y": clamp_coord(y)}
        payload["channel"] = int(channel)
        return self._post("/measure", payload)

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        payload = self._base(self._next_id("clear"))
        payload["position"] = {"x": clamp_coord(x), "y": clamp_coord(y)}
        payload["channel"] = int(channel)
        return self._post("/clear", payload)

    # ---- 预算 ----
    def elapsed_real(self) -> float:
        # 必须走 ``self.clock()`` 而不是 ``time.time()`` —— 否则 ``__init__`` 注入的
        # 自推进时钟
        # 会被静默忽略, 1200 s 现实预算闸门在进程内永不生效, 那些工具的
        # "结果完全可复现" 也就不成立。
        return 0.0 if self.entered_wall is None else self.clock() - self.entered_wall

    def real_budget_left(self) -> Optional[float]:
        if self.remaining_real_s is None:
            return None
        return self.remaining_real_s - self.elapsed_real()

    def close(self) -> None:
        self._close_conn()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
