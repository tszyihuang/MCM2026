"""HTTP + JSON 通信层: 严格实现附件2 第5~9节的请求校验与状态码语义。

负责:
  * 路径 / 方法 / Content-Type / Content-Encoding / 请求体大小与格式校验;
  * 请求体 JSON 严格解析 (拒绝重复键、非法字面量、超过 16 层嵌套);
  * 把请求分派给 SimulatorEngine, 并把异常映射为附件2 表2 的 HTTP 状态;
  * 通过 on_exchange 回调把每条请求/响应交给界面与日志模块。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional

from .core import (
    BODY_LIMIT,
    MAX_JSON_DEPTH,
    ProtocolError,
    SimulatorEngine,
)

KNOWN_PATHS = ("/enter", "/measure", "/clear", "/exit")


# --------------------------------------------------------------------------------------
# 严格 JSON 解析
# --------------------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise ValueError("非法 JSON 字面量: %s" % name)


def _object_pairs_hook(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError("JSON 对象含有重复键: %s" % key)
        seen.add(key)
    return dict(pairs)


def parse_body(raw: bytes) -> Dict[str, Any]:
    """解析请求体。任何结构问题抛 ProtocolError(400)。"""
    if len(raw) > BODY_LIMIT:
        raise ProtocolError(413, "body_too_large", "请求体超过 65536 字节")
    if raw[:3] == b"\xef\xbb\xbf":
        raise ProtocolError(400, "invalid_body", "请求体不能带 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ProtocolError(400, "invalid_body", "请求体必须是 UTF-8 编码")
    try:
        payload = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_object_pairs_hook
        )
    except ValueError as exc:
        raise ProtocolError(400, "invalid_body", "请求体不是合法 JSON: %s" % exc)
    if not isinstance(payload, dict):
        raise ProtocolError(400, "invalid_body", "请求体必须是 JSON 对象")
    if _depth(payload) > MAX_JSON_DEPTH:
        raise ProtocolError(400, "json_too_deep", "JSON 嵌套不得超过 16 层")
    return payload


def _depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + (max((_depth(v) for v in value.values()), default=0))
    if isinstance(value, list):
        return 1 + (max((_depth(v) for v in value), default=0))
    return 0


# --------------------------------------------------------------------------------------
# HTTP 服务器
# --------------------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RadioInterferenceSimulator/1.0"
    sys_version = ""
    # 关掉 Nagle: 否则"发响应头 + 发响应体"两次写会被合并等待 ACK,
    # 在 Windows 上与对端延迟确认叠加, 单次往返要多花数毫秒。
    disable_nagle_algorithm = True

    # ---- 基础设施 ----
    @property
    def engine(self) -> SimulatorEngine:
        return self.server.engine

    @property
    def bus(self) -> "EventBus":
        return self.server.bus

    def _enter_request(self) -> None:
        httpd = self.server
        with httpd._stats_lock:
            httpd._active_requests += 1
            httpd.total_requests += 1
            if httpd._active_requests > httpd.max_concurrent_requests:
                httpd.max_concurrent_requests = httpd._active_requests

    def _leave_request(self) -> None:
        httpd = self.server
        with httpd._stats_lock:
            httpd._active_requests = max(0, httpd._active_requests - 1)

    def log_message(self, fmt, *args):  # noqa: A003 - 屏蔽默认 stderr 日志
        return

    def handle_one_request(self) -> None:  # noqa: D401
        """屏蔽 HTTP 解析层异常 (请求行过长/客户端中断等) 的堆栈输出。

        模拟器接口未开放或测试结束时连接可能被直接关闭, 此时不应向操作界面输出堆栈。
        """
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError, OSError):
            self.close_connection = True

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            # 一次性把状态行 + 头部 + 响应体写出去。原先是 send_response + 两个
            # send_header + wfile.write(body), 4 次 sendall 系统调用; 本机实测
            # 单次请求 382.7 µs -> 351.7 µs (-8.1%)。头部内容与 http.server 的
            # send_response/send_header 完全一致, 线上字节不变。
            head = (
                "HTTP/1.1 %d %s\r\n"
                "Server: %s\r\n"
                "Date: %s\r\n"
                "Content-Type: application/json; charset=utf-8\r\n"
                "Content-Length: %d\r\n"
                "Cache-Control: no-store\r\n"
                "\r\n"
                % (
                    status,
                    self.responses[status][0],
                    self.version_string(),
                    self.date_time_string(),
                    len(body),
                )
            ).encode("latin-1")
            self.wfile.write(head + body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def _base(self, accepted: bool) -> Dict[str, Any]:
        return {
            "accepted": accepted,
            "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": 0,
        }

    def _error_body(self, status: int, code: str, message: str) -> Dict[str, Any]:
        body = self._base(False)
        # 附件2 4.1: accepted=false 时 virtual_time_s 为 0, 不要把 0 当作当前虚拟时刻
        body["virtual_time_s"] = 0
        body["error"] = code
        body["message"] = message
        body["http_status"] = status
        return body

    # ---- 请求体读取 ----
    def _read_body(self) -> Optional[bytes]:
        te = (self.headers.get("Transfer-Encoding") or "").strip().lower()
        if te:
            if te != "chunked":
                self._send(400, self._error_body(400, "bad_transfer_encoding", "不支持的 Transfer-Encoding"))
                return None
            try:
                buf = bytearray()
                while True:
                    line = self.rfile.readline(1024)
                    if not line:
                        raise ValueError("分块传输提前结束")
                    size_token = line.split(b";")[0].strip()
                    size = int(size_token, 16)
                    if size == 0:
                        while True:
                            trailer = self.rfile.readline(1024)
                            if trailer in (b"\r\n", b"\n", b""):
                                break
                        break
                    buf += self.rfile.read(size)
                    self.rfile.read(2)  # CRLF
                    if len(buf) > BODY_LIMIT:
                        self._send(413, self._error_body(413, "body_too_large", "请求体超过 65536 字节"))
                        return None
                return bytes(buf)
            except Exception as exc:  # noqa: BLE001
                self._send(400, self._error_body(400, "invalid_body", "分块请求体解析失败: %s" % exc))
                return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send(411, self._error_body(411, "length_required", "缺少 Content-Length"))
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._send(400, self._error_body(400, "invalid_body", "Content-Length 非法"))
            return None
        if length < 0:
            self._send(400, self._error_body(400, "invalid_body", "Content-Length 非法"))
            return None
        if length > BODY_LIMIT:
            self._send(413, self._error_body(413, "body_too_large", "请求体超过 65536 字节"))
            return None
        return self.rfile.read(length) if length else b""

    # ---- HTTP 方法 ----
    def do_GET(self):  # noqa: N802
        self._method_not_post("GET")

    def do_HEAD(self):  # noqa: N802 - 已知路径非 POST 必须返回 405 (附件2 表2)
        self._method_not_post("HEAD")

    def do_POST(self):  # noqa: N802
        self._handle_action("POST")

    def do_PUT(self):  # noqa: N802
        self._method_not_post("PUT")

    def do_DELETE(self):  # noqa: N802
        self._method_not_post("DELETE")

    def do_PATCH(self):  # noqa: N802
        self._method_not_post("PATCH")

    def do_OPTIONS(self):  # noqa: N802
        self._method_not_post("OPTIONS")

    def _method_not_post(self, method: str) -> None:
        """已知路径的非 POST 方法返回 405; 未知路径返回 404。"""
        path = self.path
        status = 405 if path in KNOWN_PATHS else 404
        code = "method_not_allowed" if status == 405 else "not_found"
        message = "已知路径必须使用 POST" if status == 405 else "路径未知或路径不精确: %s" % path
        if method == "HEAD":
            # HEAD 不得返回响应体, 只返回头部
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send(status, self._error_body(status, code, message))
        self.bus.publish("http", {"method": method, "path": path, "status": status})

    def _handle_action(self, method: str) -> None:
        path = self.path
        if path not in KNOWN_PATHS:
            self._send(404, self._error_body(404, "not_found", "路径未知或路径不精确: %s" % path))
            self.bus.publish("http", {"method": method, "path": path, "status": 404})
            return
        if method != "POST":
            self._send(405, self._error_body(405, "method_not_allowed", "已知路径必须使用 POST"))
            self.bus.publish("http", {"method": method, "path": path, "status": 405})
            return
        content_type = self.headers.get("Content-Type") or ""
        if not self._content_type_ok(content_type):
            self._send(
                415,
                self._error_body(415, "unsupported_media_type", "Content-Type 必须是 application/json [; charset=utf-8]"),
            )
            self.bus.publish("http", {"method": method, "path": path, "status": 415})
            return
        encoding = (self.headers.get("Content-Encoding") or "identity").strip().lower()
        if encoding not in ("", "identity"):
            self._send(
                415,
                self._error_body(415, "unsupported_content_encoding", "Content-Encoding 只能省略或为 identity"),
            )
            self.bus.publish("http", {"method": method, "path": path, "status": 415})
            return
        raw = self._read_body()
        if raw is None:
            return
        t0 = time.perf_counter()
        try:
            payload = parse_body(raw)
        except ProtocolError as exc:
            self._send(exc.status, self._error_body(exc.status, exc.code, exc.message))
            self.bus.publish(
                "request",
                {
                    "path": path,
                    "body": _safe_preview(raw),
                    "status": exc.status,
                    "error": exc.code,
                    "message": exc.message,
                },
            )
            return
        action = path.lstrip("/")
        try:
            self._enter_request()
            try:
                response = self.engine.handle(action, payload)
                status = 200
            finally:
                self._leave_request()
            # 引擎业务层的并发冲突以 409 返回 (附件2 表2)
            if response.get("accepted") is False and response.get("error") == "concurrent_request":
                status = 409
        except ProtocolError as exc:
            status, response = exc.status, self._error_body(exc.status, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - 内部错误统一 500
            status, response = 500, self._error_body(500, "internal_error", "模拟器内部错误: %s" % exc)
        self._send(status, response)
        entry: Dict[str, Any] = {
            "path": path,
            "body": payload,
            "status": status,
            # perf_counter 而非 monotonic: 后者在 Windows 上分辨率是 15.6 ms,
            # 单次动作耗时会被量化成 0 或 16, 这条诊断字段就没有意义了。
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
        if status == 200:
            entry["response"] = response
        elif status == 409:
            entry["response"] = response
            entry["error"] = response.get("error") or "request_id_conflict"
            entry["message"] = response.get("message", "")
        else:
            entry["error"] = response.get("error")
            entry["message"] = response.get("message")
        self.bus.publish("request", entry)

    @staticmethod
    def _content_type_ok(value: str) -> bool:
        parts = [p.strip() for p in value.split(";") if p.strip()]
        if not parts:
            return False
        if parts[0].lower() != "application/json":
            return False
        for param in parts[1:]:
            if "=" not in param:
                return False
            key, _, val = param.partition("=")
            if key.strip().lower() != "charset":
                return False
            if val.strip().strip('"').lower() != "utf-8":
                return False
        return True


def _safe_preview(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "<binary %d bytes>" % len(raw)
    return text if len(text) <= 4096 else text[:4096] + "...(truncated)"


# --------------------------------------------------------------------------------------
# 事件总线 (界面 / 日志 / 遥测共用)
# --------------------------------------------------------------------------------------


class EventBus:
    """极简发布订阅, 线程安全。"""

    def __init__(self) -> None:
        self._subs: Dict[str, list] = {}
        self._lock = threading.Lock()
        self.max_exchanges = 2000
        self.exchanges: list = []

    def subscribe(self, topic: str, fn: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs.setdefault(topic, []).append(fn)

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if topic == "request":
            with self._lock:
                self.exchanges.append(payload)
                if len(self.exchanges) > self.max_exchanges:
                    del self.exchanges[: len(self.exchanges) - self.max_exchanges]
        with self._lock:
            subs = list(self._subs.get(topic, ()))
        for fn in subs:
            try:
                fn(payload)
            except Exception:
                pass


# --------------------------------------------------------------------------------------
# 服务器封装
# --------------------------------------------------------------------------------------


class _SimulatorHTTPServer(ThreadingHTTPServer):
    """带引擎引用与并发计数的 HTTP 服务器。

    并发计数原先是用 ``httpd.total_requests = 0`` 之类的方式临时挂到实例上的,
    于是每一处访问都要写 ``# type: ignore[attr-defined]``; 这里显式声明为类属性。
    """

    daemon_threads = True
    allow_reuse_address = True

    engine: SimulatorEngine
    bus: "EventBus"

    def __init__(self, address, handler, engine: SimulatorEngine, bus: "EventBus") -> None:
        super().__init__(address, handler)
        self.engine = engine
        self.bus = bus
        self._stats_lock = threading.Lock()
        self._active_requests = 0
        self.max_concurrent_requests = 0
        self.total_requests = 0


class SimulatorServer:
    """把 SimulatorEngine 暴露为 HTTP 接口。"""

    def __init__(
        self,
        engine: SimulatorEngine,
        host: str = "127.0.0.1",
        port: int = 2026,
        bus: Optional[EventBus] = None,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self.bus = bus or EventBus()
        self._httpd: Optional[_SimulatorHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def concurrency_stats(self) -> Dict[str, Any]:
        """并发观测: HTTP 层请求峰值 + 引擎内动作执行峰值 (后者必须恒为 1)。"""
        httpd = self._httpd
        engine_stats = self.engine.engine_concurrency_stats()
        if httpd is None:
            return {"active": 0, "max_concurrent": 0, "total_requests": 0, **engine_stats}
        with httpd._stats_lock:
            return {
                "active": httpd._active_requests,
                "max_concurrent": httpd.max_concurrent_requests,
                "total_requests": httpd.total_requests,
                "engine_active": engine_stats["active"],
                "engine_max_concurrent": engine_stats["max_concurrent"],
            }

    # ---- 生命周期 ----
    def start(self) -> int:
        httpd = _SimulatorHTTPServer((self.host, self.port), _Handler, self.engine, self.bus)
        self._httpd = httpd
        self.port = httpd.server_address[1]
        # poll_interval 默认 0.5 s: shutdown() 只能等一次 select 返回后才能生效,
        # 于是每次 stop() 都要空等最多 0.5 s。测试套件里每个用例都要起停一次
        # 模拟器 (110 次), 累计约 50 s 纯等待; 改成 0.05 s 后每次约 0.06 s。
        self._thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="simulator-http",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is None:
            return
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception as exc:  # noqa: BLE001 - 清理失败必须可见 (端口/线程泄漏)
            print("[模拟器] HTTP 服务停止异常: %r" % (exc,), file=sys.stderr)
        finally:
            self._httpd = None

    @property
    def base_url(self) -> str:
        return "http://%s:%d" % (self.host, self.port)

    def tick(self) -> None:
        self.engine.tick()
        self.engine.release_ended_run()
