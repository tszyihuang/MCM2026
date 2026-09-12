"""比赛会话编排: 登录/时间校验、四个测试模块、5 秒倒计时、25 分钟窗口、日志队列与导出。

对应附件1 第 4 节「模拟器操作指南」与附件2 第 4.5 节「现实时间和虚拟时间」。

脱机(offline)模式下不联网, 仅做本地参赛队号校验与时间校验占位, 便于自测;
联网(online)模式会尝试向竞赛服务器进行登录与时间校验, 失败则拒绝启动新测试。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .core import (
    END_REASON_TEXT,
    LOG_ENCRYPT_MAX_BYTES,
    MAX_REAL_DURATION_S,
    WINDOW_DURATION_S,
    CaseGenerator,
    SimulatorEngine,
    export_log,
)
from .server import EventBus, SimulatorServer

PASSWORD_MIN, PASSWORD_MAX = 12, 64


class LoginError(Exception):
    pass


# --------------------------------------------------------------------------------------
# 登录 / 服务器时间校验
# --------------------------------------------------------------------------------------


def validate_password(password: str, team_id: str) -> Optional[str]:
    """按附件1 4.1 的规则校验密码。返回错误说明或 None。"""
    if not (PASSWORD_MIN <= len(password) <= PASSWORD_MAX):
        return "密码长度必须为 12 至 64 个字符"
    if not re.search(r"[A-Za-z]", password):
        return "密码必须包含字母"
    if not re.search(r"\d", password):
        return "密码必须包含数字"
    weak = {
        "123456789012",
        "password1234",
        "qwertyuiop12",
        "administrator",
        "abcd12345678",
    }
    if password.lower() in weak:
        return "密码过于简单"
    digits = re.sub(r"\D", "", team_id)
    if len(digits) >= 6 and len(password) >= 6:
        for i in range(len(digits) - 5):
            if digits[i : i + 6] in password:
                return "密码不能包含完整参赛队号中的连续 6 位数字"
    digits_pw = re.sub(r"\D", "", password)
    if len(digits_pw) >= 6 and len(team_id) >= 6:
        for i in range(len(digits_pw) - 5):
            if digits_pw[i : i + 6] in team_id:
                return "密码不能包含手机号中的连续 6 位数字"
    return None


@dataclass
class LoginState:
    team_id: str
    logged_in: bool = False
    mode: str = "offline"
    checked_at: float = 0.0
    server_time_offset_s: float = 0.0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "logged_in": self.logged_in,
            "mode": self.mode,
            "server_time_offset_s": round(self.server_time_offset_s, 3),
            "detail": self.detail,
        }


class LoginClient:
    """登录与服务器时间校验。offline 模式仅做本地校验。"""

    def __init__(
        self,
        team_id: str,
        mode: str = "offline",
        server_url: Optional[str] = None,
        state_path: Optional[str] = None,
    ) -> None:
        self.team_id = team_id
        self.mode = mode
        self.server_url = (server_url or "").rstrip("/")
        self.state_path = state_path
        self.state = LoginState(team_id=team_id, mode=mode)
        self._load()

    def _load(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("team_id") == self.team_id:
                self.state.logged_in = bool(data.get("logged_in", False))
                self.state.detail = "已从本地状态恢复登录"
        except Exception:
            pass

    def _save(self) -> None:
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(self.state.to_dict(), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def login(self, password: Optional[str] = None) -> LoginState:
        if not self.team_id or not self.team_id.strip():
            raise LoginError("参赛队号不能为空")
        if self.mode == "offline":
            self.state.logged_in = True
            self.state.detail = "离线模式: 仅本地校验参赛队号, 未联网注册/登录"
            self.state.checked_at = time.time()
            self._save()
            return self.state
        if not self.server_url:
            raise LoginError("联网模式必须配置竞赛服务器地址")
        payload = json.dumps(
            {"team_id": self.team_id, "password": password or ""}, ensure_ascii=False
        ).encode("utf-8")
        req = urllib.request.Request(
            self.server_url + "/login",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise LoginError("登录失败: %s" % exc)
        if not data.get("ok"):
            raise LoginError("服务器拒绝登录: %s" % data.get("message", "未知原因"))
        self.state.logged_in = True
        self.state.detail = "联网登录成功"
        self._save()
        return self.state

    def verify(self) -> LoginState:
        """每次测试前重新校验登录状态与服务器时间 (不超过 60 s 偏差)。"""
        if not self.state.logged_in:
            raise LoginError("尚未登录或登录状态失效")
        if self.mode == "offline":
            self.state.checked_at = time.time()
            self.state.detail = "离线模式: 已跳过服务器时间校验"
            return self.state
        if not self.server_url:
            raise LoginError("联网模式必须配置竞赛服务器地址")
        try:
            with urllib.request.urlopen(self.server_url + "/time", timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            server_ms = float(data["server_time_ms"])
        except Exception as exc:  # noqa: BLE001
            raise LoginError("服务器时间校验失败: %s" % exc)
        offset = (server_ms / 1000.0) - time.time()
        self.state.server_time_offset_s = offset
        if abs(offset) > 60.0:
            raise LoginError("本机时间与服务器时间相差超过 60 秒 (%.1f s)" % offset)
        self.state.checked_at = time.time()
        self.state.detail = "联网登录与时间校验通过"
        return self.state


# --------------------------------------------------------------------------------------
# 测试编排
# --------------------------------------------------------------------------------------

MODULES = {
    "q3_practice": {"problem": 3, "official": False, "title": "问题3 演练测试", "attempts": None},
    "q4_practice": {"problem": 4, "official": False, "title": "问题4 演练测试", "attempts": None},
    "q3_formal": {"problem": 3, "official": True, "title": "问题3 正式测试", "attempts": 3},
    "q4_formal": {"problem": 4, "official": True, "title": "问题4 正式测试", "attempts": 3},
}

DEADLINE_EPOCH = time.mktime(time.strptime("2026-09-13 17:30:00", "%Y-%m-%d %H:%M:%S"))


@dataclass
class ModuleState:
    key: str
    problem: int
    official: bool
    title: str
    attempts_used: int = 0
    max_attempts: Optional[int] = None
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        # 附件1 4.6: 正式测试在任何界面和完成提示中都不显示案例真值。
        # 因此正式测试模块的历史记录里不能带上"干扰源总数"等真值字段,
        # 控制接口 (/api/state) 与界面一同遵守该约束。
        records = []
        for r in self.records:
            item: Dict[str, Any] = {
                "case_code": r.get("case_code"),
                "cleared": r.get("stats", {}).get("cleared_count"),
                "avg_s": r.get("stats", {}).get("avg_clear_duration_s"),
                "runtime_s": r.get("stats", {}).get("program_runtime_s"),
                "end_reason": r.get("end_reason"),
            }
            if not self.official:
                item["total"] = r.get("stats", {}).get("source_total")
            records.append(item)
        return {
            "key": self.key,
            "title": self.title,
            "problem": self.problem,
            "official": self.official,
            "attempts_used": self.attempts_used,
            "max_attempts": self.max_attempts,
            "records": records,
        }


class SessionManager:
    """管理登录状态、四个测试模块、当前测试与日志队列。"""

    def __init__(
        self,
        team_id: str = "MCM2026",
        host: str = "127.0.0.1",
        port: int = 2026,
        data_dir: str = "data",
        login_mode: str = "offline",
        login_server: Optional[str] = None,
        countdown_s: float = 5.0,
        window_s: float = WINDOW_DURATION_S,
        max_real_s: float = MAX_REAL_DURATION_S,
        max_virtual_s: float = 360_000.0,
        enforce_deadline: bool = False,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.data_dir = os.path.abspath(data_dir)
        self.log_dir = os.path.join(self.data_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.bus = EventBus()
        self.engine = SimulatorEngine(
            team_id=team_id,
            countdown_s=countdown_s,
            window_s=window_s,
            max_real_s=max_real_s,
            max_virtual_s=max_virtual_s,
        )
        self.engine.on_run_end = self._on_run_end
        self.server = SimulatorServer(self.engine, host=host, port=port, bus=self.bus)
        self.login = LoginClient(
            team_id=team_id,
            mode=login_mode,
            server_url=login_server,
            state_path=os.path.join(self.data_dir, "login_state.json"),
        )
        self.modules: Dict[str, ModuleState] = {
            key: ModuleState(
                key=key,
                problem=meta["problem"],
                official=meta["official"],
                title=meta["title"],
                max_attempts=meta["attempts"],
            )
            for key, meta in MODULES.items()
        }
        self.current_module: Optional[str] = None
        self.current_record: Optional[Dict[str, Any]] = None
        self.upload_queue: List[Dict[str, Any]] = []
        self.enforce_deadline = enforce_deadline
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._reporter: Optional[Callable[[Dict[str, Any]], None]] = None

    # ---------------------------------------------------------------- 生命周期

    def start(self) -> int:
        port = self.server.start()
        self.bus.publish("started", {"base_url": self.server.base_url})
        return port

    def stop(self) -> None:
        self.server.stop()

    @property
    def base_url(self) -> str:
        return self.server.base_url

    def set_reporter(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        self._reporter = fn
        self.engine.on_run_end = self._on_run_end

    # ---------------------------------------------------------------- 启动一次测试

    def deadline_blocked(self) -> Optional[str]:
        if not self.enforce_deadline:
            return None
        if time.time() >= DEADLINE_EPOCH:
            return "北京时间 2026-09-13 17:30 之后不能启动新的演练测试或正式测试"
        return None

    def start_test(self, module_key: str, seed: Optional[int] = None) -> Dict[str, Any]:
        if module_key not in self.modules:
            raise ValueError("未知测试模块: %s" % module_key)
        module = self.modules[module_key]
        if self.engine.run is not None and self.engine.run.phase != "ended":
            raise RuntimeError("已有测试正在进行, 请先结束或中止当前测试")
        blocked = self.deadline_blocked()
        if blocked:
            raise RuntimeError(blocked)
        self.login.verify()
        if module.max_attempts is not None and module.attempts_used >= module.max_attempts:
            raise RuntimeError("%s 的正式测试机会已用完 (%d/%d)" % (module.title, module.attempts_used, module.max_attempts))
        cases = CaseGenerator(seed)
        code = "C%s-P%d-%s" % (time.strftime("%Y%m%d"), module.problem, "%08X" % cases.rng.getrandbits(32))
        run = self.engine.new_run(module.problem, seed=seed, code=code)
        self.engine.current_official = bool(module.official)
        self.current_module = module_key
        self.current_record = None
        if module.official:
            module.attempts_used += 1
        # 数据准备就绪后进入 5 秒倒计时
        self.engine.arm()
        self.bus.publish(
            "test_started",
            {
                "module": module_key,
                "title": module.title,
                "case_code": run.case.code,
                "official": module.official,
                "attempts_used": module.attempts_used,
            },
        )
        return {
            "module": module_key,
            "case_code": run.case.code,
            "problem": module.problem,
            "official": module.official,
            "countdown_s": self.engine.countdown_s,
        }

    def abort_test(self) -> Optional[Dict[str, Any]]:
        run = self.engine.run
        if run is None or run.phase == "ended":
            return None
        return self.engine.end_run("manual_abort")

    # ---------------------------------------------------------------- 结束与日志

    def _on_run_end(self, report: Dict[str, Any]) -> None:
        module_key = self.current_module
        module = self.modules.get(module_key) if module_key else None
        if module is not None:
            module.records.append(report)
        self.current_record = report
        export = export_log(report, self.log_dir, official=bool(module.official) if module else True)
        entry = {
            "case_code": report["case_code"],
            "module": module_key,
            "official": bool(module.official) if module else True,
            "paths": export,
            "queued_at": time.time(),
            "uploaded": (self.login.mode == "offline"),
            "over_limit": bool(export.get("over_upload_limit")),
        }
        with self._lock:
            self.history.append(entry)
            if not entry["uploaded"]:
                self.upload_queue.append(entry)
        self.bus.publish("test_ended", {"report": report, "export": export, "module": module_key})
        if self._reporter is not None:
            try:
                self._reporter(report)
            except Exception:
                pass

    def export_current(self) -> Optional[Dict[str, Any]]:
        if not self.current_record:
            return None
        module = self.modules.get(self.current_module or "")
        return export_log(
            self.current_record,
            self.log_dir,
            official=bool(module.official) if module else True,
        )

    def log_list(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = []
            for entry in reversed(self.history):
                stat = os.stat(entry["paths"]["readable"]) if os.path.exists(entry["paths"]["readable"]) else None
                items.append(
                    {
                        "case_code": entry["case_code"],
                        "module": entry["module"],
                        "official": entry["official"],
                        "path": entry["paths"].get("encrypted") or entry["paths"]["readable"],
                        "readable_path": entry["paths"]["readable"],
                        "size": stat.st_size if stat else 0,
                        "uploaded": entry["uploaded"],
                        "queued": not entry["uploaded"],
                    }
                )
            return items

    # ---------------------------------------------------------------- 状态

    def ui_state(self, log_limit: int = 200) -> Dict[str, Any]:
        self.engine.tick()
        state = {
            "login": self.login.state.to_dict(),
            "base_url": self.server.base_url,
            "modules": {k: m.to_dict() for k, m in self.modules.items()},
            "current_module": self.current_module,
            "status": self.engine.status(),
            "logs": self.log_list()[:50],
            "deadline_blocked": self.deadline_blocked(),
            "exchanges": self.bus.exchanges[-log_limit:],
            "upload_queue": len(self.upload_queue),
            "sim_limits": {
                "countdown_s": self.engine.countdown_s,
                "window_s": self.engine.window_s,
                "max_real_s": self.engine.max_real_s,
                "max_virtual_s": self.engine.max_virtual_s,
                "log_limit_bytes": LOG_ENCRYPT_MAX_BYTES,
            },
        }
        if self.current_record is not None and self.engine.run is not None and self.engine.run.phase == "ended":
            module = self.modules.get(self.current_module or "")
            official = bool(module.official) if module else True
            truth = self.current_record["truth"]
            # 附件1 4.6: 正式测试在任何界面和完成提示中都不显示案例真值。
            # 注意: 屏蔽必须在"服务端数据"层面完成 — 控制接口发出的 JSON 里不能含真值,
            # 不能只在界面上不渲染。
            if official:
                masked_truth: Dict[str, Any] = {"hidden": True, "reason": "正式测试不显示案例真值"}
                stats = mask_formal_stats(self.current_record["stats"])
            else:
                masked_truth = truth
                stats = self.current_record["stats"]
            state["last_report"] = {
                "case_code": self.current_record["case_code"],
                "problem": self.current_record["problem"],
                "official": official,
                "end_reason": self.current_record["end_reason"],
                "end_reason_text": END_REASON_TEXT.get(self.current_record["end_reason"], ""),
                "stats": stats,
                "truth": masked_truth,
                "uncleared": self.current_record["uncleared_sources"] if not official else [],
            }
        return state


# 正式测试中会泄露案例真值的统计字段 (必须以 "未知" 占位, 而不是真实数值)
FORMAL_TRUTH_FIELDS = ("source_total", "cleared_ratio")


def mask_formal_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """正式测试: 隐去所有可反推案例真值的统计字段。"""
    masked = dict(stats)
    for key in FORMAL_TRUTH_FIELDS:
        if key in masked:
            masked[key] = None
    return masked
