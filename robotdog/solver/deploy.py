"""机器狗程序: 贝叶斯信念 + 确定性清除规划器, 走 HTTP+JSON.

通信层只用 ``/enter`` ``/measure`` ``/clear`` ``/exit`` 四条指令, 逐次等待响应,
复用 ``request_id`` 做幂等重试, 同时检查 HTTP 状态与 ``accepted`` 字段,
现实预算取自 ``/enter`` 的 ``remaining_real_duration_s``。

决策层:
  * 问题3 (全向源) —— ``sweeper``: 起点盲扫 → 就近清除已定位源 → 清除后原地补测
    → 无目标时按信息价值专程探测 → 收尾集合覆盖。**不需要任何模型权重**。

本地信念 (``belief.py``) 只由模拟器返回的读数驱动, 不读取任何真值 ——
这是本地评估与真机行为一致的前提。

用法::

    python -m robotdog.solver.deploy --team-id <参赛队号>
    python -m robotdog.solver.deploy --url http://127.0.0.1:2026 --problem 3
"""

from __future__ import annotations

import argparse
import functools
import http.client
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from robotdog.consts import CHANNEL_SWITCH_S, DETECT_DURATION_S, SPEED_MPS
from robotdog.solver.belief import Belief
from robotdog.solver.world import StepInfo, World


# --------------------------------------------------------------------------------------
# HTTP 客户端
# --------------------------------------------------------------------------------------
class SimClient:
    def __init__(self, base_url: str, team_id: str, log_path: Optional[str] = None,
                 timeout: float = 10.0, max_retry: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.team_id = team_id
        self.timeout = timeout
        self.max_retry = max_retry
        self.counter = 0
        self.virtual_time_s = 0.0
        self.remaining_real_s: Optional[float] = None
        self.entered_wall: Optional[float] = None
        self.conn: Optional[http.client.HTTPConnection] = None
        self._fh = None
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._fh = open(log_path, "a", encoding="utf-8")

    # ---- 连接 ----
    def _conn(self) -> http.client.HTTPConnection:
        if self.conn is not None and self.conn.sock is not None:
            return self.conn
        host, _, port = self.base_url.partition("://")[2].partition(":")
        self.conn = http.client.HTTPConnection(host or "127.0.0.1", int(port or 80),
                                               timeout=self.timeout)
        return self.conn

    def _close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = None

    def _post_once(self, path: str, body: bytes) -> Tuple[int, bytes]:
        conn = self._conn()
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": "application/json; charset=utf-8"})
            resp = conn.getresponse()
            return resp.status, resp.read()
        except (http.client.HTTPException, OSError):
            self._close()
            conn = self._conn()
            conn.request("POST", path, body=body,
                         headers={"Content-Type": "application/json; charset=utf-8"})
            resp = conn.getresponse()
            return resp.status, resp.read()

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retry + 1):
            try:
                status, raw = self._post_once(path, body)      # 重试复用同一 body/request_id
            except Exception as exc:  # noqa: BLE001
                last = exc
                self._close()
                time.sleep(min(0.3 * attempt, 1.5))
                continue
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:  # noqa: BLE001
                data = {"accepted": False, "error": "http_%d" % status}
            data["_http_status"] = status
            if status == 409:                                   # 并发冲突: 复用同一请求重试
                time.sleep(0.2 * attempt)
                continue
            if data.get("accepted") is True:
                self.virtual_time_s = float(data.get("virtual_time_s", self.virtual_time_s))
            self._record(path, payload, status, data)
            return data
        raise RuntimeError("请求 %s 连续失败: %s" % (path, last))

    def _record(self, path: str, payload: Dict[str, Any], status: int, data: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps({"wall": round(time.time(), 3), "path": path,
                                   "request": payload, "http_status": status,
                                   "response": data}, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ---- 四条指令 ----
    def _base(self, prefix: str) -> Dict[str, Any]:
        self.counter += 1
        return {"arena_id": "default", "robot_id": self.team_id,
                "request_id": "%s-%d" % (prefix, self.counter)}

    def enter(self) -> Dict[str, Any]:
        d = self.post("/enter", self._base("enter"))
        if d.get("accepted") is True:
            self.remaining_real_s = float(d.get("remaining_real_duration_s", 1200))
            self.virtual_time_s = float(d.get("virtual_time_s") or 0.0)
            self.entered_wall = time.time()
        return d

    def exit(self) -> Dict[str, Any]:
        return self.post("/exit", self._base("exit"))

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        p = self._base("measure")
        p["position"] = {"x": float(x), "y": float(y)}
        p["channel"] = int(channel)
        return self.post("/measure", p)

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        p = self._base("clear")
        p["position"] = {"x": float(x), "y": float(y)}
        p["channel"] = int(channel)
        return self.post("/clear", p)

    def real_left(self) -> float:
        if self.remaining_real_s is None or self.entered_wall is None:
            return float("inf")
        return self.remaining_real_s - (time.time() - self.entered_wall)

    def close(self) -> None:
        self._close()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --------------------------------------------------------------------------------------
# 远程世界: 与 world.World 同接口, 但所有读数来自模拟器
# --------------------------------------------------------------------------------------
class RemoteWorld(World):
    def __init__(self, client: SimClient, belief_cls=None) -> None:
        self.client = client
        self.seed = -1
        self.run_id = "remote"
        self.sources = []
        self.by_channel = {}
        self.n_sources = 0
        self.x = 0.0
        self.y = 0.0
        self.channel = 1
        self.virtual_t = 0.0
        # ``belief_cls`` 可被调用方覆盖 (测试用空实现), 默认是 ``Belief``。
        self.belief = belief_cls() if belief_cls is not None else Belief()
        self.cleared: List[int] = []
        self.actions = 0
        self.measures = 0
        self.clears = 0
        self.failed_clears = 0
        self.moved_m = 0.0

    # 只相信模拟器的响应
    def measure(self, x: float, y: float, channel: int) -> StepInfo:
        move_s = math.hypot(x - self.x, y - self.y) / SPEED_MPS
        resp = self.client.measure(x, y, channel)
        if resp.get("accepted") is not True:
            raise RuntimeError("measure 被拒绝: %s" % json.dumps(resp, ensure_ascii=False))
        dt = float(resp.get("virtual_time_s", self.virtual_t)) - self.virtual_t
        if dt <= 0.0:
            dt = move_s + (CHANNEL_SWITCH_S if channel != self.channel else 0.0) + DETECT_DURATION_S
        self.virtual_t = float(resp.get("virtual_time_s", self.virtual_t + dt))
        self.x, self.y, self.channel = float(x), float(y), int(channel)
        self.actions += 1
        self.measures += 1
        self.moved_m += move_s * SPEED_MPS
        result = resp.get("measure_result", "no_signal")
        if result == "direction":
            svd = float(resp["svd_deg"])
            self.belief.on_measure(x, y, channel, "direction", svd, self.virtual_t)
            return StepInfo("measure", "direction", svd, dt, channel, x, y)
        if result == "near":
            self.belief.on_measure(x, y, channel, "near", None, self.virtual_t)
            return StepInfo("measure", "near", None, dt, channel, x, y)
        self.belief.on_measure(x, y, channel, "no_signal", None, self.virtual_t)
        return StepInfo("measure", "no_signal", None, dt, channel, x, y)

    def clear(self, x: float, y: float, channel: int) -> StepInfo:
        move_s = math.hypot(x - self.x, y - self.y) / SPEED_MPS
        resp = self.client.clear(x, y, channel)
        if resp.get("accepted") is not True:
            raise RuntimeError("clear 被拒绝: %s" % json.dumps(resp, ensure_ascii=False))
        dt = float(resp.get("virtual_time_s", self.virtual_t)) - self.virtual_t
        self.virtual_t = float(resp.get("virtual_time_s", self.virtual_t + dt))
        self.x, self.y = float(x), float(y)
        self.actions += 1
        self.clears += 1
        self.moved_m += move_s * SPEED_MPS
        hit = resp.get("clear_result") == "success"
        if hit:
            self.cleared.append(channel)
            self.belief.on_clear(x, y, channel, True)
            return StepInfo("clear", "success", None, dt, channel, x, y)
        self.failed_clears += 1
        self.belief.on_clear(x, y, channel, False)
        return StepInfo("clear", "no_target_in_range", None, dt, channel, x, y)

    def truth(self):
        return []


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def run_planner_deploy(client: SimClient, reserve_s: float = 20.0, verbose: bool = True,
                       cfg=None) -> Dict[str, Any]:
    """跑确定性规划器, 不需要任何模型权重。

    ``knows_total=False``: **正式测试时机器狗看不到总源数** (真值屏蔽), 因此不能用
    ``all_cleared`` 作为结束条件, 只能靠 "没有可清目标 + 探测已无收益 + 残余期望质量低于
    阈值" 收手。
    """
    from robotdog.solver import sweeper as sw

    world = RemoteWorld(client)
    # 用 ``build_cfg()`` —— 评测工具与正式测试走**同一个**配置入口, 这样
    # "报告里的数字"与"真正上场的程序"不可能是两个策略。
    cfg = cfg or sw.build_cfg()
    res = sw.run_sweeper(world, cfg, trace=verbose, real_left=client.real_left,
                         reserve_s=reserve_s, knows_total=False)
    return {"cleared_count": len(world.cleared), "cleared_channels": sorted(world.cleared),
            "virtual_time_s": world.virtual_t, "measures": world.measures,
            "clears": world.clears, "failed_clears": world.failed_clears,
            "moved_m": world.moved_m, "steps": res.steps,
            "real_elapsed_s": (time.time() - client.entered_wall) if client.entered_wall else 0.0}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="机器狗程序 (问题3: sweeper)")
    ap.add_argument("--problem", type=int, default=3, choices=(3,),
                    help="3=全向源 (sweeper); 本工程只提供问题3 求解器")
    ap.add_argument("--url", default=os.environ.get("SIM_BASE_URL", "http://127.0.0.1:2026"))
    ap.add_argument("--team-id", default=os.environ.get("SIM_TEAM_ID", "MCM2026"))
    ap.add_argument("--log", default=os.environ.get("SIM_ROBOT_LOG", ""))
    ap.add_argument("--reserve", type=float, default=20.0, help="预留现实时间 (秒)")
    ap.add_argument("--wait-enter", type=float, default=0.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--cfg-module", default="",
                    help="可选: 提供一个 build_cfg()->SweepConfig 的模块名, 覆盖规划器默认参数")
    args = ap.parse_args(argv)

    if args.wait_enter > 0:
        time.sleep(args.wait_enter)
    client = SimClient(args.url, args.team_id, log_path=args.log or None)
    try:
        enter = client.enter()
        if enter.get("accepted") is not True:
            print("[solver] /enter 失败: %s" % json.dumps(enter, ensure_ascii=False),
                  file=sys.stderr)
            return 1
        if not args.quiet:
            print("[solver] 进入目标区域 | 可用现实时间 %.0f s" % (client.remaining_real_s or 0),
                  flush=True)
        fn = run_planner_deploy
        if args.cfg_module:
            import importlib
            cfg = importlib.import_module(args.cfg_module).build_cfg()
            fn = functools.partial(fn, cfg=cfg)
        summary = fn(client, reserve_s=args.reserve, verbose=not args.quiet)
        try:
            client.exit()
        except Exception:  # noqa: BLE001
            pass
    finally:
        client.close()
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
