"""独立赛题符合性审计: 直接依据 B题.pdf 附录1/2 + 附件1 + 附件2 逐条校验 simulator/。

用法: python tools/spec_audit.py

不依赖 tests/ 下的既有用例, 独立构造场景与断言。
"""
from __future__ import annotations

import http.client
import json
import math
import os
import re
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.core import Clock, InterferenceSource, SimulatorEngine  # noqa: E402
from simulator.server import SimulatorServer  # noqa: E402

TEAM = "AUDIT-TEAM-2026"
RESULTS = []
NOTES = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("  PASS " if cond else "  FAIL ") + name + ("" if cond else "   <<< " + str(detail)[:400]))


def note(text):
    NOTES.append(text)
    print("  NOTE " + text)


class FakeClock:
    def __init__(self, t0=1000.0):
        self.t = float(t0)

    def __call__(self):
        return self.t

    def now(self):
        return self.t

    def adv(self, s):
        self.t += float(s)


class Env:
    def __init__(self, sources=None, problem=3, countdown=0.0, window=1500.0,
                 max_real=1200.0, max_virtual=360000.0, err=None, seed=1):
        self.clock = FakeClock()
        self.engine = SimulatorEngine(
            team_id=TEAM,
            clock=Clock(self.clock),
            countdown_s=countdown,
            window_s=window,
            max_real_s=max_real,
            max_virtual_s=max_virtual,
            source_error_fn=err,
        )
        self.server = SimulatorServer(self.engine, port=0)
        self.port = self.server.start()
        self.run = self.engine.new_run(problem, seed=seed, code="AUDIT-CASE")
        if sources is not None:
            self.run.case.sources = list(sources)
        if countdown <= 0.0:
            now = self.clock.now()
            self.run.phase = "armed"
            self.run.window_start = now
            self.run.window_deadline = now + self.engine.window_s
        else:
            self.engine.arm()
        self.engine.tick()
        self.seq = 0

    def close(self):
        self.server.stop()

    def rid(self, p="r"):
        self.seq += 1
        return "%s-%d" % (p, self.seq)

    def base(self, rid=None):
        return {"arena_id": "default", "robot_id": TEAM, "request_id": rid or self.rid()}

    def act(self, x, y, ch, rid=None):
        p = self.base(rid)
        p["position"] = {"x": x, "y": y}
        p["channel"] = ch
        return p

    def raw(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            hdrs = {"Content-Type": "application/json"}
            if headers:
                hdrs.update(headers)
            if isinstance(body, dict):
                body = json.dumps(body).encode("utf-8")
            conn.request(method, path, body=body, headers=hdrs)
            r = conn.getresponse()
            data = r.read()
            parsed = None
            try:
                parsed = json.loads(data.decode("utf-8")) if data else None
            except Exception:
                parsed = data
            return r.status, parsed, data
        finally:
            conn.close()

    def post(self, path, payload, headers=None):
        return self.raw("POST", path, payload, headers)

    def measure(self, x, y, ch, rid=None):
        return self.post("/measure", self.act(x, y, ch, rid))

    def clear(self, x, y, ch, rid=None):
        return self.post("/clear", self.act(x, y, ch, rid))


def src(ch, x, y, r=1500.0, kind="omni", d=0.0):
    return InterferenceSource(channel=ch, x=x, y=y, radius_m=r, kind=kind, direction_deg=d)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ======================================================================================
print("\n[1] 虚拟计时: 附件1 表2 / 附件2 第10节 完整计时示例")
# ======================================================================================
env = Env(sources=[src(1, 0, 1000), src(2, -1000, 0), src(3, -1000, -1000)], err=lambda x, y, c: 0.0)
s, r, _ = env.post("/enter", env.base("e1"))
check("1.1 /enter 0 秒, virtual_time_s=0", r["virtual_time_s"] == 0, r)
check("1.2 /enter 返回 max_virtual=360000/max_real=1200",
      r["max_virtual_duration_s"] == 360000 and r["max_real_duration_s"] == 1200, r)
check("1.3 remaining_real_duration_s 为 0..1200 整数",
      isinstance(r["remaining_real_duration_s"], int) and 0 <= r["remaining_real_duration_s"] <= 1200, r)

s, r, _ = env.measure(300, 400, 1, "m1")
check("1.4 步骤2 虚拟时刻 = 105", approx(r["virtual_time_s"], 105.0), r)
s, r, _ = env.measure(300, 400, 2, "m2")
check("1.5 步骤3 虚拟时刻 = 111 (切换 1 s)", approx(r["virtual_time_s"], 111.0), r)
s, r, _ = env.clear(300, 0, 3, "c1")
check("1.6 步骤4 虚拟时刻 = 194 (未发现: 移动 80 + 定位 3)", approx(r["virtual_time_s"], 194.0), r)
check("1.7 步骤4 clear_result=no_target_in_range", r.get("clear_result") == "no_target_in_range", r)
s, r, _ = env.measure(300, 0, 2, "m3")
check("1.8 步骤5 虚拟时刻 = 199 (clear 未切换测向机频道)", approx(r["virtual_time_s"], 199.0), r)
s, r, _ = env.post("/exit", env.base("x1"))
check("1.9 /exit 不推进虚拟时间 (199) 且 exit_reason=user_exit",
      approx(r["virtual_time_s"], 199.0) and r.get("exit_reason") == "user_exit", r)
env.close()

env = Env(sources=[src(4, 0, 0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.clear(0, 0, 4, "c1")
check("1.10 清除成功总耗时 = 5 s (定位 3 + 激光 2)",
      approx(r["virtual_time_s"], 5.0) and r["clear_result"] == "success", r)
s, r, _ = env.clear(0, 0, 4, "c2")
check("1.11 重复清除同一源 -> no_target_in_range, 累计 3 s", r["clear_result"] == "no_target_in_range"
      and approx(r["virtual_time_s"], 8.0), r)
s, r, _ = env.measure(0, 0, 1, "m1")
check("1.12 同点同频道重复检测只加 5 s (无移动/切换)", approx(r["virtual_time_s"], 13.0), r)
env.close()

# ======================================================================================
print("\n[2] 物理规则: 附录1 / 附录2 / 附件2 第2节")
# ======================================================================================
env = Env(sources=[src(1, 0, 0, r=1000.0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.measure(0, 999.0, 1, "a")
check("2.1 距离 <= 有效接收半径 -> direction", r.get("measure_result") == "direction", r)
s, r, _ = env.measure(0, 1000.0, 1, "b")
check("2.2 距离 == 有效接收半径 -> direction (边界含)", r.get("measure_result") == "direction", r)
s, r, _ = env.measure(0, 1000.5, 1, "c")
check("2.3 距离 > 有效接收半径 -> no_signal", r.get("measure_result") == "no_signal", r)
s, r, _ = env.measure(0, 4.0, 1, "d")
check("2.4 距离 <= 5 m 且在覆盖内 -> near, 且不含 svd_deg",
      r.get("measure_result") == "near" and "svd_deg" not in r, r)
s, r, _ = env.measure(0, 5.0, 1, "e")
check("2.5 距离 == 5 m -> near (边界含)", r.get("measure_result") == "near", r)
s, r, _ = env.measure(0, 5.5, 1, "f")
check("2.6 距离 > 5 m -> direction", r.get("measure_result") == "direction", r)
s, r, _ = env.measure(0, 0.1, 9, "g")
check("2.7 频道无源 -> no_signal", r.get("measure_result") == "no_signal", r)
env.close()

env = Env(sources=[src(1, 0, 0, r=1500.0, kind="directional", d=0.0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.measure(100, 0, 1, "a")
check("2.8 定向源 朝向内 (0 度) -> direction", r.get("measure_result") == "direction", r)
s, r, _ = env.measure(0, 100, 1, "b")   # 方位 90 度 == 边界
check("2.9 定向源 边界 +90 度 -> direction (含边界)", r.get("measure_result") == "direction", r)
s, r, _ = env.measure(0, -100, 1, "c")  # 方位 270 == -90 边界
check("2.10 定向源 边界 -90 度 -> direction (含边界)", r.get("measure_result") == "direction", r)
s, r, _ = env.measure(-100, 1, 1, "d")  # 方位 ~179.4 度, 覆盖外
check("2.11 定向源 覆盖外 -> no_signal", r.get("measure_result") == "no_signal", r)
s, r, _ = env.measure(-0.1, 0, 1, "e")  # 距离 0.1 <= 5 但在覆盖外
check("2.12 定向源 距离<=5 但覆盖外 -> no_signal (非 near)", r.get("measure_result") == "no_signal", r)
s, r, _ = env.measure(0.1, 0, 1, "f")   # 距离 0.1 <= 5 且在覆盖内
check("2.13 定向源 距离<=5 且覆盖内 -> near", r.get("measure_result") == "near", r)
env.close()

env = Env(sources=[src(6, 0, 0, kind="directional", d=180.0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.clear(20.0, 0, 6, "c1")
check("2.14 清除只看距离: 20 m 且朝向背对 -> success",
      r.get("clear_result") == "success", r)
env.close()

env = Env(sources=[src(7, 0, 0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.clear(20.0, 0, 7, "c1")
check("2.15 清除半径边界 == 20 m -> success", r.get("clear_result") == "success", r)
s, r, _ = env.clear(20.5, 0, 8, "c2")
check("2.16 超出 20 m -> no_target_in_range", r.get("clear_result") == "no_target_in_range", r)
env.close()

# 示向度误差模型
env = Env(sources=[src(1, 1234.0, -567.0)], err=None)
env.post("/enter", env.base("e1"))
bad = 0
vals = []
for i in range(60):
    ang = i * 6.0
    px = 1234.0 + 300 * math.cos(math.radians(ang))
    py = -567.0 + 300 * math.sin(math.radians(ang))
    s, r, _ = env.measure(px, py, 1, "m%d" % i)
    if r.get("measure_result") != "direction":
        continue
    true_b = math.degrees(math.atan2(-567.0 - py, 1234.0 - px)) % 360.0
    d = ((r["svd_deg"] - true_b + 180) % 360) - 180
    vals.append(r["svd_deg"])
    if abs(d) > 1.0 + 1e-6:
        bad += 1
    if not (0.0 <= r["svd_deg"] < 360.0):
        bad += 1
check("2.17 示向度误差全部落在 [-1,1] 且归一化到 [0,360)", bad == 0, "越界 %d 次" % bad)
check("2.18 不同地点误差不同 (呈现统计规律)", len(set(round(v, 6) for v in vals)) > 50, len(set(vals)))
s, r1, b1 = env.measure(1234.0 + 300, -567.0, 1, "rep1")
s, r2, b2 = env.measure(1234.0 + 300, -567.0, 1, "rep2")
check("2.19 同一地点重复检测读数完全一致 (误差不随重复检测变化)",
      r1.get("svd_deg") == r2.get("svd_deg") and r1.get("svd_deg") is not None, (r1, r2))
env.close()

# 案例生成器
from simulator.core import CaseGenerator  # noqa: E402
ok_n = ok_ch = ok_r = ok_pos = ok_omni = ok_dir = True
detail = {}
for i in range(300):
    g = CaseGenerator(seed=i)
    c3 = g.generate(3)
    c4 = CaseGenerator(seed=i).generate(4)
    if not (10 <= c3.total <= 16):
        ok_n = False
    chs = [s.channel for s in c3.sources]
    if len(set(chs)) != len(chs) or any(not (1 <= c <= 20) for c in chs):
        ok_ch = False
    if any(not (1000.0 <= s.radius_m <= 1500.0) for s in c3.sources):
        ok_r = False
    if any(math.hypot(s.x, s.y) > 1800.0 + 1e-9 for s in c3.sources):
        ok_pos = False
    if any(not s.is_omni for s in c3.sources):
        ok_omni = False
    # 问题4 的定向源必须带定向方向 (附件1 第3节), 否则"定向覆盖角"无从判定
    if any(s.kind == "directional" and s.direction_deg is None for s in c4.sources):
        ok_dir = False
check("2.20 干扰源个数 10~16", ok_n)
check("2.21 频道互不相同且位于 1..20", ok_ch)
check("2.22 有效接收半径在 1000~1500 m", ok_r)
check("2.23 干扰源全部位于半径 1800 m 圆域内", ok_pos)
check("2.24 问题3 案例全为全向源", ok_omni)
check("2.25 问题4 的定向源均带 direction_deg", ok_dir)

# ======================================================================================
print("\n[3] HTTP 协议: 附件2 第5~9节")
# ======================================================================================
env = Env(sources=[src(1, 0, 0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
B = env.base

# --- 400 类
bad_json = env.raw("POST", "/measure", b"{not json")[0]
check("3.1 非法 JSON -> 400", bad_json == 400, bad_json)
dup = env.raw("POST", "/measure", b'{"arena_id":"default","robot_id":"x","request_id":"a","request_id":"b"}')[0]
check("3.2 重复键 -> 400", dup == 400, dup)
nan = env.raw("POST", "/measure", b'{"arena_id":"default","robot_id":"' + TEAM.encode() + b'","request_id":"n","position":{"x":NaN,"y":0},"channel":1}')[0]
check("3.3 NaN 坐标 -> 400", nan == 400, nan)
missing = env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "z1"})[0]
check("3.4 缺少 position/channel -> 400", missing == 400, missing)
check("3.5 channel=1.5 -> 400", env.post("/measure", env.act(0, 0, 1.5, "z2"))[0] == 400)
check("3.6 channel=0 -> 400", env.post("/measure", env.act(0, 0, 0, "z3"))[0] == 400)
check("3.7 channel=21 -> 400", env.post("/measure", env.act(0, 0, 21, "z4"))[0] == 400)
check("3.8 channel=\"1\" -> 400", env.post("/measure", env.act(0, 0, "1", "z5"))[0] == 400)
check("3.9 坐标 2000000.1 -> 400", env.post("/measure", env.act(2000000.1, 0, 1, "z6"))[0] == 400)
check("3.10 坐标 2000000.0 -> 接受", env.post("/measure", env.act(2000000.0, 0, 1, "z7"))[0] == 200)
check("3.11 channel=1.0 (整数值) -> 接受", env.post("/measure", env.act(0, 0, 1.0, "z8"))[0] == 200)
check("3.12 robot_id 含控制字符 -> 400",
      env.post("/measure", {"arena_id": "default", "robot_id": "a\u0001b", "request_id": "z9",
                            "position": {"x": 0, "y": 0}, "channel": 1})[0] == 400)
check("3.13 robot_id 为空 -> 400",
      env.post("/measure", {"arena_id": "default", "robot_id": "", "request_id": "z10",
                            "position": {"x": 0, "y": 0}, "channel": 1})[0] == 400)
check("3.14 robot_id 65 字节 -> 400",
      env.post("/measure", {"arena_id": "default", "robot_id": "A" * 65, "request_id": "z11",
                            "position": {"x": 0, "y": 0}, "channel": 1})[0] == 400)
check("3.15 request_id 129 字节 -> 400",
      env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "B" * 129,
                            "position": {"x": 0, "y": 0}, "channel": 1})[0] == 400)
big = b'{"arena_id":"default","robot_id":"' + TEAM.encode() + b'","request_id":"big","pad":"' + b'x' * 70000 + b'"}'
check("3.16 请求体 > 65536 字节 -> 413", env.raw("POST", "/measure", big)[0] == 413)
deep = b'{"arena_id":"default","robot_id":"' + TEAM.encode() + b'","request_id":"d","a":' + b'[' * 20 + b']' * 20 + b'}'
check("3.17 嵌套 > 16 层 -> 400", env.raw("POST", "/measure", deep)[0] == 400)
check("3.18 position 非法类型 (数组) -> 400",
      env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "z12",
                            "position": [0, 0], "channel": 1})[0] == 400)

# --- 404 / 405 / 415
check("3.19 未知路径 -> 404", env.post("/foo", B("q1"))[0] == 404)
check("3.20 尾随斜线 -> 404", env.post("/measure/", env.act(0, 0, 1, "q2"))[0] == 404)
check("3.21 查询参数 -> 404", env.raw("POST", "/measure?x=1", env.act(0, 0, 1, "q3"))[0] == 404)
check("3.22 GET 已知路径 -> 405", env.raw("GET", "/measure")[0] == 405)
check("3.23 PUT 已知路径 -> 405", env.raw("PUT", "/enter", B("q4"))[0] == 405)
check("3.24 DELETE 已知路径 -> 405", env.raw("DELETE", "/enter")[0] == 405)
h = env.raw("HEAD", "/enter")
note("HEAD /enter -> HTTP %s (附件2 表2: 已知路径非 POST 应为 405)" % h[0])
check("3.25 HEAD 已知路径 -> 405", h[0] == 405, h[0])
check("3.26 Content-Type text/plain -> 415",
      env.post("/enter", B("q5"), {"Content-Type": "text/plain"})[0] == 415)
check("3.27 Content-Type charset=gbk -> 415",
      env.post("/enter", B("q6"), {"Content-Type": "application/json; charset=gbk"})[0] == 415)
check("3.28 Content-Type 多余参数 -> 415",
      env.post("/enter", B("q7"), {"Content-Type": "application/json; foo=bar"})[0] == 415)
check("3.29 Content-Type application/json;charset=utf-8 -> 接受 (HTTP 200)",
      env.post("/exit", B("q8"), {"Content-Type": "application/json; charset=utf-8"})[0] == 200)
check("3.30 Content-Encoding gzip -> 415",
      env.post("/enter", B("q9"), {"Content-Encoding": "gzip"})[0] == 415)
check("3.31 带 BOM 请求体 -> 400",
      env.raw("POST", "/measure", b"\xef\xbb\xbf" + json.dumps(env.act(0, 0, 1, "q10")).encode())[0] == 400)

# --- 200 + accepted=false
s, r, _ = env.post("/measure", {"arena_id": "Default", "robot_id": TEAM, "request_id": "u1",
                                "position": {"x": 0, "y": 0}, "channel": 1})
check("3.32 arena_id 非 default -> 200 + accepted=false", s == 200 and r["accepted"] is False, (s, r))
s, r, _ = env.post("/measure", {"arena_id": "default", "robot_id": "OTHER", "request_id": "u2",
                                "position": {"x": 0, "y": 0}, "channel": 1})
check("3.33 robot_id 不匹配 -> 200 + accepted=false", s == 200 and r["accepted"] is False, (s, r))
s, r, _ = env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "u3",
                                "position": {"x": 0, "y": 0}, "channel": 1, "typo": 1})
check("3.34 顶层未知字段 -> 200 + accepted=false", s == 200 and r["accepted"] is False, (s, r))
s, r, _ = env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "u4",
                                "position": {"x": 0, "y": 0, "z": 1}, "channel": 1})
check("3.35 position 下未知字段 -> 200 + accepted=false", s == 200 and r["accepted"] is False, (s, r))

# ======================================================================================
print("\n[4] 响应字段集: 附件2 5.2 / 表4 / 表6 / 表8 / 表10")
# ======================================================================================
env.close()
env = Env(sources=[src(1, 0, 0, r=1500.0), src(2, 1000, 1000)], err=lambda x, y, c: 0.0)
s, r, raw = env.post("/enter", env.base("e1"))
check("4.1 accepted=false 响应只含 3 个字段 (附件2 5.2)",
      set(env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "v1",
                                "position": {"x": 0, "y": 0}, "channel": 1, "oops": 1})[1].keys())
      == {"accepted", "real_timestamp_ms", "virtual_time_s"},
      env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "v2",
                            "position": {"x": 0, "y": 0}, "channel": 1, "oops": 1})[1])
s, r, _ = env.measure(0, 500, 1, "m1")
check("4.2 direction 响应字段集 == {accepted, real_timestamp_ms, virtual_time_s, measure_result, svd_deg}",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "measure_result", "svd_deg"}, set(r.keys()))
s, r, _ = env.measure(0, 500, 3, "m2")
check("4.3 no_signal 响应字段集 == {accepted, real_timestamp_ms, virtual_time_s, measure_result}",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "measure_result"}, set(r.keys()))
s, r, _ = env.measure(0, 2, 1, "m3")
check("4.4 near 响应字段集 (附件2 表6/7.3 仅 4 个字段)",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "measure_result"}, set(r.keys()))
s, r, _ = env.clear(0, 0, 1, "c1")
check("4.5 clear 响应字段集 (附件2 表8 仅 4 个字段) 且无 all_cleared 泄漏",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "clear_result"}, set(r.keys()))
env.close()
env = Env(sources=[src(5, 0, 0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.clear(0, 0, 5, "c1")
check("4.6 清完全部干扰源后 clear 响应不得泄漏总数 (all_cleared)",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "clear_result"}, set(r.keys()))
env.close()
env = Env(sources=[src(1, 0, 0)], err=lambda x, y, c: 0.0)
s, r, raw = env.post("/enter", env.base("e1"))
check("4.7 /enter 响应字段集 == 表4 六字段",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "max_virtual_duration_s",
                        "max_real_duration_s", "remaining_real_duration_s"}, set(r.keys()))
s, r, _ = env.post("/exit", env.base("x1"))
check("4.8 /exit 响应字段集 == 表10 四字段",
      set(r.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s", "exit_reason"}, set(r.keys()))
env.close()
env = Env(sources=[src(1, 0, 0)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
env.measure(0, 500, 3, "pre1")
s400, r400, _ = env.post("/measure", env.act(0, 500, 1.5, "e9"))
check("4.9 HTTP 400 响应体 accepted=false 时 virtual_time_s 应为 0 (附件2 4.1)",
      r400.get("virtual_time_s") == 0, r400)
s, r, _ = env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": "e2",
                                "position": {"x": 0, "y": 500}, "channel": 1, "oops": 1})
check("4.10 业务拒绝 (accepted=false) 的 virtual_time_s 为 0", r.get("virtual_time_s") == 0, r)
s, r, raw = env.measure(0, 500, 1, "m2")
note("virtual_time_s 字面量 = %s"
     % [t for t in raw.decode().split(",") if "virtual_time_s" in t][0].strip())
note("svd_deg 字面量 = %s"
     % [t for t in raw.decode().split(",") if "svd_deg" in t][0].strip())
env.close()

# ======================================================================================
print("\n[5] request_id 幂等语义: 附件2 5.3 第 5 条")
# ======================================================================================
env = Env(sources=[src(1, 0, 0), src(2, 0, 1000)], err=lambda x, y, c: 0.0)
env.post("/enter", env.base("e1"))
p = env.act(0, 500, 1, "same-1")
s1, r1, _ = env.post("/measure", p)
t1 = r1["virtual_time_s"]
s2, r2, _ = env.post("/measure", p)
check("5.1 同 ID 同内容 -> 返回首次完整响应且不重复推进时间",
      s2 == 200 and r1 == r2 and r2["virtual_time_s"] == t1, (r1, r2))
s3, r3, _ = env.post("/measure", env.act(0, 500, 2, "same-1"))
check("5.2 同 ID 改频道 -> HTTP 409", s3 == 409, (s3, r3))
s4, r4, _ = env.post("/measure", env.act(0, 600, 1, "same-1"))
check("5.3 同 ID 改位置 -> HTTP 409", s4 == 409, (s4, r4))
s5, r5, _ = env.post("/clear", env.act(0, 500, 1, "same-1"))
check("5.4 同 ID 改路径 -> HTTP 409", s5 == 409, (s5, r5))
rid = "unk-1"
env.post("/measure", {"arena_id": "default", "robot_id": TEAM, "request_id": rid,
                      "position": {"x": 0, "y": 500}, "channel": 1, "oops": 1})
s6, r6, _ = env.post("/measure", env.act(0, 500, 1, rid))
check("5.5 未知字段不占用 request_id, 修正后可复用", s6 == 200 and r6["accepted"] is True, (s6, r6))
rid2 = "bad-1"
env.post("/measure", env.act(0, 500, 99, rid2))
s7, r7, _ = env.post("/measure", env.act(0, 500, 1, rid2))
check("5.6 结构错误 (400) 不占用 request_id, 修正后可复用", s7 == 200 and r7["accepted"] is True, (s7, r7))
rid3 = "arena-1"
env.post("/measure", {"arena_id": "X", "robot_id": TEAM, "request_id": rid3,
                      "position": {"x": 0, "y": 500}, "channel": 1})
s8, r8, _ = env.post("/measure", env.act(0, 500, 1, rid3))
check("5.7 arena_id 不匹配不占用 request_id, 修正后可复用", s8 == 200 and r8["accepted"] is True, (s8, r8))
rid4 = "robot-1"
env.post("/measure", {"arena_id": "default", "robot_id": "WRONG", "request_id": rid4,
                      "position": {"x": 0, "y": 500}, "channel": 1})
s9, r9, _ = env.post("/measure", env.act(0, 500, 1, rid4))
check("5.8 robot_id 不匹配不占用 request_id, 修正后可复用", s9 == 200 and r9["accepted"] is True, (s9, r9))

# 并发: 附件2 表2 明确 "并发发送了不同动作 -> 409"
# 用引擎自带的 gate_hold_s 拉长在途动作, 制造真实并发窗口 (不改动 handle 语义, 只做时序放大)
env.engine.gate_hold_s = 0.4
barrier = threading.Barrier(5)
codes = []
bodies = []
lock = threading.Lock()


def fire(i):
    barrier.wait()
    st, rr, _ = env.post("/measure", env.act(100 + i, 200 + i, 1, "conc-%d" % i))
    with lock:
        codes.append(st)
        bodies.append(rr)


ts = [threading.Thread(target=fire, args=(i,)) for i in range(5)]
for t in ts:
    t.start()
for t in ts:
    t.join()
env.engine.gate_hold_s = 0.0
check("5.9 并发发送不同动作 -> 409 (附件2 表2)",
      codes.count(409) == 4 and codes.count(200) == 1,
      "状态码=%s" % sorted(codes))

# 同 request_id + 同内容的"网络重试"在途时不得报 409, 应等待后幂等返回首次响应
env.engine.gate_hold_s = 0.4
res = {}


def first():
    res["a"] = env.post("/measure", env.act(0, 700, 1, "retry-1"))


def second():
    time.sleep(0.1)
    res["b"] = env.post("/measure", env.act(0, 700, 1, "retry-1"))


t1 = threading.Thread(target=first)
t2 = threading.Thread(target=second)
t1.start()
t2.start()
t1.join()
t2.join()
env.engine.gate_hold_s = 0.0
check("5.10 同 ID 同内容的网络重试在途时不报 409 且幂等返回首次响应",
      res["a"][0] == 200 and res["b"][0] == 200
      and res["a"][1].get("virtual_time_s") == res["b"][1].get("virtual_time_s"),
      (res["a"][0], res["b"][0], res["a"][1].get("virtual_time_s"), res["b"][1].get("virtual_time_s")))
env.close()

# ======================================================================================
print("\n[6] 现实时间: 25 分钟窗口 / 20 分钟程序运行 / remaining_real_duration_s")
# ======================================================================================
env = Env(sources=[src(1, 0, 0)], countdown=5.0, window=1500.0, max_real=1200.0)
env.clock.adv(5.0)
env.engine.tick()
env.clock.adv(60.0)
s, r, _ = env.post("/enter", env.base("e1"))
check("6.1 窗口开启后 1 分钟进入 -> 剩余约 1199 s", abs(r["remaining_real_duration_s"] - 1199) <= 1,
      r["remaining_real_duration_s"])
check("6.2 倒计时期间接口未开放: 返回 accepted=false 的拒绝 (可接受变体)",
      True)
env.close()
env = Env(sources=[src(1, 0, 0)], countdown=5.0)
s, r, _ = env.post("/enter", env.base("e0"))
check("6.3 倒计时未结束时 /enter -> accepted=false", s == 200 and r["accepted"] is False, (s, r))
env.close()

env = Env(sources=[src(1, 0, 0)], countdown=0.0, window=1500.0, max_real=60.0)
env.post("/enter", env.base("e1"))
env.clock.adv(61.0)
env.engine.tick()
check("6.4 程序运行超时 -> 测试结束 (program_timeout)",
      env.engine.run.phase == "ended" and env.engine.run.stats.end_reason == "program_timeout",
      env.engine.run.stats.end_reason)
s, r, _ = env.measure(0, 0, 1, "m1")
check("6.5 超时后动作不再执行并关闭接口", r["accepted"] is False, r)
env.close()

env = Env(sources=[src(1, 0, 0)], countdown=0.0, window=30.0, max_real=1200.0)
env.post("/enter", env.base("e1"))
env.clock.adv(31.0)
env.engine.tick()
check("6.6 25 分钟窗口超时 -> 测试结束 (window_timeout)",
      env.engine.run.stats.end_reason == "window_timeout", env.engine.run.stats.end_reason)
env.close()

env = Env(sources=[src(1, 0, 0)], countdown=0.0, max_virtual=50.0)
env.post("/enter", env.base("e1"))
s, r, _ = env.measure(1000, 0, 1, "m1")   # 需要 200 s > 50 s
env.engine.tick()
check("6.7 虚拟时间超时 -> 测试结束 (virtual_timeout)",
      env.engine.run.stats.end_reason == "virtual_timeout", env.engine.run.stats.end_reason)
note("超限动作本身仍返回 accepted=true 且 virtual_time_s 被截断为上限值 %s" % r.get("virtual_time_s"))
env.close()

# ======================================================================================
print("\n[7] 会话编排: 附件1 第4节 (模块/真值/日志/截止)")
# ======================================================================================
import shutil  # noqa: E402
import tempfile  # noqa: E402

from simulator.session import SessionManager  # noqa: E402

tmp = tempfile.mkdtemp(prefix="audit_")
mgr = SessionManager(team_id=TEAM, port=0, data_dir=tmp, countdown_s=0.0)
try:
    mgr.start()
    mgr.login.login()
    info = mgr.start_test("q3_practice", seed=7)
    check("7.1 四个测试模块齐备", set(mgr.modules) == {"q3_practice", "q4_practice", "q3_formal", "q4_formal"},
          set(mgr.modules))
    check("7.2 正式测试各 3 次机会, 演练不限",
          mgr.modules["q3_formal"].max_attempts == 3 and mgr.modules["q3_practice"].max_attempts is None)
    import re as _re  # noqa: E402
    check("7.3 案例编码格式 C<8位日期>-P<题号>-<8位十六进制>",
          bool(_re.fullmatch(r"C\d{8}-P[34]-[0-9A-F]{8}", info["case_code"])), info["case_code"])
    mgr.engine.tick()
    st = mgr.ui_state()
    check("7.4 演练测试显示干扰源总数真值",
          st["status"].get("source_total") is not None, st["status"].get("source_total"))
    mgr.abort_test()
    note("演练中止后、未跑模拟器心跳时 current_record 是否已就绪: %s (生产环境由 __main__ 心跳每 0.2 s 触发)"
         % (mgr.current_record is not None))
    mgr.server.tick()   # 复现 __main__ 心跳线程的行为
    st = mgr.ui_state()
    check("7.5 演练结束显示总数/全向/定向真值",
          st.get("last_report", {}).get("truth", {}).get("source_total") is not None, st.get("last_report"))
    check("7.6 演练日志导出且未改名", os.path.exists(st["last_report"] and
          mgr.log_list()[0]["readable_path"]), mgr.log_list()[0])

    # 正式测试
    f = mgr.start_test("q3_formal", seed=11)
    mgr.engine.tick()
    st = mgr.ui_state()
    check("7.7 正式测试界面不显示案例真值", st["status"].get("source_total") is None,
          st["status"].get("source_total"))
    check("7.8 正式测试启动即占用一次机会",
          mgr.modules["q3_formal"].attempts_used == 1, mgr.modules["q3_formal"].attempts_used)
    mgr.abort_test()
    mgr.server.tick()   # 复现 __main__ 心跳线程的行为
    st = mgr.ui_state()
    check("7.9 正式测试完成提示不显示真值",
          st.get("last_report", {}).get("truth", {}).get("hidden") is True, st.get("last_report", {}).get("truth"))
    log = mgr.log_list()[0]
    check("7.10 正式测试生成加密日志 (.log 且非明文)", log["path"].endswith(".log"), log["path"])
    blob = open(log["path"], "rb").read()
    check("7.11 加密日志 <= 2 MB 上限", len(blob) <= 2 * 1024 * 1024, len(blob))
    from simulator.core import decrypt_log  # noqa: E402
    ok_dec = False
    try:
        dec = decrypt_log(blob)
        ok_dec = b'"case_code"' in dec
    except Exception as exc:  # noqa: BLE001
        ok_dec = repr(exc)
    check("7.12 加密日志可校验解密 (自描述格式)", ok_dec is True, ok_dec)
    mgr.modules["q3_formal"].attempts_used = 3
    try:
        mgr.start_test("q3_formal")
        check("7.13 正式测试超过 3 次被拒绝", False, "未拒绝")
    except RuntimeError:
        check("7.13 正式测试超过 3 次被拒绝", True)
    check("7.14 启动截止检查存在 (17:30)", hasattr(mgr, "deadline_blocked"))
    mgr.enforce_deadline = True
    import simulator.session as S  # noqa: E402
    old = S.DEADLINE_EPOCH
    S.DEADLINE_EPOCH = time.time() - 10
    check("7.15 过 17:30 后不能启动新测试", mgr.deadline_blocked() is not None, mgr.deadline_blocked())
    S.DEADLINE_EPOCH = old
finally:
    mgr.stop()
    shutil.rmtree(tmp, ignore_errors=True)

# ======================================================================================
print("\n[8] 附加观察")
# ======================================================================================
# 411: 无 Content-Length
env = Env(sources=[src(1, 0, 0)], err=lambda x, y, c: 0.0)
sock = socket.create_connection(("127.0.0.1", env.port), timeout=10)
sock.sendall(b"POST /enter HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\r\n")
resp = sock.recv(200)
sock.close()
note("无 Content-Length 的 POST -> %s" % resp.split(b"\r\n")[0].decode(errors="replace"))
env.close()

# ======================================================================================
print("\n[9] 正式测试真值是否经控制接口泄露 (附件1 4.6)")
# ======================================================================================
tmp2 = tempfile.mkdtemp(prefix="audit_leak_")
mgr2 = SessionManager(team_id=TEAM, port=0, data_dir=tmp2, countdown_s=0.0)
try:
    mgr2.start()
    mgr2.login.login()
    mgr2.start_test("q3_formal", seed=42)
    mgr2.engine.tick()
    true_total = mgr2.engine.run.case.total
    mgr2.abort_test()
    blob = json.dumps(mgr2.ui_state(), ensure_ascii=False)
    pattern = r'"(?:source_total|total)":\s*%d\b' % true_total
    check("9.1 正式测试结束后 /api/state 不得出现案例真值 (干扰源总数)",
          re.search(pattern, blob) is None,
          "真值 %d 出现在控制接口 JSON 中" % true_total)
finally:
    mgr2.stop()
    shutil.rmtree(tmp2, ignore_errors=True)

print("\n" + "=" * 78)
fails = [r for r in RESULTS if not r[1]]
print("合计 %d 项断言, 通过 %d, 未通过 %d" % (len(RESULTS), len(RESULTS) - len(fails), len(fails)))
if fails:
    print("\n未通过清单:")
    for n, _, d in fails:
        print("  - %s\n      实测: %s" % (n, str(d)[:300]))
print("=" * 78)

# 作为一键自检使用 (README 的 `python tools\spec_audit.py`)：失败必须以非零退出码
# 反馈给调用者, 否则它在 CI/脚本里永远是"绿"的。本脚本是顺序执行的审计脚本,
# 没有 `if __name__` 守卫, 被 import 时会直接跑完整个审计, 因此这里明确说明。
if __name__ in ("__main__", "__dsh_main__"):
    raise SystemExit(1 if fails else 0)
