"""一键冒烟测试: 启动模拟器 -> 校验协议与物理规则 -> 真实进程跑通演练测试.

用法 (在项目根目录执行):
    python tools/smoke_test.py            # 快速冒烟 (约 30 秒)
    python tools/smoke_test.py --full     # 附加完整单元测试套件

退出码 0 表示全部通过。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.core import (  # noqa: E402
    CHANNEL_SWITCH_S,
    CLEAR_RADIUS_M,
    DETECT_DURATION_S,
    decrypt_log,
)
from simulator.session import SessionManager  # noqa: E402

TEAM = "SMOKE2026"
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  <- " + detail) if detail and not ok else ""))
    return ok


def post(base: str, path: str, payload: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw}


def base_payload(rid: str) -> dict:
    return {"arena_id": "default", "robot_id": TEAM, "request_id": rid}


def raw_request(base: str, method: str, path: str):
    """原始 HTTP 请求, 返回 (状态码, 响应体, 头部)。"""
    import http.client

    host, _, port = base.replace("http://", "").partition(":")
    conn = http.client.HTTPConnection(host, int(port or 80), timeout=10)
    try:
        conn.request(method, path, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, resp.read(), dict(resp.getheaders())
    finally:
        conn.close()


def action_payload(rid: str, x: float, y: float, ch: int) -> dict:
    p = base_payload(rid)
    p["position"] = {"x": x, "y": y}
    p["channel"] = ch
    return p


def wait_interface_open(manager: SessionManager, timeout: float = 5.0) -> bool:
    """等到倒计时结束、机器狗接口开放。

    原先每处都写 ``time.sleep(1.3)``: 倒计时只有 1.0 s, 固定的 1.3 s 既慢又
    在负载高时可能不够。改成轮询真实状态, 通常十几毫秒就返回。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if manager.ui_state()["status"]["interface_open"]:
            return True
        time.sleep(0.01)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="附加运行完整单元测试套件")
    ap.add_argument("--keep-data", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="smoke_")
    print("=" * 84)
    print("无线电干扰源环境模拟器 · 冒烟测试")
    print("=" * 84)

    manager = SessionManager(team_id=TEAM, port=0, data_dir=tmp, countdown_s=1.0)
    manager.login.login()
    port = manager.start()
    base = manager.base_url
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                manager.server.tick()
            except Exception:
                pass
            stop.wait(0.05)

    threading.Thread(target=heartbeat, daemon=True).start()

    # ---------------- 1. 协议层 ----------------
    print("\n[1] 通信协议一致性 (附件2 第5节)")
    st, body = post(base, "/enter", base_payload("smoke-early"))
    check("倒计时期间接口未开放", st == 200 and body.get("accepted") is False and body.get("virtual_time_s") == 0, str(body))
    st, _ = post(base, "/nope", base_payload("smoke-404"))
    check("未知路径返回 404", st == 404)
    req = urllib.request.Request(base + "/enter", headers={"Content-Type": "text/plain"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        check("非法 Content-Type 返回 415", False, "未返回错误")
    except urllib.error.HTTPError as exc:
        check("非法 Content-Type 返回 415", exc.code == 415)
    st, body = post(base, "/measure", action_payload("smoke-x", 0, 0, 21))
    check("频道越界返回 400", st == 400)
    st, body = post(base, "/measure", action_payload("smoke-x", 2000001, 0, 1))
    check("坐标越界返回 400", st == 400)

    # ---------------- 2. 计时与物理规则 ----------------
    print("\n[2] 虚拟计时与物理规则 (附件1 表2 / 附件2 第4节)")
    info = manager.start_test("q3_practice", seed=1001)
    case_code = info["case_code"]
    wait_interface_open(manager)
    st, enter = post(base, "/enter", base_payload("smoke-1"))
    check("/enter 成功且返回现实预算", st == 200 and enter.get("accepted") is True and "remaining_real_duration_s" in enter)
    check("/enter 不推进虚拟时钟", enter.get("virtual_time_s") == 0)
    budget = enter.get("remaining_real_duration_s")
    st, m1 = post(base, "/measure", action_payload("smoke-2", 300, 400, 1))
    expect1 = 500 / 5.0 + DETECT_DURATION_S
    check("移动 500 m + 检测 5 s = 105 s", abs(m1.get("virtual_time_s", -1) - expect1) < 1e-6, str(m1))
    st, m2 = post(base, "/measure", action_payload("smoke-3", 300, 400, 2))
    check("频道切换 1 s", abs(m2.get("virtual_time_s", -1) - (expect1 + CHANNEL_SWITCH_S + DETECT_DURATION_S)) < 1e-6, str(m2))
    st, c1 = post(base, "/clear", action_payload("smoke-4", 300, 400, 3))
    check("/clear 不切换频道", abs(c1.get("virtual_time_s", -1) - (m2["virtual_time_s"] + 3.0)) < 1e-6, str(c1))
    st, dup = post(base, "/measure", action_payload("smoke-2", 300, 400, 1))
    check("request_id 幂等重放不推进时间", dup.get("virtual_time_s") == m1.get("virtual_time_s"), str(dup))
    st, conflict = post(base, "/measure", action_payload("smoke-2", 1, 1, 1))
    check("同一 request_id 不同内容返回 409", st == 409)
    check("每次测试仅允许一次 /enter", post(base, "/enter", base_payload("smoke-5"))[1].get("accepted") is False)

    # 与真值对照近距阈值 / 有效半径
    case = manager.engine.run.case
    truth_ok = True
    detail = ""
    for src in case.sources[:3]:
        st, near = post(base, "/measure", action_payload("smoke-near-%d" % src.channel, src.x + 1.0, src.y, src.channel))
        if near.get("measure_result") != "near" and src.in_coverage(src.x + 1.0, src.y):
            truth_ok, detail = False, "频道 %d 在 1 m 处应为 near, 实际 %s" % (src.channel, near.get("measure_result"))
        far_x, far_y = src.x + src.radius_m + 5.0, src.y
        st, far = post(base, "/measure", action_payload("smoke-far-%d" % src.channel, far_x, far_y, src.channel))
        if far.get("measure_result") != "no_signal":
            truth_ok, detail = False, "频道 %d 在有效半径外应为 no_signal" % src.channel
    check("近距阈值与有效接收半径判定正确", truth_ok, detail)

    # 清除半径边界
    src0 = case.sources[0]
    st, ok20 = post(base, "/clear", action_payload("smoke-c20", src0.x + CLEAR_RADIUS_M, src0.y, src0.channel))
    hit20 = ok20.get("clear_result") == "success" if st == 200 else False
    check("清除半径 20 m 边界可清除", hit20, str(ok20))
    if not hit20:
        st, ok25 = post(base, "/clear", action_payload("smoke-c25", src0.x + CLEAR_RADIUS_M + 0.5, src0.y, src0.channel))
        check("超出 20 m 不可清除", ok25.get("clear_result") == "no_target_in_range", str(ok25))

    st, ex = post(base, "/exit", base_payload("smoke-9"))
    check("/exit 返回 user_exit", ex.get("exit_reason") == "user_exit", str(ex))
    time.sleep(0.4)
    report = manager.current_record
    check("测试结束后生成报告与行为日志", report is not None and os.path.exists(
        os.path.join(manager.log_dir, "practice_p3_%s.json" % case_code)))

    # ---------------- 2.5 并发与 HTTP 错误体 ----------------
    print("\n[2.5] 并发串行化与错误体语义 (附件2 表2 / 4.1)")
    manager.start_test("q3_practice", seed=1001)
    wait_interface_open(manager)
    post(base, "/enter", base_payload("smoke-c0"))
    st, m = post(base, "/measure", action_payload("smoke-c1", 100, 0, 1))
    check("接口开放后 /measure 正常接受",
          st == 200 and m.get("accepted") is True, str((st, m)))
    st, err = post(base, "/measure", action_payload("smoke-c2", 0, 0, 99))
    check("400 错误体 accepted=false 且 virtual_time_s=0",
          st == 400 and err.get("accepted") is False and err.get("virtual_time_s") == 0, str(err))
    st, raw, _ = raw_request(base, "HEAD", "/enter")
    check("HEAD 已知路径返回 405 且无响应体", st == 405 and raw in (b"", None), "status=%s" % st)
    # 并发: 延迟临界区, 5 个不同动作同时发送
    manager.engine.gate_hold_s = 1.2
    import concurrent.futures as cf

    def fire(i):
        return post(base, "/measure", action_payload("smoke-p%d" % i, 100 + 10 * i, 0, 1))

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(fire, range(5)))
    manager.engine.gate_hold_s = 0.0
    codes = [s for s, _ in results]
    accepted = sum(1 for s in codes if s == 200)
    conflicts = sum(1 for s in codes if s == 409)
    check("并发发送不同动作产生 409", conflicts >= 1, "状态码 %s" % codes)
    check("同一时刻只接受一个动作", accepted == 1, "状态码 %s" % codes)
    stats = manager.server.concurrency_stats()
    check("引擎内动作从未重叠执行", stats["engine_max_concurrent"] == 1, str(stats))
    for st, body in results:
        if st == 409:
            check("409 响应体 accepted=false 且 virtual_time_s=0",
                  body.get("accepted") is False and body.get("virtual_time_s") == 0, str(body))
            break
    # 顺序请求不受影响
    ok_seq = all(post(base, "/measure", action_payload("smoke-s%d" % i, 10 * i, 0, 1))[1].get("accepted") for i in range(3))
    check("串行请求全部被接受", ok_seq)
    post(base, "/exit", base_payload("smoke-cx"))

    # ---------------- 3. 真实进程端到端 ----------------
    print("\n[3] 机器狗程序端到端演练 (真实进程 + HTTP)")
    for problem, seed in ((3, 1001), (4, 2003)):
        info = manager.start_test("q%d_practice" % problem, seed=seed)
        code2 = info["case_code"]
        wait_interface_open(manager)
        proc = subprocess.run(
            [sys.executable, "-m", "robotdog.solver.deploy", "--url", base, "--team-id", TEAM,
             "--problem", str(problem), "--quiet"],
            cwd=ROOT, capture_output=True, text=True, timeout=600, encoding="utf-8",
        )
        check("问题%d: 机器狗进程正常结束 (退出码 0)" % problem, proc.returncode == 0, (proc.stderr or "")[-500:])
        for _ in range(60):
            if manager.engine.run.phase == "ended":
                break
            time.sleep(0.2)
        rep2 = manager.current_record
        ok2 = rep2 is not None and rep2["case_code"] == code2
        check("问题%d: 机器狗主动 /exit 结束测试" % problem, ok2 and rep2["end_reason"] == "user_exit",
              str(rep2 and rep2["end_reason"]))
        stats = (rep2 or {}).get("stats", {})
        check("问题%d: 清除个数不少于 1" % problem, (stats.get("cleared_count") or 0) >= 1,
              "%s / %s" % (stats.get("cleared_count"), (rep2 or {}).get("truth", {}).get("source_total")))
        # 两种时间口径必须分开校验 (附件1 §2.5 / 附件2 §4.5):
        #   现实 程序运行时间 <= 1200 s 是约束; 虚拟 世界活动时长 <= 360000 s
        #   只为防止死循环。二者独立, 现实 1 秒可推进任意多虚拟秒 (附件2 §1.4),
        #   因此**不能**断言虚拟总时长 <= 1200 s —— 旧断言正是这样掩盖了
        #   "把虚拟上限当现实预算"的缺陷。
        check("问题%d: 现实程序运行时间不超过 1200 s 预算" % problem,
              (stats.get("program_runtime_s") or 0) <= 1225.0,
              str(stats.get("program_runtime_s")))
        check("问题%d: 虚拟总时长不超过 360000 s 上限" % problem,
              (stats.get("total_duration_s") or 0) <= 360000.0,
              str(stats.get("total_duration_s")))
        if stats.get("cleared_count"):
            check(
                "问题%d: 平均定位清除时间 = 总时间 / 清除个数" % problem,
                abs(stats["avg_clear_duration_s"] - stats["total_duration_s"] / stats["cleared_count"]) < 1e-6,
            )
        parts = (stats.get("travel_duration_s", 0) + stats.get("switch_duration_s", 0)
                 + stats.get("detect_duration_s", 0) + stats.get("clear_duration_s", 0))
        check("问题%d: 耗时统计自洽 (移动+切换+检测+清除 = 总时间)" % problem,
              abs(parts - stats.get("total_duration_s", 0)) < 1e-3)
        acts = sum(a["total_duration_s"] for a in (rep2 or {}).get("actions", []))
        check("问题%d: 动作日志累计耗时 = 虚拟总时间" % problem,
              abs(acts - stats.get("total_duration_s", 0)) < 1e-3)
        if problem == 3:
            # 演练测试的实时状态必须显示干扰源总数 (正式测试才隐藏, 见第 4 节)
            practice_total = manager.ui_state()["status"]["source_total"]
            check("演练测试实时状态显示干扰源总数",
                  practice_total == (rep2 or {}).get("truth", {}).get("source_total"),
                  str(practice_total))

    # ---------------- 4. 正式测试日志 ----------------
    print("\n[4] 正式测试模块与加密日志 (附件1 4.6)")
    manager.start_test("q3_formal", seed=1009)
    wait_interface_open(manager)
    post(base, "/enter", base_payload("smoke-f1"))
    post(base, "/measure", action_payload("smoke-f2", 120, 80, 3))
    post(base, "/exit", base_payload("smoke-f3"))
    time.sleep(0.4)
    formal = [l for l in manager.log_list() if l["official"]]
    check("正式测试生成加密日志", bool(formal) and formal[0]["path"].endswith(".log"))
    if formal:
        with open(formal[0]["path"], "rb") as fh:
            blob = fh.read()
        check("加密日志小于 2 MB", len(blob) <= 2 * 1024 * 1024, "%d 字节" % len(blob))
        try:
            plain = json.loads(decrypt_log(blob).decode("utf-8"))
            check("加密日志可解密且内容完整", plain["case_code"] == formal[0]["case_code"])
        except Exception as exc:  # noqa: BLE001
            check("加密日志可解密且内容完整", False, str(exc))
    state = manager.ui_state()
    truth_total = manager.current_record["truth"]["source_total"]
    blob_text = json.dumps(state, ensure_ascii=False)
    check("正式测试界面不显示案例真值", state["last_report"]["truth"].get("hidden") is True)
    check("正式测试统计中的总数被掩码",
          state["last_report"]["stats"].get("source_total") is None
          and state["last_report"]["stats"].get("cleared_ratio") is None)
    check("正式测试实时状态不含总数", state["status"].get("source_total") is None)
    check("正式测试模块历史记录不含总数",
          all("total" not in r for r in state["modules"]["q3_formal"]["records"]))
    check("控制接口 JSON 中不出现真实总数",
          ('"source_total": %d' % truth_total) not in blob_text
          and ('"total": %d' % truth_total) not in blob_text)

    stop.set()
    manager.stop()
    if not args.keep_data:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    # ---------------- 5. 单元测试套件 ----------------
    if args.full:
        print("\n[5] 完整测试套件 (unittest)")
        result = subprocess.run(
            [sys.executable, "-W", "ignore::ResourceWarning", "-m", "unittest", "discover", "-s", "tests", "-t", "."],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        tail = (result.stderr or "")[-1500:]
        check("unittest 全部通过", result.returncode == 0, tail)
        print(tail.strip().splitlines()[-1] if tail.strip() else "")

    print("\n" + "=" * 84)
    print("冒烟测试结果: 通过 %d 项, 失败 %d 项" % (len(PASS), len(FAIL)))
    if FAIL:
        for name in FAIL:
            print("  失败: %s" % name)
    print("=" * 84)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
