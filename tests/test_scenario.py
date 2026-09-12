"""端到端场景测试: 四个测试模块、倒计时、窗口/程序超时、日志导出与加密、界面接口。

对照附件1 第4节与附件2 第4.5节。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import http.client

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.core import decrypt_log  # noqa: E402
from simulator.session import SessionManager, validate_password  # noqa: E402
from tests.harness import TEAM  # noqa: E402


def http_post(port: int, path: str, payload: dict):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw.decode("utf-8")) if raw else None
    finally:
        conn.close()


def http_get(port: int, path: str):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, raw
    finally:
        conn.close()


class SessionBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="simtest_")
        self.manager = SessionManager(
            team_id=TEAM,
            port=0,
            data_dir=self.tmp,
            countdown_s=0.4,
            window_s=1500.0,
            max_real_s=1200.0,
        )
        self.manager.login.login()
        self.port = self.manager.start()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._heartbeat, daemon=True)
        self._t.start()

    def _heartbeat(self):
        while not self._stop.is_set():
            try:
                self.manager.server.tick()
            except Exception:
                pass
            self._stop.wait(0.05)

    def tearDown(self):
        self._stop.set()
        self.manager.stop()
        time.sleep(0.05)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def base(self, rid):
        return {"arena_id": "default", "robot_id": TEAM, "request_id": rid}

    def action(self, rid, x, y, ch):
        p = self.base(rid)
        p["position"] = {"x": x, "y": y}
        p["channel"] = ch
        return p


class LifecycleTest(SessionBase):
    def test_countdown_then_interface_open(self):
        info = self.manager.start_test("q3_practice", seed=7)
        self.assertIn("case_code", info)
        status, body = http_post(self.port, "/enter", self.base("e1"))
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"], "倒计时期间必须关闭接口")
        time.sleep(0.6)
        status, body = http_post(self.port, "/enter", self.base("e2"))
        self.assertTrue(body["accepted"])
        state = self.manager.ui_state()
        self.assertTrue(state["status"]["interface_open"])
        self.assertEqual(state["status"]["cleared"], 0)

    def test_practice_reports_truth(self):
        self.manager.start_test("q4_practice", seed=11)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        state = self.manager.ui_state()
        report = state["last_report"]
        self.assertFalse(report["truth"].get("hidden"))
        self.assertIn("source_total", report["truth"])
        self.assertGreaterEqual(report["truth"]["source_total"], 10)
        self.assertLessEqual(report["truth"]["source_total"], 16)
        self.assertEqual(
            report["truth"]["omni_count"] + report["truth"]["directional_count"],
            report["truth"]["source_total"],
        )

    def test_formal_hides_truth(self):
        self.manager.start_test("q3_formal", seed=12)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        state = self.manager.ui_state()
        self.assertTrue(state["last_report"]["truth"]["hidden"])
        self.assertEqual(state["last_report"]["uncleared"], [])
        self.assertIsNone(state["status"]["source_total"])
        # 正式测试机会计数
        self.assertEqual(self.manager.modules["q3_formal"].attempts_used, 1)

    def test_formal_truth_not_leaked_through_control_api(self):
        """附件1 4.6: 正式测试真值不得出现在控制接口返回的任何字段中。"""
        self.manager.start_test("q3_formal", seed=13)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/measure", self.action("m1", 100, 100, 1))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.25)
        truth_total = self.manager.current_record["truth"]["source_total"]
        self.assertGreaterEqual(truth_total, 10)
        state = self.manager.ui_state()
        blob = json.dumps(state, ensure_ascii=False)
        # 1) 报告统计中的真值字段必须被掩码
        self.assertIsNone(state["last_report"]["stats"]["source_total"])
        self.assertIsNone(state["last_report"]["stats"]["cleared_ratio"])
        # 2) 真值主体必须完全隐藏
        self.assertTrue(state["last_report"]["truth"]["hidden"])
        self.assertNotIn("sources", state["last_report"]["truth"])
        # 3) 实时状态中的总数必须为空
        self.assertIsNone(state["status"]["source_total"])
        # 4) 模块历史记录不得带 total
        for rec in state["modules"]["q3_formal"]["records"]:
            self.assertNotIn("total", rec)
        # 5) 整个 JSON 中不得出现真实总数
        self.assertNotIn('"source_total": %d' % truth_total, blob)
        self.assertNotIn('"total": %d' % truth_total, blob)

    def test_practice_still_shows_truth(self):
        self.manager.start_test("q3_practice", seed=14)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        state = self.manager.ui_state()
        self.assertIsNotNone(state["status"]["source_total"])
        self.assertIsNotNone(state["last_report"]["stats"]["source_total"])
        self.assertIn("total", state["modules"]["q3_practice"]["records"][0])

    def test_formal_attempt_limit(self):
        for i in range(3):
            self.manager.start_test("q3_formal", seed=100 + i)
            time.sleep(0.5)
            http_post(self.port, "/enter", self.base("e%d" % i))
            http_post(self.port, "/exit", self.base("x%d" % i))
            time.sleep(0.15)
        with self.assertRaises(RuntimeError):
            self.manager.start_test("q3_formal", seed=999)

    def test_practice_unlimited(self):
        for i in range(5):
            self.manager.start_test("q3_practice", seed=200 + i)
            time.sleep(0.5)
            http_post(self.port, "/enter", self.base("e%d" % i))
            http_post(self.port, "/exit", self.base("x%d" % i))
            time.sleep(0.1)
        self.assertEqual(self.manager.modules["q3_practice"].attempts_used, 0)

    def test_manual_abort(self):
        self.manager.start_test("q3_practice", seed=5)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        report = self.manager.abort_test()
        self.assertEqual(report["end_reason"], "manual_abort")
        status, body = http_post(self.port, "/measure", self.action("m1", 0, 0, 1))
        self.assertFalse(body["accepted"])

    def test_cannot_start_while_running(self):
        self.manager.start_test("q3_practice", seed=5)
        with self.assertRaises(RuntimeError):
            self.manager.start_test("q3_practice", seed=6)


class LogExportTest(SessionBase):
    def test_practice_log_readable(self):
        self.manager.start_test("q3_practice", seed=21)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/measure", self.action("m1", 100, 0, 1))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        logs = self.manager.log_list()
        self.assertEqual(len(logs), 1)
        self.assertTrue(os.path.exists(logs[0]["readable_path"]))
        with open(logs[0]["readable_path"], encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["case_code"], logs[0]["case_code"])
        self.assertEqual(data["stats"]["measure_count"], 1)
        self.assertGreaterEqual(len(data["actions"]), 3)

    def test_formal_log_encrypted_and_decryptable(self):
        self.manager.start_test("q4_formal", seed=31)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        logs = self.manager.log_list()
        enc = logs[0]["path"]
        self.assertTrue(enc.endswith(".log"))
        with open(enc, "rb") as fh:
            blob = fh.read()
        self.assertEqual(blob[:4], b"ENC1")
        self.assertLessEqual(len(blob), 2 * 1024 * 1024, "加密日志必须小于 2 MB")
        plain = json.loads(decrypt_log(blob).decode("utf-8"))
        self.assertEqual(plain["case_code"], logs[0]["case_code"])
        self.assertNotIn("干扰源", blob[:200].decode("latin-1", "ignore"))

    def test_export_endpoint(self):
        self.manager.start_test("q3_formal", seed=41)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        paths = self.manager.export_current()
        self.assertIn("encrypted", paths)
        self.assertTrue(os.path.exists(paths["encrypted"]))

    def test_report_statistics_match_actions(self):
        self.manager.start_test("q3_practice", seed=51)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        http_post(self.port, "/measure", self.action("m1", 300, 400, 1))
        http_post(self.port, "/clear", self.action("c1", 300, 0, 3))
        http_post(self.port, "/measure", self.action("m2", 300, 0, 2))
        http_post(self.port, "/exit", self.base("x1"))
        time.sleep(0.2)
        report = self.manager.current_record
        stats = report["stats"]
        # 500 m 移动 100 s + 检测 5 s (频道 1 无切换)
        # + 400 m 移动 80 s + 清除 3 s (未发现, /clear 不切换频道)
        # + 原地检测 5 s + 频道 1->2 切换 1 s = 194 s
        self.assertAlmostEqual(stats["total_duration_s"], 194.0, places=6)
        self.assertAlmostEqual(stats["moved_distance_m"], 900.0, places=6)
        self.assertEqual(stats["measure_count"], 2)
        self.assertAlmostEqual(stats["detect_duration_s"], 10.0, places=6)
        self.assertAlmostEqual(stats["clear_duration_s"], 3.0, places=6)
        self.assertAlmostEqual(stats["switch_duration_s"], 1.0, places=6)
        self.assertEqual(len(report["actions"]), 5)  # enter + measure + clear + measure + exit
        self.assertEqual(report["actions"][-1]["action"], "exit")
        # 动作日志与虚拟时钟、各项耗时统计必须完全一致
        action_sum = sum(a["total_duration_s"] for a in report["actions"])
        self.assertAlmostEqual(action_sum, stats["total_duration_s"], places=6)
        parts = (
            stats["travel_duration_s"]
            + stats["switch_duration_s"]
            + stats["detect_duration_s"]
            + stats["clear_duration_s"]
        )
        self.assertAlmostEqual(parts, stats["total_duration_s"], places=6)
        self.assertEqual(stats["clear_attempts"], 1)
        self.assertEqual(stats["channel_switches"], 1)
        if stats["cleared_count"]:
            self.assertAlmostEqual(
                stats["avg_clear_duration_s"],
                stats["total_duration_s"] / stats["cleared_count"],
                places=6,
            )


class GuiTest(SessionBase):
    def setUp(self):
        super().setUp()
        from simulator.gui import GuiServer

        self.gui = GuiServer(self.manager, port=0)
        self.gui_port = self.gui.start()

    def tearDown(self):
        self.gui.stop()
        super().tearDown()

    def test_index_served(self):
        status, body = http_get(self.gui_port, "/")
        self.assertEqual(status, 200)
        self.assertIn("无线电干扰源环境模拟器".encode("utf-8"), body)

    def test_state_endpoint(self):
        status, body = http_get(self.gui_port, "/api/state")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("status", data)
        self.assertIn("modules", data)
        self.assertEqual(set(data["modules"]), {"q3_practice", "q4_practice", "q3_formal", "q4_formal"})

    def test_start_via_gui_api(self):
        status, body = http_get(self.gui_port, "/api/state")
        conn = http.client.HTTPConnection("127.0.0.1", self.gui_port, timeout=10)
        conn.request(
            "POST",
            "/api/start",
            body=json.dumps({"module": "q3_practice", "seed": 77}).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()
        self.assertTrue(data["ok"])
        self.assertTrue(data["case_code"])

    def test_exchange_log_visible(self):
        self.manager.start_test("q3_practice", seed=88)
        time.sleep(0.6)
        http_post(self.port, "/enter", self.base("e1"))
        time.sleep(0.2)
        status, body = http_get(self.gui_port, "/api/state")
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(any(e.get("path") == "/enter" for e in data["exchanges"]))


class PasswordRuleTest(unittest.TestCase):
    def test_rules(self):
        self.assertIsNotNone(validate_password("short1", "1234567890123"))
        self.assertIsNotNone(validate_password("aaaaaaaaaaaa", "1234"))  # 无数字
        self.assertIsNotNone(validate_password("123456789012", "1234"))  # 无字母
        self.assertIsNotNone(validate_password("abc123456789", "123456789012"))  # 含队号连续 6 位
        self.assertIsNone(validate_password("Team2026Pass!x", "9876543210"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
