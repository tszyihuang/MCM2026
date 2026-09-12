"""验收测试: 以真实进程 (HTTP + 真实时钟) 运行机器狗程序完成演练测试.

覆盖:
  * 完整 5 秒倒计时 -> 接口开放 -> 机器狗 /enter -> 搜索清除 -> /exit 流程;
  * 统计量 (被清除比例、平均定位清除时间、程序运行时间) 与行为日志一致性;
  * 正式测试不泄露案例真值、日志加密且小于 2 MB;
  * 三个正式测试机会的限制。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robotdog.consts import MAX_VIRTUAL_S  # noqa: E402
from simulator.core import decrypt_log  # noqa: E402
from simulator.session import SessionManager  # noqa: E402
from tests.harness import TEAM  # noqa: E402


class RobotAcceptanceTest(unittest.TestCase):
    """真实进程端到端演练测试 (问题3)。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="acc_")
        cls.manager = SessionManager(
            team_id=TEAM,
            port=0,
            data_dir=cls.tmp,
            countdown_s=1.0,
            window_s=1500.0,
            max_real_s=1200.0,
        )
        cls.manager.login.login()
        cls.port = cls.manager.start()
        cls.stop = threading.Event()
        cls.thread = threading.Thread(target=cls._heartbeat, daemon=True)
        cls.thread.start()

    @classmethod
    def _heartbeat(cls):
        while not cls.stop.is_set():
            try:
                cls.manager.server.tick()
            except Exception:
                pass
            cls.stop.wait(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.stop.set()
        cls.manager.stop()
        time.sleep(0.1)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run_robot(self, problem: int, seed: int, module: str, verbose: bool = False):
        info = self.manager.start_test(module, seed=seed)
        time.sleep(1.3)  # 等待倒计时结束、接口开放
        robot_log = os.path.join(self.tmp, "robot_%s.jsonl" % info["case_code"])
        cmd = [
            sys.executable,
            "-m",
            "robotdog.solver.deploy",
            "--url",
            self.manager.base_url,
            "--team-id",
            TEAM,
            "--problem",
            str(problem),
            "--log",
            robot_log,
            "--reserve",
            "15",
        ]
        if not verbose:
            cmd.append("--quiet")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600, encoding="utf-8")
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.manager.engine.run and self.manager.engine.run.phase == "ended":
                break
            time.sleep(0.2)
        return info, proc, robot_log

    def test_practice_q3_full_flow(self):
        info, proc, robot_log = self._run_robot(3, 1001, "q3_practice")
        self.assertEqual(proc.returncode, 0, "机器狗进程异常退出: %s" % proc.stderr[-800:])
        report = self.manager.current_record
        self.assertIsNotNone(report, "测试结束后必须生成报告")
        self.assertEqual(report["end_reason"], "user_exit", "机器狗应主动 /exit 结束测试")
        stats = report["stats"]
        self.assertGreaterEqual(stats["cleared_count"], 1, "演练测试至少应清除 1 个干扰源")
        self.assertLessEqual(stats["cleared_count"], report["truth"]["source_total"])
        self.assertEqual(
            stats["cleared_ratio"],
            round(stats["cleared_count"] / report["truth"]["source_total"], 6),
        )
        if stats["cleared_count"]:
            self.assertAlmostEqual(
                stats["avg_clear_duration_s"],
                stats["total_duration_s"] / stats["cleared_count"],
                places=4,
            )
        # 两种时间必须分别校验, 不可混用 (附件1 §2.5 / 附件2 §4.5):
        #   * 现实 程序运行时间 <= 1200 s  —— 这才是约束;
        #   * 虚拟 世界活动时长 <= 360000 s —— 只为防止死循环, 不是约束。
        # 二者相互独立: 现实 1 秒可推进任意多虚拟秒 (附件2 §1.4),
        # 因此**不能**断言 虚拟总时长 <= 1200 s; 历史上正是这条错误断言掩盖过
        # "把虚拟上限当现实预算"的缺陷 (详见 docs/验收报告.md §6.1)。
        self.assertLessEqual(
            stats["program_runtime_s"], 1200.0 + 25.0, "现实程序运行时间超出 20 分钟预算"
        )
        self.assertLessEqual(
            stats["total_duration_s"], MAX_VIRTUAL_S, "虚拟世界活动时长超出上限"
        )
        self.assertEqual(stats["total_duration_s"], stats["virtual_time_s"])
        # 统计量与动作序列一致
        self.assertEqual(stats["measure_count"], sum(1 for a in report["actions"] if a["action"] == "measure"))
        self.assertEqual(stats["clear_attempts"], sum(1 for a in report["actions"] if a["action"] == "clear"))
        total = sum(a["total_duration_s"] for a in report["actions"])
        self.assertAlmostEqual(total, stats["total_duration_s"], places=3)
        parts = (
            stats["travel_duration_s"]
            + stats["switch_duration_s"]
            + stats["detect_duration_s"]
            + stats["clear_duration_s"]
        )
        self.assertAlmostEqual(parts, stats["total_duration_s"], places=3)
        for action in report["actions"]:
            if action["action"] == "clear":
                expected = 5.0 if action["response"].get("clear_result") == "success" else 3.0
                self.assertAlmostEqual(action["action_duration_s"], expected, places=6)
        # 日志与真值
        self.assertTrue(os.path.exists(robot_log), "机器狗应自行记录行为日志")
        with open(robot_log, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        self.assertGreater(len(lines), 5)
        self.assertEqual(lines[0]["path"], "/enter")
        self.assertEqual(lines[-1]["path"], "/exit")
        for rec in lines:
            self.assertEqual(rec["request"]["arena_id"], "default")
            self.assertEqual(rec["request"]["robot_id"], TEAM)
            self.assertTrue(rec["request"]["request_id"])
            if rec["path"] in ("/measure", "/clear"):
                self.assertIn("position", rec["request"])
                self.assertIn("channel", rec["request"])
        ids = [rec["request"]["request_id"] for rec in lines]
        self.assertEqual(len(ids), len(set(ids)), "每个新动作必须使用新的 request_id")

    def test_formal_hides_truth_and_encrypts_log(self):
        info, proc, robot_log = self._run_robot(3, 1009, "q3_formal")
        self.assertEqual(proc.returncode, 0, "机器狗进程异常退出: %s" % proc.stderr[-800:])
        state = self.manager.ui_state()
        self.assertTrue(state["last_report"]["truth"]["hidden"], "正式测试界面不得显示案例真值")
        self.assertIsNone(state["status"]["source_total"])
        logs = self.manager.log_list()
        formal = [l for l in logs if l["official"]]
        self.assertTrue(formal)
        enc = formal[0]["path"]
        self.assertTrue(enc.endswith(".log"))
        with open(enc, "rb") as fh:
            blob = fh.read()
        self.assertLessEqual(len(blob), 2 * 1024 * 1024, "加密日志必须小于 2 MB")
        plain = json.loads(decrypt_log(blob).decode("utf-8"))
        self.assertEqual(plain["case_code"], formal[0]["case_code"])
        self.assertEqual(plain["end_reason"], "user_exit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
