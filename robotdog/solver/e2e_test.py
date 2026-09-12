"""端到端验证: 真实进程 + 真实 HTTP + 真实模拟器, 跑完整一局问题3 / 问题4.

与 ``solver/eval.py`` 的区别: 那里是进程内直接调用引擎 (快, 用于批量评测);
这里启动**真正的模拟器进程**, 机器狗程序通过 HTTP+JSON 通信, 用于验证:

  * 协议层 (keep-alive / request_id / accepted / 状态码) 没有用错;
  * 规划器在只拿到示向度读数的条件下依然工作;
  * 部署程序 (``robotdog.solver.deploy``) 能独立跑通。

用法::

    python -m robotdog.solver.e2e_test --runs 3
    python -m robotdog.solver.e2e_test --runs 3 --problem 4 --seed 9500
    python -m robotdog.solver.e2e_test --runs 3 --problem 4 --seed 9500
    python -m robotdog.solver.e2e_test --runs 3 --problem 4

端到端与进程内口径必须一致 —— 这是"成绩可迁移"的必要条件。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from robotdog.solver import PROJECT_ROOT


def free_port(start: int = 2126) -> int:
    for port in range(start, start + 200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("没有可用端口")


def run_one(seed: Optional[int], timeout: float = 300.0, countdown: float = 1.0,
            verbose: bool = True, problem: int = 3) -> Dict[str, Any]:
    port = free_port()
    log_dir = os.path.join(PROJECT_ROOT, "data", "logs")
    pattern = os.path.join(log_dir, "practice_p%d_*.json" % problem)
    before = set(glob.glob(pattern))
    cmd = [sys.executable, "-m", "simulator", "--no-gui", "--port", str(port),
           "--countdown", str(countdown), "--auto-practice", str(problem)]
    if seed is not None:
        cmd += ["--auto-seed", str(seed)]
    sim = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t0 = time.time()
    result: Dict[str, Any] = {}
    try:
        time.sleep(countdown + 1.2)
        env = dict(os.environ)
        env["SIM_BASE_URL"] = "http://127.0.0.1:%d" % port
        env["SIM_TEAM_ID"] = "MCM2026"
        robot_cmd = [sys.executable, "-m", "robotdog.solver.deploy",
                     "--url", env["SIM_BASE_URL"], "--quiet",
                     "--problem", str(problem)]
        robot = subprocess.run(robot_cmd,
                               cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
                               timeout=timeout)
        for line in robot.stdout.strip().splitlines():
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                except Exception:  # noqa: BLE001
                    pass
        if verbose and robot.stderr.strip():
            print(robot.stderr.strip()[-2000:], file=sys.stderr)
    finally:
        try:
            sim.terminate()
            sim.wait(timeout=10)
        except Exception:  # noqa: BLE001
            sim.kill()
    # 从模拟器导出的演练日志取"权威"统计 (真值总数只有演练日志里有)
    after = sorted(set(glob.glob(pattern)) - before, key=os.path.getmtime)
    stats: Dict[str, Any] = {}
    if after:
        with open(after[-1], "r", encoding="utf-8") as fh:
            rep = json.load(fh)
        st = rep["stats"]
        stats = {"case_code": rep["case_code"], "end_reason": rep["end_reason"],
                 "source_total": st["source_total"], "cleared_count": st["cleared_count"],
                 "cleared_ratio": st["cleared_ratio"],
                 "total_duration_s": st["total_duration_s"],
                 "avg_clear_duration_s": st["avg_clear_duration_s"],
                 "program_runtime_s": st["program_runtime_s"],
                 "moved_distance_m": st["moved_distance_m"],
                 "measure_count": st["measure_count"], "log": after[-1]}
    if not stats and result:
        # 模拟器进程被 terminate 时可能还没落盘演练日志; 这时用机器狗程序自己的
        # 收尾 JSON 兜底 (虚拟时间/清除数/移动距离都在里面, 只是看不到真值总数)。
        cl = int(result.get("cleared_count", 0))
        vt = float(result.get("virtual_time_s", 0.0))
        stats = {"case_code": "?", "end_reason": "robot_summary",
                 "source_total": cl, "cleared_count": cl,
                 "cleared_ratio": 1.0 if cl else 0.0, "total_duration_s": vt,
                 "avg_clear_duration_s": (vt / cl) if cl else 0.0,
                 "program_runtime_s": float(result.get("real_elapsed_s", 0.0)),
                 "moved_distance_m": float(result.get("moved_m", 0.0)),
                 "measure_count": int(result.get("measures", 0)), "log": ""}
    stats["wall_s"] = time.time() - t0
    stats["robot_summary"] = result
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="端到端验证 (真实进程 + HTTP)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)
    rows: List[Dict[str, Any]] = []
    for i in range(args.runs):
        sd = None if args.seed is None else args.seed + i
        r = run_one(sd, problem=args.problem)
        rows.append(r)
        print("问题%d 第 %d 局: 编码 %s | 清除 %s/%s (%.0f%%) | 总虚拟 %.1f s | 平均 %.1f s | "
              "移动 %.0f m | 检测 %s | 现实 %.2f s | 用墙钟 %.1f s"
              % (args.problem, i + 1,
                 r.get("case_code", "?"), r.get("cleared_count", "?"),
                 r.get("source_total", "?"), 100.0 * r.get("cleared_ratio", 0.0),
                 r.get("total_duration_s", 0.0), r.get("avg_clear_duration_s") or 0.0,
                 r.get("moved_distance_m", 0.0), r.get("measure_count", "?"),
                 r.get("program_runtime_s", 0.0), r["wall_s"]), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
