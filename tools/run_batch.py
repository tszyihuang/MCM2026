"""演练/正式测试批处理运行器.

流程: 启动模拟器 -> 等待 5 秒倒计时 -> 启动机器狗程序 (/enter) -> 等待结束 ->
导出行为日志 -> 输出统计表 (被清除比例 / 平均定位清除时间 / 程序运行时间)。

``--jobs N`` 可把 N 局分发到 N 个进程同时跑 (每局独享一个模拟器实例与端口),
8 核机器上跑 8 局的时间与跑 1 局相当。各局之间互不共享状态, 因此结果与串行一致。

用法:
    python tools/run_batch.py --problem 3 --module practice --runs 5 --seed 1001
    python tools/run_batch.py --problem 3 --module formal --runs 3
    python tools/run_batch.py --problem 4 --module formal --runs 8 --jobs 8   # 多核并行
    python tools/run_batch.py --problem 4 --module formal --runs 3 --robot-args "--reserve 10"
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue as queue_module
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simulator.session import SessionManager  # noqa: E402

RESULTS_PATH = os.path.join(ROOT, "data", "results.jsonl")


def run_one(
    problem: int,
    module_kind: str,
    seed,
    countdown: float,
    window: float,
    max_real: float,
    robot_args,
    team_id: str,
    data_dir: str,
    port: int,
    gui: bool,
    verbose_robot: bool,
    robot_module: str = "robotdog.solver.deploy",
) -> dict:
    module = "q%d_%s" % (problem, module_kind)
    manager = SessionManager(
        team_id=team_id,
        host="127.0.0.1",
        port=port,
        data_dir=data_dir,
        countdown_s=countdown,
        window_s=window,
        max_real_s=max_real,
    )
    manager.login.login()
    real_port = manager.start()
    gui_server = None
    if gui:
        from simulator.gui import GuiServer

        gui_server = GuiServer(manager, port=0)
        gui_server.start()

    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                manager.server.tick()
            except Exception:
                pass
            stop.wait(0.05)

    threading.Thread(target=heartbeat, daemon=True).start()

    info = manager.start_test(module, seed=seed)
    case_code = info["case_code"]

    robot_log = os.path.join(data_dir, "robot_logs", "%s.jsonl" % case_code)
    os.makedirs(os.path.dirname(robot_log), exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        robot_module,
        "--url",
        manager.base_url,
        "--team-id",
        team_id,
        "--problem",
        str(problem),
        "--log",
        robot_log,
    ] + list(robot_args)
    if not verbose_robot:
        cmd.append("--quiet")

    proc = None
    if countdown > 0:
        time.sleep(countdown + 0.2)
    proc = subprocess.Popen(cmd, cwd=ROOT)

    deadline = time.monotonic() + window + countdown + 30.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    # 机器狗退出后若测试仍未结束 (异常情况), 手工中止以生成日志
    for _ in range(40):
        if manager.engine.run is not None and manager.engine.run.phase == "ended":
            break
        time.sleep(0.25)
    if manager.engine.run is not None and manager.engine.run.phase != "ended":
        manager.abort_test()

    stop.set()
    report = manager.current_record or {}
    paths = manager.log_list()
    if gui_server:
        gui_server.stop()
    manager.stop()

    stats = (report or {}).get("stats", {})
    truth = (report or {}).get("truth", {})
    return {
        "problem": problem,
        "module": module,
        "case_code": case_code,
        "seed": seed,
        "end_reason": report.get("end_reason"),
        "cleared_count": stats.get("cleared_count"),
        "source_total": truth.get("source_total"),
        "cleared_ratio": stats.get("cleared_ratio"),
        "avg_clear_duration_s": stats.get("avg_clear_duration_s"),
        "total_duration_s": stats.get("total_duration_s"),
        "program_runtime_s": stats.get("program_runtime_s"),
        "measure_count": stats.get("measure_count"),
        "clear_attempts": stats.get("clear_attempts"),
        "moved_distance_m": stats.get("moved_distance_m"),
        "log": (paths[0]["path"] if paths else None),
        "robot_log": robot_log,
    }


def _chunk_job(job: Dict[str, Any]) -> List[tuple]:
    """工作进程: 反复"领取一局 -> 跑一局", 直到队列取空。

    这是给"小批量 + 多核"用的: 若把 N 局一次性静态分给 N 个进程, 快核跑完后
    只能空等慢核; 改成按需领取后, 谁先空出来谁接着领下一局, 核心不会闲置。
    每局仍然新建独立的 ``SessionManager`` 与端口, 因此局与局之间没有状态残留。

    队列元素是 ``(序号, seed)``; ``seed is None`` 表示随机种子。这里**不能**用
    字符串哨兵 + ``is`` 判断: 对象经 ``Manager().Queue()`` pickle 往返后不再是
    同一个对象, 哨兵会被当成真正的 seed 传下去。
    """
    base = job["payload"]
    queue = job["queue"]
    out: List[tuple] = []
    while True:
        try:
            index, seed = queue.get_nowait()
        except queue_module.Empty:
            return out
        out.append((index, run_one(seed=seed, **base)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--module", choices=["practice", "formal"], default="practice")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None, help="首个案例的随机种子, 逐次 +1")
    ap.add_argument("--countdown", type=float, default=5.0)
    ap.add_argument("--window", type=float, default=1500.0)
    ap.add_argument("--max-real", type=float, default=1200.0)
    ap.add_argument("--team-id", default="MCM2026")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--verbose-robot", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="并行进程数; 每局独立模拟器实例与端口, 结果与串行一致 "
                         "(0 = 自动取 CPU 逻辑核数)")
    ap.add_argument("--robot-module", default="robotdog.solver.deploy",
                    help="机器狗程序模块 (默认 robotdog.solver.deploy)")
    ap.add_argument("--robot-args", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, args.runs))
    if jobs > 1 and args.gui:
        print("[警告] --gui 与 --jobs>1 同时使用时, 只有第一个并行批次会开界面", file=sys.stderr)

    common: Dict[str, Any] = dict(
        problem=args.problem,
        module_kind=args.module,
        countdown=args.countdown,
        window=args.window,
        max_real=args.max_real,
        # --robot-args 用 REMAINDER 收集, 用户既可能写 `--robot-args --reserve 10`
        # (多个 token), 也可能写 `--robot-args "--reserve 10"` (一个带空格的字符串)。
        # 两种都要能用: 带空格的元素按空白再切一次。
        robot_args=[a for tok in args.robot_args
                    for a in (tok.split() if " " in tok else [tok]) if a != "--"],
        team_id=args.team_id,
        data_dir=args.data_dir,
        port=args.port,
        gui=args.gui,
        verbose_robot=args.verbose_robot,
        robot_module=args.robot_module,
    )
    seeds: List[Optional[int]] = [
        None if args.seed is None else args.seed + i for i in range(args.runs)
    ]
    print("将运行 %d 局 (%s 测试, 问题%d), 并行进程数 %d" % (args.runs, args.module, args.problem, jobs))
    t_start = time.time()
    if jobs <= 1:
        rows = []
        for i, seed in enumerate(seeds):
            print("=" * 90)
            print("第 %d/%d 次 %s测试 (问题%d, seed=%s)"
                  % (i + 1, args.runs, args.module, args.problem, seed))
            print("=" * 90)
            row = run_one(seed=seed, **common)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        # 用 Manager().Queue 做工作队列: 各进程按需领取, 天然负载均衡。
        manager = multiprocessing.Manager()
        try:
            task_queue = manager.Queue()
            for i, seed in enumerate(seeds):
                task_queue.put((i, seed))
            jobs_payload = [{"payload": common, "queue": task_queue} for _ in range(jobs)]
            indexed: List[tuple] = []
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                for chunk in pool.map(_chunk_job, jobs_payload, chunksize=1):
                    indexed.extend(chunk)
        finally:
            manager.shutdown()
        # 领取顺序与完成顺序无关, 按序号还原成与串行一致的报表顺序
        indexed.sort(key=lambda item: item[0])
        rows = [row for _, row in indexed]
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, indent=2))

    elapsed = time.time() - t_start
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 90)
    print("%-10s %-26s %-8s %-8s %-10s %-12s %s" % ("模块", "案例编码", "已清除", "总数", "清除比例", "平均耗时(s)", "程序运行(s)"))
    print("-" * 90)
    for row in rows:
        print(
            "%-10s %-26s %-8s %-8s %-10s %-12s %s"
            % (
                row["module"],
                row["case_code"],
                row["cleared_count"],
                row["source_total"],
                ("%.1f%%" % (100.0 * row["cleared_ratio"])) if row["cleared_ratio"] is not None else "-",
                ("%.1f" % row["avg_clear_duration_s"]) if row["avg_clear_duration_s"] else "-",
                ("%.1f" % row["program_runtime_s"]) if row["program_runtime_s"] is not None else "-",
            )
        )
    done = [r for r in rows if r.get("cleared_ratio") is not None]
    print("-" * 90)
    print("完成 %d/%d 局, 总耗时 %.1f s (并行 %d, 串行等效 %.1f s)"
          % (len(done), len(rows), elapsed, jobs, elapsed * jobs))
    if done:
        print("平均清除比例 %.1f%%   平均程序运行 %.1f s"
              % (100.0 * sum(r["cleared_ratio"] for r in done) / len(done),
                 sum(r["program_runtime_s"] or 0.0 for r in done) / len(done)))
    if len(done) < len(rows):
        print("[注意] 有 %d 局没有生成日志: 机器狗程序可能启动即失败, 请加 --verbose-robot 查看"
              % (len(rows) - len(done)), file=sys.stderr)
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
