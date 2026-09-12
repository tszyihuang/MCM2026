"""模拟器命令行入口.

用法示例:
    python -m simulator --team-id MCM2026 --port 2026 --gui-port 2027
    python -m simulator --team-id MCM2026 --auto-practice 3 --robot "python -m robotdog.solver.deploy"

界面地址: http://127.0.0.1:<gui-port>/
机器狗接口: http://127.0.0.1:<port>/
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

from .gui import GuiServer
from .session import SessionManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="无线电干扰源环境模拟器 (2026 高教社杯 B 题)")
    parser.add_argument("--team-id", default=os.environ.get("SIM_TEAM_ID", "MCM2026"), help="参赛队号, 用作 robot_id")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址 (默认仅本机回环)")
    parser.add_argument("--port", type=int, default=2026, help="机器狗接口端口, 默认 2026")
    parser.add_argument("--gui-port", type=int, default=2027, help="模拟器界面端口, 默认 2027")
    parser.add_argument("--no-gui", action="store_true", help="不启动网页界面")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    parser.add_argument("--login-mode", choices=["offline", "online"], default="offline", help="登录模式")
    parser.add_argument("--login-server", default=os.environ.get("SIM_LOGIN_SERVER", ""), help="联网登录服务器地址")
    parser.add_argument("--countdown", type=float, default=5.0, help="倒计时秒数, 默认 5")
    parser.add_argument("--window", type=float, default=25 * 60.0, help="测试窗口秒数, 默认 1500")
    parser.add_argument("--max-real", type=float, default=1200.0, help="程序运行时间上限秒数, 默认 1200")
    parser.add_argument("--max-virtual", type=float, default=360000.0, help="虚拟世界限时秒数, 默认 360000")
    parser.add_argument("--enforce-deadline", action="store_true", help="启用 2026-09-13 17:30 启动截止检查")
    parser.add_argument("--auto-practice", type=int, choices=[3, 4], help="启动后自动开始一次演练测试")
    parser.add_argument("--auto-formal", type=int, choices=[3, 4], help="启动后自动开始一次正式测试")
    parser.add_argument("--auto-seed", type=int, default=None, help="自动测试使用的固定随机种子")
    parser.add_argument("--robot", default=None,
                        help="自动启动的机器狗程序命令 (以空格分隔, 支持引号路径; 不支持重定向)")
    parser.add_argument("--robot-delay", type=float, default=1.0, help="倒计时结束后延迟多久启动机器狗程序")
    parser.add_argument("--run-seconds", type=float, default=0.0, help="自动运行指定秒数后退出 (0 表示常驻)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manager = SessionManager(
        team_id=args.team_id,
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        login_mode=args.login_mode,
        login_server=args.login_server,
        countdown_s=args.countdown,
        window_s=args.window,
        max_real_s=args.max_real,
        max_virtual_s=args.max_virtual,
        enforce_deadline=args.enforce_deadline,
    )
    try:
        state = manager.login.login()
    except Exception as exc:  # noqa: BLE001
        print("[模拟器] 登录失败: %s" % exc, file=sys.stderr)
        return 2
    port = manager.start()
    gui = None
    if not args.no_gui:
        gui = GuiServer(manager, host=args.host, port=args.gui_port)
        gui.start()

    print("=" * 78)
    print("无线电干扰源环境模拟器 (2026 高教社杯全国大学生数学建模竞赛 B 题 · 自建复现版)")
    print("  参赛队号 (robot_id): %s" % args.team_id)
    print("  登录状态          : %s" % state.detail)
    print("  机器狗接口        : http://%s:%d/   (POST /enter /measure /clear /exit)" % (args.host, port))
    if gui is not None:
        print("  模拟器界面        : %s" % gui.url)
    print("  行为日志目录      : %s" % manager.log_dir)
    print("  倒计时/窗口/程序  : %.0f s / %.0f s / %.0f s" % (args.countdown, args.window, args.max_real))
    print("=" * 78)

    stop_event = threading.Event()

    def _signal(_signum, _frame):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal)
        except Exception:
            pass

    # 后台心跳: 推进阶段 (倒计时 -> 接口开放 -> 超时结束)
    def _heartbeat() -> None:
        while not stop_event.is_set():
            try:
                manager.server.tick()
            except Exception as exc:  # noqa: BLE001
                # 心跳绝不能中断, 但也绝不能静默: tick() 是唯一推进倒计时/窗口/
                # 超时的机制, 一旦抛错而无人知晓, 现场表现就是"测试永远不结束"。
                print("[模拟器] tick 异常: %r" % (exc,), file=sys.stderr)
            stop_event.wait(0.2)

    threading.Thread(target=_heartbeat, name="simulator-tick", daemon=True).start()

    robot_proc: Optional[subprocess.Popen] = None
    if args.auto_practice or args.auto_formal:
        module = ("q%d_practice" if args.auto_practice else "q%d_formal") % (
            args.auto_practice or args.auto_formal
        )
        info = manager.start_test(module, seed=args.auto_seed)
        print("[模拟器] 已启动 %s, 案例编码 %s, 倒计时 %.0f 秒" % (module, info["case_code"], args.countdown))
        if args.robot:
            def _launch() -> None:
                # 用 stop_event.wait 而非 sleep: 若模拟器在倒计时结束前就该退出
                # (例如 --run-seconds 很小), 这里直接放弃启动, 不会在退出后又
                # 生出一个没人回收的机器狗进程。
                if stop_event.wait(args.countdown + args.robot_delay):
                    return
                nonlocal robot_proc
                cmd = shlex.split(args.robot, posix=False)
                env = dict(os.environ)
                env["SIM_BASE_URL"] = manager.base_url
                env["SIM_TEAM_ID"] = args.team_id
                if args.auto_seed is not None:
                    env["SIM_SEED"] = str(args.auto_seed)
                robot_proc = subprocess.Popen(cmd, env=env)
                print("[模拟器] 已启动机器狗程序: %s (pid=%s)" % (args.robot, robot_proc.pid))

            threading.Thread(target=_launch, name="robot-launcher", daemon=True).start()

    try:
        deadline = time.monotonic() + args.run_seconds if args.run_seconds > 0 else None
        while not stop_event.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                break
            stop_event.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if robot_proc is not None and robot_proc.poll() is None:
            robot_proc.terminate()
            try:
                robot_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                robot_proc.kill()
        if gui is not None:
            gui.stop()
        manager.stop()
    print("[模拟器] 已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
