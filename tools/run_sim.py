"""启动一个模拟器会话并保持运行 (供手工/脚本冒烟测试)。

用法:
    python tools\run_sim.py --problem 3 --port 2026 --countdown 1 --auto-seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.session import SessionManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, default=3)
    ap.add_argument("--practice", action="store_true")
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--gui-port", type=int, default=2027)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--countdown", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=1500.0)
    ap.add_argument("--max-real", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--team-id", default="MCM2026")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--auto-seconds", type=float, default=0.0)
    ap.add_argument("--keep-alive", action="store_true")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    manager = SessionManager(
        team_id=args.team_id,
        host="127.0.0.1",
        port=args.port,
        data_dir=data_dir,
        countdown_s=args.countdown,
        window_s=args.window,
        max_real_s=args.max_real,
    )
    manager.login.login()
    port = manager.start()
    gui = None
    if not args.no_gui:
        from simulator.gui import GuiServer

        gui = GuiServer(manager, port=args.gui_port)
        gui.start()

    stop = threading.Event()

    def hb():
        while not stop.is_set():
            try:
                manager.server.tick()
            except Exception:
                pass
            stop.wait(0.2)

    threading.Thread(target=hb, daemon=True).start()

    info = {"base_url": manager.base_url, "gui": gui.url if gui else None, "log_dir": manager.log_dir}
    if args.practice or args.formal:
        module = "q%d_%s" % (args.problem, "formal" if args.formal else "practice")
        info["test"] = manager.start_test(module, seed=args.seed)
    print(json.dumps(info, ensure_ascii=False), flush=True)

    if args.auto_seconds > 0 and not args.keep_alive:
        time.sleep(args.auto_seconds)
        if manager.engine.run and manager.engine.run.phase != "ended":
            manager.abort_test()
        stop.set()
        manager.stop()
        if gui:
            gui.stop()
        return 0
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        manager.stop()
        if gui:
            gui.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
