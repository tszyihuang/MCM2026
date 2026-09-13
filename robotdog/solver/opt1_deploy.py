"""opt1 策略的部署入口（真实模拟器进程 + HTTP+JSON）。

与 ``robotdog.solver.deploy`` 使用同一套 ``SimClient`` / ``RemoteWorld``，
但决策层换成 opt1 单文件策略。可直接被 ``tools/run_batch.py`` 作为
``--robot-module`` 调用::

    python tools/run_batch.py --problem 4 --module practice --runs 3 --seed 9500 \
        --countdown 0 --robot-module robotdog.solver.opt1_deploy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

from .deploy import RemoteWorld, SimClient
from .opt1_port import _NullBelief, _Opt1Ctx, _load_opt1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="opt1 问题4 部署入口")
    ap.add_argument("--problem", type=int, default=4, choices=(4,))
    ap.add_argument("--url", default=os.environ.get("SIM_BASE_URL", "http://127.0.0.1:2026"))
    ap.add_argument("--team-id", default=os.environ.get("SIM_TEAM_ID", "MCM2026"))
    ap.add_argument("--log", default=os.environ.get("SIM_ROBOT_LOG", ""))
    ap.add_argument("--quiet", action="store_true")
    # 与 deploy.py 的 CLI 保持兼容，run_batch.py 会传这些参数。
    ap.add_argument("--reserve", type=float, default=20.0)
    ap.add_argument("--wait-enter", type=float, default=0.0)
    ap.add_argument("--cfg-module", default="")
    args = ap.parse_args(argv)

    if args.wait_enter > 0:
        time.sleep(args.wait_enter)

    client = SimClient(args.url, args.team_id, log_path=args.log or None)
    try:
        enter = client.enter()
        if enter.get("accepted") is not True:
            print("[opt1] /enter 失败: %s" % json.dumps(enter, ensure_ascii=False),
                  file=sys.stderr)
            return 1
        if not args.quiet:
            print("[opt1] 进入目标区域 | 可用现实时间 %.0f s"
                  % (client.remaining_real_s or 0), flush=True)
        world = RemoteWorld(client, belief_cls=_NullBelief)
        _load_opt1().strategy(_Opt1Ctx(world), None)
        try:
            client.exit()
        except Exception:  # noqa: BLE001
            pass
        summary = {
            "ok": True,
            "cleared_count": len(world.cleared),
            "cleared_channels": sorted(world.cleared),
            "virtual_time_s": world.virtual_t,
            "measures": world.measures,
            "clears": world.clears,
            "failed_clears": world.failed_clears,
            "moved_m": world.moved_m,
            "real_elapsed_s": (time.time() - client.entered_wall) if client.entered_wall else 0.0,
        }
    finally:
        client.close()
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
