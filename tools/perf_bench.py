"""多核性能基准: 量化模拟器/策略的 CPU 开销, 并用全部核心并行跑小批量.

三件事:

1. ``--mode micro``  —— 模拟器核心热点微基准 (动作处理 / 报告 / 加密导出 / HTTP 往返);
2. ``--mode cases``  —— 进程内跑 N 局策略对局, 统计清除率 / 平均虚拟耗时 / 墙钟;
3. ``--mode all``    —— 两者都跑, 并输出 CPU 利用率 (user+sys / wall)。

并行策略: ``--workers N`` 用 ``ProcessPoolExecutor`` 把 (问题, 种子) 分片给 N 个进程;
每个分片自带独立的 ``SimulatorEngine``, 因此结果与串行完全一致 (逐局独立、无共享状态)。
``--workers 0`` 表示按 ``os.cpu_count()`` 自动选择。

用法:
    python tools/perf_bench.py --mode micro
    python tools/perf_bench.py --mode cases --problem 3 --seeds 1001-1040 --workers 16
    python tools/perf_bench.py --mode all --seeds 1001-1008 --baseline data/perf_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------


def cpu_times() -> Tuple[float, float]:
    """返回 (user, sys) 累计 CPU 秒数 (进程级)。"""
    t = os.times()
    return t.user, t.system


def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def digest_of(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def resolve_workers(requested: int, cases: Optional[int] = None) -> int:
    """并行度: 显式值优先; 否则按批量大小挑最优 (小批量开满核反而更慢)。"""
    if requested > 0:
        return requested
    cap = os.cpu_count() or 1
    if cases is None:
        return cap
    from tools.bench_cases import suggest_workers

    return suggest_workers(cases, cap)


# --------------------------------------------------------------------------------------
# 1) 模拟器核心微基准
# --------------------------------------------------------------------------------------


def bench_engine_actions(problem: int = 3, seed: int = 4242, actions: int = 20000) -> Dict[str, Any]:
    """模拟器 handle() 路径: 去掉网络, 只测引擎校验/物理/记录/幂等。"""
    from simulator.core import SimulatorEngine

    engine = SimulatorEngine(team_id="BENCH", countdown_s=0.0)
    engine.new_run(problem, seed=seed, code="BENCH")
    run = engine.run
    now = engine.clock.now()
    run.phase = "armed"
    run.window_start = now
    run.window_deadline = now + 10_000_000.0
    engine.handle("enter", {"arena_id": "default", "robot_id": "BENCH", "request_id": "e"})
    i = 0
    t0 = time.perf_counter()
    while i < actions:
        run.stats.virtual_timeout_flagged = False
        run.virtual_us = 0
        engine.handle(
            "measure",
            {
                "arena_id": "default",
                "robot_id": "BENCH",
                "request_id": "m-%d" % i,
                "position": {"x": float(i % 900), "y": float((i * 7) % 900)},
                "channel": (i % 20) + 1,
            },
        )
        i += 1
        if i % 5000 == 0:
            run.actions.clear()
            run.idempotency.clear()
    wall = time.perf_counter() - t0
    return {
        "actions": actions,
        "wall_s": wall,
        "us_per_action": wall / actions * 1e6,
        "actions_per_s": actions / wall,
    }


def bench_report(problem: int = 3, seed: int = 4242, actions: int = 2000, repeats: int = 20) -> Dict[str, Any]:
    """build_report() + JSON 序列化 (导出日志前的最后一次全量编码)。"""
    from simulator.core import SimulatorEngine

    engine = SimulatorEngine(team_id="BENCH", countdown_s=0.0)
    engine.new_run(problem, seed=seed, code="BENCH")
    run = engine.run
    now = engine.clock.now()
    run.phase = "armed"
    run.window_start = now
    run.window_deadline = now + 10_000_000.0
    engine.handle("enter", {"arena_id": "default", "robot_id": "BENCH", "request_id": "e"})
    for i in range(actions):
        engine.handle(
            "measure",
            {
                "arena_id": "default",
                "robot_id": "BENCH",
                "request_id": "m-%d" % i,
                "position": {"x": float(i % 900), "y": float((i * 7) % 900)},
                "channel": (i % 20) + 1,
            },
        )
    t0 = time.perf_counter()
    for _ in range(repeats):
        report = engine.build_report("user_exit")
        json.dumps(report, ensure_ascii=False)
    wall = time.perf_counter() - t0
    return {
        "recorded_actions": len(run.actions),
        "repeats": repeats,
        "wall_s": wall,
        "ms_per_report": wall / repeats * 1000.0,
    }


def bench_encrypt(size_kb: int = 900, repeats: int = 5) -> Dict[str, Any]:
    """行为日志加密导出 (正式测试每局都跑一次)。"""
    from simulator.core import encrypt_log, decrypt_log

    plain = (json.dumps({"fill": "x" * 64, "i": 0}, ensure_ascii=False) + "\n").encode("utf-8")
    target = size_kb * 1024
    plain = plain * (target // len(plain) + 1)
    plain = plain[:target]
    t0 = time.perf_counter()
    for _ in range(repeats):
        blob = encrypt_log(plain)
        decrypt_log(blob)
    wall = time.perf_counter() - t0
    return {
        "size_kb": len(plain) // 1024,
        "repeats": repeats,
        "wall_s": wall,
        "ms_per_roundtrip": wall / repeats * 1000.0,
    }


def bench_http(requests: int = 3000, concurrency: int = 1) -> Dict[str, Any]:
    """HTTP 往返基线: 每个请求都新建连接 (``urllib.request.urlopen``)。

    这一项**故意**保留"每请求一条新连接"的老做法, 作为对照基线; 真实机器狗
    程序走的是 ``RobotClient`` 的 keep-alive 复用路径, 见 ``bench_client``。
    """
    import threading
    import urllib.request

    from simulator.core import SimulatorEngine
    from simulator.server import SimulatorServer

    engine = SimulatorEngine(team_id="BENCH", countdown_s=0.0)
    engine.new_run(3, seed=4242, code="BENCH")
    run = engine.run
    now = engine.clock.now()
    run.phase = "armed"
    run.window_start = now
    run.window_deadline = now + 10_000_000.0
    server = SimulatorServer(engine, port=0)
    port = server.start()
    base = "http://127.0.0.1:%d" % port
    body = json.dumps(
        {
            "arena_id": "default",
            "robot_id": "BENCH",
            "request_id": "e",
        }
    ).encode("utf-8")

    def post(path: str, blob: bytes) -> None:
        req = urllib.request.Request(
            base + path, data=blob, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()

    post("/enter", body)
    per = max(1, requests // concurrency)
    errors: List[BaseException] = []

    def worker(idx: int) -> None:
        try:
            for i in range(per):
                rid = "w%d-%d" % (idx, i)
                blob = json.dumps(
                    {
                        "arena_id": "default",
                        "robot_id": "BENCH",
                        "request_id": rid,
                        "position": {"x": float(i % 900), "y": float((i * 7) % 900)},
                        "channel": (i % 20) + 1,
                    }
                ).encode("utf-8")
                post("/measure", blob)
                if i % 500 == 0:
                    run.actions.clear()
                    run.idempotency.clear()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    total = per * concurrency
    server.stop()
    if errors:
        raise errors[0]
    return {
        "requests": total,
        "concurrency": concurrency,
        "wall_s": wall,
        "us_per_request": wall / total * 1e6,
        "requests_per_s": total / wall,
    }


def bench_client(requests: int = 3000) -> Dict[str, Any]:
    """真实路径: ``RobotClient`` 走 HTTP/1.1 keep-alive 连接复用。

    这是机器人程序实际使用的传输实现, 也是本项优化的主战场 ——
    原实现每个动作新建 TCP 连接, 本机约 4.6 ms/次, 其中 3.2 ms 是
    ``socket.connect``; 复用连接后降到约 0.37 ms/次。
    """
    from robotdog import RobotClient
    from simulator.core import SimulatorEngine
    from simulator.server import SimulatorServer

    engine = SimulatorEngine(team_id="BENCH", countdown_s=0.0)
    engine.new_run(3, seed=4242, code="BENCH")
    run = engine.run
    now = engine.clock.now()
    run.phase = "armed"
    run.window_start = now
    run.window_deadline = now + 10_000_000.0
    server = SimulatorServer(engine, port=0)
    port = server.start()
    client = RobotClient(base_url="http://127.0.0.1:%d" % port, team_id="BENCH")
    try:
        client.enter()
        client.measure(0.0, 0.0, 1)  # 预热: 建连
        t0 = time.perf_counter()
        for i in range(requests):
            client.measure(float(i % 900), float((i * 7) % 900), (i % 20) + 1)
            if i % 500 == 0:
                run.actions.clear()
                run.idempotency.clear()
        wall = time.perf_counter() - t0
    finally:
        client.close()
        server.stop()
    return {
        "requests": requests,
        "wall_s": wall,
        "us_per_request": wall / requests * 1e6,
        "requests_per_s": requests / wall,
    }


def run_micro(quick: bool) -> Dict[str, Any]:
    scale = 0.25 if quick else 1.0
    out: Dict[str, Any] = {}
    out["engine_actions"] = bench_engine_actions(actions=max(2000, int(20000 * scale)))
    out["report"] = bench_report(actions=max(200, int(2000 * scale)), repeats=20)
    out["encrypt"] = bench_encrypt(size_kb=int(900 * scale) or 100, repeats=5)
    out["client"] = bench_client(requests=max(500, int(4000 * scale)))
    out["http_1"] = bench_http(requests=max(300, int(3000 * scale)), concurrency=1)
    out["http_8"] = bench_http(requests=max(600, int(6000 * scale)), concurrency=8)
    return out


# --------------------------------------------------------------------------------------
# 2) 进程内对局基准 (可多核分片)
# --------------------------------------------------------------------------------------


def _case_worker(job: Tuple[int, int, Dict[str, Any]]) -> Dict[str, Any]:
    """子进程入口: 跑一局策略, 返回统计 (必须可 pickle)。

    注意: ``os.times()`` 只统计**本进程**的 CPU 时间, 父进程拿不到子进程的用量,
    因此这里让每个子进程自己报告, 由父进程汇总, 否则多核利用率会被严重低估。
    """
    problem, seed, params = job
    from tools.tune import run_case

    c0 = cpu_times()
    t0 = time.perf_counter()
    row = run_case(problem, seed, params)
    row["cpu_s"] = (cpu_times()[0] + cpu_times()[1]) - (c0[0] + c0[1])
    row["wall_s"] = time.perf_counter() - t0
    row.pop("params", None)
    return row


def run_cases(
    problem: int, seeds: List[int], workers: int, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    params = params or {}
    jobs = [(problem, s, params) for s in seeds]
    t0 = time.perf_counter()
    if workers <= 1:
        rows = [_case_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_case_worker, jobs, chunksize=1))
    wall = time.perf_counter() - t0
    cpu = sum(r.pop("cpu_s", 0.0) for r in rows)
    inner_wall = sum(r.pop("wall_s", 0.0) for r in rows)
    cleared = sum(r["cleared"] for r in rows)
    total = sum(r["total"] for r in rows)
    return {
        "problem": problem,
        "cases": len(rows),
        "workers": workers,
        "wall_s": round(wall, 4),
        "cpu_s": round(cpu, 4),
        "cpu_utilization": round(cpu / wall, 3) if wall > 0 else 0.0,
        # 并行加速比: 各子进程内部耗时之和 / 总墙钟 (理想值 = workers)
        "parallel_speedup": round(inner_wall / wall, 3) if wall > 0 else 0.0,
        "cases_per_s": round(len(rows) / wall, 2) if wall > 0 else 0.0,
        "cpu_ms_per_case": round(cpu / len(rows) * 1000.0, 2) if rows else 0.0,
        "cleared_ratio": round(cleared / total, 6) if total else 0.0,
        "cleared_sum": cleared,
        "source_sum": total,
        "mean_virtual_s": round(statistics.fmean(r["virtual_s"] for r in rows), 3) if rows else 0.0,
        "mean_measures": round(statistics.fmean(r["measures"] for r in rows), 2) if rows else 0.0,
        "rows": rows,
    }


# --------------------------------------------------------------------------------------
# 基线对比
# --------------------------------------------------------------------------------------


def load_baseline(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def engine_invariant(problem: int = 3, seed: int = 4242, steps: int = 400) -> Dict[str, Any]:
    """模拟器的"行为指纹": 固定种子 + 固定动作序列下的全部响应与报告。

    注意这里只记录**虚拟时间驱动**的量 (虚拟时间、响应字段、物理结果、报告),
    不记录任何墙钟量; 因此它是判定"优化是否改变了行为"的可靠判据。
    与之相对, 策略最终清除了几个干扰源与墙钟预算耦合, 本身逐次不同, 不能用作判据。
    """
    from simulator.core import ProtocolError, SimulatorEngine

    engine = SimulatorEngine(team_id="BENCH", countdown_s=0.0)
    engine.new_run(problem, seed=seed, code="INVARIANT")
    run = engine.run
    now = engine.clock.now()
    run.phase = "armed"
    run.window_start = now
    run.window_deadline = now + 10_000_000.0

    def call(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = dict(engine.handle(action, payload))
        except ProtocolError as exc:
            # 协议错误 (400/409/...) 也属于行为的一部分: 记下状态码与错误码
            return {"http_status": exc.status, "error": exc.code}
        resp.pop("real_timestamp_ms", None)  # 墙钟字段, 与本判据无关
        return resp

    out: List[Any] = []
    out.append(call("enter", {"arena_id": "default", "robot_id": "BENCH", "request_id": "e"}))
    for i in range(steps):
        x = float((i * 37) % 1700) - 850.0
        y = float((i * 91) % 1700) - 850.0
        ch = (i % 20) + 1
        action = "clear" if i % 11 == 0 else "measure"
        out.append(
            call(
                action,
                {
                    "arena_id": "default",
                    "robot_id": "BENCH",
                    "request_id": "%s-%d" % (action, i),
                    "position": {"x": x, "y": y},
                    "channel": ch,
                },
            )
        )
    # 幂等重放 / 冲突 / 业务拒绝 / 协议错误 各验一次
    out.append(call("measure", {
        "arena_id": "default", "robot_id": "BENCH", "request_id": "measure-10",
        "position": {"x": float((10 * 37) % 1700) - 850.0, "y": float((10 * 91) % 1700) - 850.0},
        "channel": (10 % 20) + 1,
    }))
    out.append(call("measure", {
        "arena_id": "default", "robot_id": "BENCH", "request_id": "measure-10",
        "position": {"x": 1.0, "y": 2.0}, "channel": 5,
    }))
    out.append(call("measure", {
        "arena_id": "default", "robot_id": "OTHER", "request_id": "zz",
        "position": {"x": 1.0, "y": 2.0}, "channel": 5,
    }))
    out.append(call("measure", {"arena_id": "default", "robot_id": "BENCH", "request_id": "bad"}))
    report = engine.build_report("user_exit")
    return {
        "responses": out,
        "report": {
            "stats": report["stats"],
            "truth": report["truth"],
            "uncleared": report["uncleared_sources"],
            "actions": report["actions"],
            "rejected": report["rejected"],
        },
    }


def build_signature(micro: Dict[str, Any], cases: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """行为指纹: 只取与性能、与墙钟都无关的确定性输出。

    ``cases`` 里的清除数逐次不同 (策略预算与墙钟耦合), 因此只统计分布,
    不参与等价性判据 —— 判据由 ``engine_invariant()`` 承担。
    """
    sig: Dict[str, Any] = {"micro_keys": sorted(micro)}
    if cases is not None:
        sig["cases_summary"] = {
            "n": cases["cases"],
            "cleared_ratio": cases["cleared_ratio"],
            "mean_virtual_s": cases["mean_virtual_s"],
        }
    return sig


def compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """生成人类可读的对比行。"""
    lines: List[str] = []
    b_micro = baseline.get("micro") or {}
    c_micro = current.get("micro") or {}
    for key in sorted(c_micro):
        b = b_micro.get(key)
        c = c_micro[key]
        if not isinstance(c, dict) or b is None:
            continue
        metric = None
        for cand in ("us_per_action", "ms_per_report", "ms_per_roundtrip", "us_per_request"):
            if cand in c:
                metric = cand
                break
        if metric is None:
            continue
        bv, cv = b.get(metric), c.get(metric)
        if not bv:
            continue
        lines.append(
            "  %-16s %-18s %10.2f -> %10.2f  x%.2f"
            % (key, metric, bv, cv, bv / cv if cv else float("inf"))
        )
    b_cases = baseline.get("cases") or {}
    c_cases = current.get("cases") or {}
    if b_cases and c_cases:
        lines.append(
            "  %-16s %-18s %10.4f -> %10.4f  x%.2f"
            % ("cases(wall)", "wall_s", b_cases.get("wall_s", 0.0), c_cases.get("wall_s", 0.0),
               (b_cases.get("wall_s") or 1) / (c_cases.get("wall_s") or 1))
        )
    return lines


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="模拟器多核性能基准")
    ap.add_argument("--mode", choices=["micro", "cases", "all"], default="all")
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--seeds", default="1001-1008")
    ap.add_argument("--workers", type=int, default=0, help="0 = 自动 (os.cpu_count())")
    ap.add_argument("--quick", action="store_true", help="微基准规模减到 1/4")
    ap.add_argument("--baseline", default=None, help="对比基线 JSON (由 --save 生成)")
    ap.add_argument("--save", default=None, help="把本次结果写入 JSON")
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过模拟器行为指纹校验 (默认每次都算, 用于证明优化不改行为)")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    workers = resolve_workers(args.workers, len(seeds))
    result: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "problem": args.problem,
        "seeds": seeds,
        "workers": workers,
    }

    if args.mode in ("micro", "all"):
        t0 = time.perf_counter()
        result["micro"] = run_micro(args.quick)
        result["micro_wall_s"] = round(time.perf_counter() - t0, 4)
    if args.mode in ("cases", "all"):
        result["cases"] = run_cases(args.problem, seeds, workers)
    if not args.no_verify:
        # 行为指纹: 固定种子 + 固定动作序列下的引擎响应与报告全文
        inv = engine_invariant(args.problem)
        result["invariant_digest"] = digest_of(inv)
        result["invariant_len"] = len(inv["responses"]) + len(inv["report"]["actions"])

    result["signature"] = build_signature(result.get("micro") or {}, result.get("cases"))
    result["signature_digest"] = digest_of(result["signature"])

    baseline = load_baseline(args.baseline) if args.baseline else None
    if baseline:
        result["baseline_signature_digest"] = baseline.get("signature_digest")
        result["signature_match"] = baseline.get("signature_digest") == result["signature_digest"]
        if "invariant_digest" in result and "invariant_digest" in baseline:
            result["invariant_match"] = baseline["invariant_digest"] == result["invariant_digest"]
            result["baseline_invariant_digest"] = baseline["invariant_digest"]
        result["speedup"] = compare(baseline, result)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print("=" * 78)
    print("模拟器性能基准  python %s  cpu=%s  workers=%d  problem=%d  cases=%d"
          % (result["python"], result["cpu_count"], workers, args.problem, len(seeds)))
    print("=" * 78)
    if "micro" in result:
        m = result["micro"]
        print("[微基准]  (越小越快)")
        e = m["engine_actions"]
        print("  引擎 handle(measure)   %8.2f us/次   %10.0f 次/秒  (%d 次)"
              % (e["us_per_action"], e["actions_per_s"], e["actions"]))
        r = m["report"]
        print("  build_report+JSON      %8.2f ms/次   (%d 条动作记录)"
              % (r["ms_per_report"], r["recorded_actions"]))
        c = m["encrypt"]
        print("  日志加解密往返         %8.2f ms/次   (%d KB)"
              % (c["ms_per_roundtrip"], c["size_kb"]))
        cl = m.get("client")
        if cl:
            print("  HTTP 复用连接 (实际)   %8.2f us/次   %10.0f 次/秒"
                  % (cl["us_per_request"], cl["requests_per_s"]))
        print("  HTTP 每请求新建连接    %8.2f us/次   %10.0f 次/秒  [对照基线]"
              % (m["http_1"]["us_per_request"], m["http_1"]["requests_per_s"]))
        print("  HTTP 8 并发新连接      %8.2f us/次   %10.0f 次/秒  [对照基线]"
              % (m["http_8"]["us_per_request"], m["http_8"]["requests_per_s"]))
    if "cases" in result:
        c = result["cases"]
        print("[对局]")
        print("  %d 局 / %d 进程        墙钟 %.3f s    CPU %.3f s    CPU/局 %.1f ms"
              % (c["cases"], c["workers"], c["wall_s"], c["cpu_s"], c["cpu_ms_per_case"]))
        print("  并行加速比 %.2fx (理想 %d)  吞吐 %.2f 局/秒"
              % (c["parallel_speedup"], c["workers"], c["cases_per_s"]))
        print("  清除 %d/%d (%.1f%%)   平均虚拟 %.1f s   平均检测 %.1f 次"
              % (c["cleared_sum"], c["source_sum"], 100.0 * c["cleared_ratio"],
                 c["mean_virtual_s"], c["mean_measures"]))
    if "invariant_digest" in result:
        print("[行为指纹] 模拟器 %s (%d 条响应/动作)"
              % (result["invariant_digest"], result["invariant_len"]))
    if baseline:
        if "invariant_match" in result:
            print("[对比基线] 行为一致 = %s%s"
                  % (result["invariant_match"],
                     "" if result["invariant_match"] else
                     "  (基线 %s)" % result.get("baseline_invariant_digest")))
        for line in result.get("speedup") or []:
            print(line)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
