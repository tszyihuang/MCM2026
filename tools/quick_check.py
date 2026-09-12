"""快速自检脚本: 对运行中的模拟器跑一遍附件示例指令序列。

用法:
    python tools/quick_check.py [base_url] [team_id]

退出码 0 表示示例序列全部按预期被接受。这里**不是**完整测试套件
(完整套件见 ``python -m unittest discover -s tests -t .``, 一键自检见
``python tools/smoke_test.py``), 它的价值是在现场排障时用一条命令取证。
"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:2026"
TEAM = sys.argv[2] if len(sys.argv) > 2 else "MCM2026"

FAIL = []


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(raw)
        except Exception:
            return exc.code, {"raw": raw}
    except Exception as exc:  # noqa: BLE001 - 模拟器没起时给出可读原因
        return 0, {"error": repr(exc)}


def base(rid):
    return {"arena_id": "default", "robot_id": TEAM, "request_id": rid}


def act(rid, x, y, ch):
    p = base(rid)
    p["position"] = {"x": x, "y": y}
    p["channel"] = ch
    return p


def step(label, path, payload):
    status, body = post(path, payload)
    ok = status == 200 and body.get("accepted") is True
    if not ok:
        FAIL.append(label)
    print("%-9s %s  %s" % (label, "OK  " if ok else "FAIL", json.dumps(body, ensure_ascii=False)))
    return body


def main() -> int:
    step("enter", "/enter", base("enter-1"))
    step("measure1", "/measure", act("m-1", 300, 400, 1))
    step("measure2", "/measure", act("m-2", 300, 400, 2))
    step("clear1", "/clear", act("c-1", 300, 0, 3))
    step("measure3", "/measure", act("m-3", 300, 0, 2))
    step("exit", "/exit", base("exit-1"))
    if FAIL:
        print("\n失败 %d 项: %s" % (len(FAIL), ", ".join(FAIL)))
        return 1
    print("\n示例指令序列全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
