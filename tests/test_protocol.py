"""通信协议一致性测试: 严格对照附件2 表2/表3~表11 与第5~9节。

覆盖 HTTP 状态码 200/400/404/405/409/413/415、accepted 语义、
request_id 幂等、字段校验、未知字段、坐标与频道边界、BOM/重复键/嵌套深度等。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.harness import SimHarness, TEAM, src  # noqa: E402


class ProtocolBase(unittest.TestCase):
    def setUp(self):
        self.h = SimHarness()
        self.h.start_run([src(1, 0.0, 0.0, 1500.0), src(2, 500.0, 0.0, 1500.0)])

    def tearDown(self):
        self.h.close()

    def post_raw(self, path, body, headers=None, method="POST"):
        return self.h.raw(method, path, body, headers)


class EnvelopeTest(ProtocolBase):
    def test_enter_before_countdown_over(self):
        h = SimHarness(countdown_s=5.0)
        run = h.engine.new_run(3, seed=1, code="CD")
        run.case.sources = [src(1, 0.0, 0.0, 1500.0)]
        h.engine.arm()
        status, body = h.post("/enter", h.base())
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "not_open")
        self.assertEqual(body["virtual_time_s"], 0)
        self.assertEqual(
            set(body) - {"accepted", "real_timestamp_ms", "virtual_time_s"},
            {"reject_code", "reject_message"},
        )
        # 倒计时结束后接口开放
        h.fake_clock.advance(5.01)
        h.engine.tick()
        self.assertTrue(h.enter()["accepted"])
        h.close()

    def test_action_before_enter_rejected(self):
        status, body, _ = self.post_raw("/measure", json.dumps(self.h.action(0, 0, 1)).encode())
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "not_entered")

    def test_exit_before_enter_rejected(self):
        status, body, _ = self.post_raw("/exit", json.dumps(self.h.base()).encode())
        self.assertFalse(body["accepted"])

    def test_enter_response_fields(self):
        body = self.h.enter()
        self.assertTrue(body["accepted"])
        self.assertEqual(body["virtual_time_s"], 0)  # /enter 不推进虚拟时钟
        for key in ("max_virtual_duration_s", "max_real_duration_s", "remaining_real_duration_s"):
            self.assertIn(key, body)
        self.assertGreaterEqual(body["remaining_real_duration_s"], 0)
        self.assertLessEqual(body["remaining_real_duration_s"], 1200)

    def test_duplicate_enter_rejected(self):
        self.h.enter()
        # 第二次 /enter 使用新 request_id -> 业务拒绝
        status, body, _ = self.post_raw("/enter", json.dumps(self.h.base()).encode())
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "duplicate_enter")

    def test_exit_then_actions_rejected(self):
        self.h.enter()
        self.h.exit()
        status, body, _ = self.post_raw("/measure", json.dumps(self.h.action(0, 0, 1)).encode())
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "test_ended")

    def test_exit_reason(self):
        self.h.enter()
        body = self.h.exit()
        self.assertEqual(body["exit_reason"], "user_exit")

    def test_all_responses_have_three_fields(self):
        self.h.enter()
        for body in (
            self.h.measure(0, 0, 1),
            self.h.clear(0, 0, 1),
        ):
            for key in ("accepted", "real_timestamp_ms", "virtual_time_s"):
                self.assertIn(key, body)


class HttpStatusTest(ProtocolBase):
    def test_unknown_path_404(self):
        payload = json.dumps(self.h.base()).encode()
        for path in ("/", "/foo", "/enter/", "/measure?x=1", "/ENTER", "/enter%20", "/enter/x", "/enter//"):
            status, body, _ = self.post_raw(path, payload)
            self.assertEqual(status, 404, "路径 %r 应返回 404" % path)

    def test_wrong_method_405(self):
        for method in ("GET", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
            status, body, _ = self.post_raw("/enter", None, method=method)
            self.assertEqual(status, 405, "%s /enter 应返回 405" % method)

    def test_wrong_method_unknown_path_404(self):
        for method in ("HEAD", "GET", "PUT"):
            status, body, _ = self.post_raw("/foo", None, method=method)
            self.assertEqual(status, 404, "%s /foo 应返回 404" % method)

    def test_head_has_no_body(self):
        status, raw, headers = self.h.raw("HEAD", "/enter")
        self.assertEqual(status, 405)
        self.assertIn(raw, (b"", None), "HEAD 响应不得带响应体")

    def test_content_type_415(self):
        payload = json.dumps(self.h.base()).encode()
        for ct in (
            "text/plain",
            "application/xml",
            "application/json; charset=gbk",
            "application/json; charset=utf-8; foo=bar",
            "application/json;foo=bar",
            "",
        ):
            status, body, _ = self.post_raw("/enter", payload, headers={"Content-Type": ct})
            self.assertEqual(status, 415, "Content-Type %r 应返回 415" % ct)

    def test_content_type_accepted_variants(self):
        payload = json.dumps(self.h.base()).encode()
        for ct in ("application/json", "application/json; charset=utf-8", "application/json;charset=UTF-8"):
            status, body, _ = self.post_raw("/enter", payload, headers={"Content-Type": ct, "Connection": "close"})
            self.assertEqual(status, 200, "Content-Type %r 应被接受" % ct)
            self.assertTrue(body["accepted"], ct)

    def test_content_encoding_415(self):
        payload = json.dumps(self.h.base()).encode()
        status, body, _ = self.post_raw(
            "/enter", payload, headers={"Content-Type": "application/json", "Content-Encoding": "gzip"}
        )
        self.assertEqual(status, 415)
        self.assertFalse(body["accepted"])
        status, body, _ = self.post_raw(
            "/enter", payload, headers={"Content-Type": "application/json", "Content-Encoding": "identity"}
        )
        self.assertEqual(status, 200)

    def test_body_too_large_413(self):
        big = {"arena_id": "default", "robot_id": TEAM, "request_id": "x" * 100, "pad": "y" * 70000}
        status, body, _ = self.post_raw("/enter", json.dumps(big).encode())
        self.assertEqual(status, 413)
        self.assertFalse(body["accepted"])

    def test_http_error_body_virtual_time_is_zero(self):
        """附件2 4.1: accepted=false 时 virtual_time_s 为 0 (HTTP 错误体同样适用)。"""
        self.h.enter()
        self.h.measure(300, 0, 1)  # 推进虚拟时间
        self.assertGreater(self.h.engine.run.virtual_seconds(), 0)
        status, body, _ = self.post_raw(
            "/measure", json.dumps(self.h.action(0, 0, 99)).encode()
        )
        self.assertEqual(status, 400)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["virtual_time_s"], 0)
        status, body, _ = self.post_raw("/measure", b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["virtual_time_s"], 0)
        status, body, _ = self.post_raw("/foo", b"{}")
        self.assertEqual(status, 404)
        self.assertEqual(body["virtual_time_s"], 0)


class ConcurrencyTest(ProtocolBase):
    """附件2 表2: '并发发送了不同动作' 必须返回 409。"""

    def setUp(self):
        super().setUp()
        self.h.enter()
        # 人为延长动作处理时间, 使并发冲突可稳定复现
        self.h.engine.gate_hold_s = 1.5

    def tearDown(self):
        self.h.engine.gate_hold_s = 0.0
        super().tearDown()

    def _fire(self, payloads, timeout: float = 30.0):
        """并行发送多个动作, 用栅栏保证它们同时进入模拟器。"""
        import threading

        results = [None] * len(payloads)
        barrier = threading.Barrier(len(payloads))

        def worker(idx, payload):
            barrier.wait(timeout=10)
            try:
                results[idx] = self.h.post("/measure", payload)
            except Exception as exc:  # noqa: BLE001
                results[idx] = (0, {"accepted": False, "error": str(exc)})

        threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(payloads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)
        return results

    def test_parallel_different_actions_conflict_409(self):
        """引擎层: 并发发送不同动作必须被拒绝为 409, 且不推进虚拟时间。"""
        import threading

        payloads = [self.h.action(100.0 * i, 0.0, 1) for i in range(5)]
        responses = [None] * len(payloads)
        errors = [0] * len(payloads)
        barrier = threading.Barrier(len(payloads))

        def worker(idx):
            barrier.wait(timeout=10)
            try:
                responses[idx] = self.h.engine.handle("measure", payloads[idx])
            except Exception:  # noqa: BLE001
                errors[idx] = 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(payloads))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        accepted = sum(1 for r in responses if r and r.get("accepted") is True)
        conflicts = sum(
            1 for r in responses if r and r.get("accepted") is False and r.get("error") == "concurrent_request"
        )
        self.assertEqual(accepted, 1, "同一时刻只允许一个动作执行: %s" % responses)
        self.assertEqual(conflicts, 4, "其余并发动作必须返回并发冲突: %s" % responses)
        for r in responses:
            if r and r.get("accepted") is False:
                self.assertEqual(r["virtual_time_s"], 0)
        # 只有被接受的那次动作推进虚拟时间: 从原点到被接受请求的坐标
        accepted_pos = max(100.0 * i for i in range(5))
        self.assertAlmostEqual(
            self.h.engine.run.virtual_seconds(), accepted_pos / 5.0 + 5.0, places=6
        )
        self.assertEqual(self.h.engine.run.stats.measures, 1)

    def test_parallel_different_actions_conflict_via_http(self):
        """HTTP 层: 并发发送不同动作时, 引擎内同时只有一个动作在执行。"""
        payloads = [self.h.action(100.0 * i, 0.0, 1) for i in range(5)]
        results = self._fire(payloads)
        codes = [status for status, _ in results]
        for status, body in results:
            if status == 409:
                self.assertFalse(body["accepted"])
                self.assertEqual(body["virtual_time_s"], 0)
            else:
                self.assertEqual(status, 200)
                self.assertTrue(body["accepted"])
        stats = self.h.server.concurrency_stats()
        self.assertEqual(
            stats["engine_max_concurrent"], 1, "引擎内不得出现两个动作同时执行: %s" % stats
        )
        accepted = sum(1 for c in codes if c == 200)
        if stats["max_concurrent"] > 1 and 409 not in codes:
            self.fail("检测到请求重叠却没有返回 409: %s" % stats)
        self.assertGreaterEqual(accepted, 1)
        # 被接受动作推进的虚拟时间合计必须与动作日志一致
        total = sum(a["total_duration_s"] for a in self.h.engine.run.actions)
        self.assertAlmostEqual(total, self.h.engine.run.virtual_seconds(), places=6)

    def test_serial_requests_never_overlap(self):
        self.h.engine.gate_hold_s = 0.02
        for i in range(6):
            status, body = self.h.post("/measure", self.h.action(50.0 * i, 0.0, 1))
            self.assertEqual(status, 200)
        stats = self.h.server.concurrency_stats()
        self.assertEqual(stats["engine_max_concurrent"], 1, "串行请求不得出现重叠: %s" % stats)
        self.assertEqual(stats["max_concurrent"], 1, "串行请求的 HTTP 层也不得重叠: %s" % stats)

    def test_parallel_same_retry_replays_once(self):
        """同一动作的网络重试必须复用同一 request_id, 且只执行一次。"""
        import threading

        payload = self.h.action(400.0, 0.0, 1, request_id="retry-concurrent")
        results = [None] * 6
        barrier = threading.Barrier(6)

        def worker(idx):
            barrier.wait(timeout=10)
            results[idx] = self.h.post("/measure", payload)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        codes = [status for status, _ in results]
        self.assertTrue(all(c == 200 for c in codes), "同内容重试不应返回 409: %s" % codes)
        times = {body["virtual_time_s"] for _, body in results}
        self.assertEqual(len(times), 1, "重试必须返回首次响应: %s" % times)
        self.assertAlmostEqual(self.h.engine.run.stats.measures, 1, places=6)

    def test_sequential_requests_unaffected(self):
        self.h.engine.gate_hold_s = 0.0
        for i in range(4):
            status, body = self.h.post("/measure", self.h.action(10.0 * i, 0.0, 1))
            self.assertEqual(status, 200)
            self.assertTrue(body["accepted"])


class RejectFieldSetTest(ProtocolBase):
    """附件2 §5.2: accepted=false 的业务响应只包含 3 个字段 (默认严格模式)。"""

    def setUp(self):
        super().setUp()
        self.h.engine.verbose_reject = False

    def _assert_three_fields(self, body):
        self.assertFalse(body["accepted"])
        self.assertEqual(
            set(body),
            {"accepted", "real_timestamp_ms", "virtual_time_s"},
            "业务拒绝响应只应包含 3 个标准字段: %s" % sorted(body),
        )
        self.assertEqual(body["virtual_time_s"], 0)

    def test_reject_before_enter(self):
        status, body, _ = self.post_raw("/measure", json.dumps(self.h.action(0, 0, 1)).encode())
        self.assertEqual(status, 200)
        self._assert_three_fields(body)

    def test_reject_unknown_field(self):
        payload = self.h.action(0, 0, 1)
        payload["oops"] = 1
        status, body, _ = self.post_raw("/measure", json.dumps(payload).encode())
        self.assertEqual(status, 200)
        self._assert_three_fields(body)

    def test_reject_arena_robot_mismatch(self):
        for override in ({"arena_id": "other"}, {"robot_id": "other-team"}):
            payload = self.h.action(0, 0, 1)
            payload.update(override)
            status, body, _ = self.post_raw("/measure", json.dumps(payload).encode())
            self._assert_three_fields(body)

    def test_reject_duplicate_enter_and_after_exit(self):
        self.h.enter()
        status, body, _ = self.post_raw("/enter", json.dumps(self.h.base()).encode())
        self._assert_three_fields(body)
        self.h.exit()
        status, body, _ = self.post_raw("/measure", json.dumps(self.h.action(0, 0, 1)).encode())
        self._assert_three_fields(body)

    def test_standard_response_field_sets(self):
        """附件2 表4/表6/表8: 成功响应的字段集必须精确匹配。"""
        self.h.engine.verbose_reject = True
        body = self.h.enter()
        self.assertEqual(
            set(body),
            {
                "accepted",
                "real_timestamp_ms",
                "virtual_time_s",
                "max_virtual_duration_s",
                "max_real_duration_s",
                "remaining_real_duration_s",
            },
            sorted(body),
        )
        sources = self.h.engine.run.case.sources
        plain = [s for s in sources if s.is_omni][0]
        for result, expected_extra in (
            ("no_signal", set()),
            ("near", set()),
            ("direction", {"svd_deg"}),
        ):
            if result == "near":
                r = self.h.measure(plain.x + 1.0, plain.y, plain.channel)
            elif result == "direction":
                r = self.h.measure(plain.x + plain.radius_m * 0.5, plain.y, plain.channel)
            else:
                r = self.h.measure(0.0, 0.0, 20)
            self.assertEqual(r["measure_result"], result, r)
            self.assertEqual(
                set(r),
                {"accepted", "real_timestamp_ms", "virtual_time_s", "measure_result"} | expected_extra,
                "measure 响应字段集不符: %s" % sorted(r),
            )
        c = self.h.clear(plain.x, plain.y, plain.channel)
        self.assertEqual(
            set(c),
            {"accepted", "real_timestamp_ms", "virtual_time_s", "clear_result"},
            sorted(c),
        )
        e = self.h.exit()
        self.assertEqual(
            set(e),
            {"accepted", "real_timestamp_ms", "virtual_time_s", "exit_reason"},
            sorted(e),
        )

    def test_verbose_reject_adds_debug_fields(self):
        self.h.engine.verbose_reject = True
        status, body, _ = self.post_raw("/measure", json.dumps(self.h.action(0, 0, 1)).encode())
        self.assertFalse(body["accepted"])
        self.assertIn("reject_code", body)
        self.assertEqual(body["reject_code"], "not_entered")


class BodyFormatTest(ProtocolBase):
    def test_malformed_json_400(self):
        for raw in (b"{", b"", b"[]", b"null", b"123", b'"str"', b"{'a':1}"):
            status, body, _ = self.post_raw("/enter", raw)
            self.assertEqual(status, 400, "非法请求体 %r 应返回 400" % raw)

    def test_duplicate_key_400(self):
        raw = b'{"arena_id":"default","arena_id":"default","robot_id":"%s","request_id":"r"}' % TEAM.encode()
        status, body, _ = self.post_raw("/enter", raw)
        self.assertEqual(status, 400)

    def test_nan_infinity_400(self):
        for token in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
            raw = b'{"arena_id":"default","robot_id":"%s","request_id":"r","position":{"x":%s,"y":0},"channel":1}' % (
                TEAM.encode(),
                token,
            )
            status, body, _ = self.post_raw("/measure", raw)
            self.assertEqual(status, 400, "%s 应返回 400" % token)

    def test_bom_rejected(self):
        raw = b"\xef\xbb\xbf" + json.dumps(self.h.base()).encode()
        status, body, _ = self.post_raw("/enter", raw)
        self.assertEqual(status, 400)

    def test_deep_nesting_400(self):
        nested = "1"
        for _ in range(20):
            nested = '{"a":%s}' % nested
        raw = ('{"arena_id":"default","robot_id":"%s","request_id":"r","position":%s,"channel":1}' % (TEAM, nested)).encode()
        status, body, _ = self.post_raw("/measure", raw)
        self.assertEqual(status, 400)

    def test_measure_list_body_400(self):
        status, body, _ = self.post_raw("/measure", b"[1,2,3]")
        self.assertEqual(status, 400)


class FieldValidationTest(ProtocolBase):
    def setUp(self):
        super().setUp()
        self.h.enter()

    def _measure(self, **overrides):
        payload = self.h.action(0, 0, 1)
        payload.update(overrides)
        return self.post_raw("/measure", json.dumps(payload).encode())

    def test_missing_required_fields_400(self):
        for missing in ("arena_id", "robot_id", "request_id", "position", "channel"):
            payload = self.h.action(0, 0, 1)
            payload.pop(missing)
            status, body, _ = self.post_raw("/measure", json.dumps(payload).encode())
            self.assertEqual(status, 400, "缺少 %s 应返回 400" % missing)

    def test_missing_position_component_400(self):
        for missing in ("x", "y"):
            payload = self.h.action(0, 0, 1)
            payload["position"].pop(missing)
            status, body, _ = self.post_raw("/measure", json.dumps(payload).encode())
            self.assertEqual(status, 400)

    def test_arena_id_mismatch(self):
        status, body, _ = self._measure(arena_id="other")
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "arena_id_mismatch")

    def test_robot_id_mismatch(self):
        status, body, _ = self._measure(robot_id="SOMEONE-ELSE")
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "robot_id_mismatch")

    def test_unknown_top_level_field(self):
        status, body, _ = self._measure(channnel=1)
        self.assertEqual(status, 200)
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "unknown_field")

    def test_unknown_position_field(self):
        payload = self.h.action(0, 0, 1)
        payload["position"]["z"] = 0
        status, body, _ = self.post_raw("/measure", json.dumps(payload).encode())
        self.assertFalse(body["accepted"])
        self.assertEqual(body["reject_code"], "unknown_field")

    def test_coordinate_bounds(self):
        for value in (2000000, -2000000, 0, 1999999.999, -1999999.5):
            status, body, _ = self._measure(position={"x": value, "y": 0})
            # 边界内坐标必须通过参数校验 (HTTP 200); 超长移动可能因虚拟时间上限被拒绝
            self.assertEqual(status, 200, "坐标 %s 应通过参数校验" % value)
            self.assertNotEqual(body.get("reject_code"), "invalid_position")
        for value in (2000000.001, -2000000.5, 1e12):
            status, body, _ = self._measure(position={"x": value, "y": 0})
            self.assertEqual(status, 400, "坐标 %s 应返回 400" % value)

    def test_coordinate_type(self):
        for value in ("100", None, True, [1], {"a": 1}):
            status, body, _ = self._measure(position={"x": value, "y": 0})
            self.assertEqual(status, 400, "坐标类型 %r 应返回 400" % (value,))

    def test_channel_bounds(self):
        for ch in (1, 20, 20.0, 1.0):
            status, body, _ = self._measure(channel=ch)
            self.assertEqual(status, 200)
            self.assertTrue(body["accepted"], "频道 %r 应被接受" % ch)
        for ch in (0, 21, -1, 1.5, "1", None, True, 1e9):
            status, body, _ = self._measure(channel=ch)
            self.assertEqual(status, 400, "频道 %r 应返回 400" % (ch,))

    def test_request_id_format(self):
        for rid in ("", "x" * 129, "bad\nid", "bad\x00id", "bad\u200bid"):
            status, body, _ = self._measure(request_id=rid)
            self.assertEqual(status, 400, "request_id %r 应返回 400" % rid)
        ok = "x" * 128
        status, body, _ = self._measure(request_id=ok)
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])

    def test_robot_id_format(self):
        for rid in ("", "x" * 65, "bad\x01id"):
            status, body, _ = self._measure(robot_id=rid)
            self.assertEqual(status, 400, "robot_id %r 应返回 400" % rid)


class IdempotencyTest(ProtocolBase):
    def setUp(self):
        super().setUp()
        self.h.enter()

    def test_replay_same_id_same_content(self):
        rid = "replay-1"
        self.h.measure(100.0, 0.0, 1)
        base = self.h.engine.run.virtual_seconds()
        first = self.h.measure(100.0, 0.0, 1, request_id=rid)
        second = self.h.measure(100.0, 0.0, 1, request_id=rid)
        self.assertEqual(first["virtual_time_s"], second["virtual_time_s"])
        self.assertEqual(first["measure_result"], second["measure_result"])
        self.assertAlmostEqual(first["virtual_time_s"] - base, 5.0, places=6)  # 原地检测 5 s
        # 重放不重复推进虚拟时间: 从 (100,0) 到 (500,0) 为 400 m = 80 s, 检测 5 s, 频道不变
        base2 = self.h.engine.run.virtual_seconds()
        third = self.h.measure(500.0, 0.0, 1)
        self.assertAlmostEqual(third["virtual_time_s"] - base2, 85.0, places=6)

    def test_same_id_different_content_409(self):
        rid = "conflict-1"
        self.h.measure(100.0, 0.0, 1, request_id=rid)
        status, body = self.h.measure_status(200.0, 0.0, 1, request_id=rid)
        self.assertEqual(status, 409)
        self.assertFalse(body["accepted"])

    def test_same_id_different_path_409(self):
        rid = "conflict-2"
        self.h.measure(100.0, 0.0, 1, request_id=rid)
        status, body = self.h.clear_status(100.0, 0.0, 1, request_id=rid)
        self.assertEqual(status, 409)

    def test_rejected_request_does_not_consume_id(self):
        rid = "reuse-1"
        payload = self.h.action(0, 0, 1, request_id=rid)
        payload["zzz"] = 1
        status, body = self.h.post("/measure", payload)
        self.assertFalse(body["accepted"])
        # 同一 request_id 修正后应可复用 (未占用)
        status, body = self.h.measure_status(10.0, 0.0, 1, request_id=rid)
        self.assertEqual(status, 200)
        self.assertTrue(body["accepted"])

    def test_accepted_request_consumes_id(self):
        rid = "used-1"
        self.h.measure(10.0, 0.0, 1, request_id=rid)
        status, body = self.h.measure_status(20.0, 0.0, 1, request_id=rid)
        self.assertEqual(status, 409)


class BudgetTest(ProtocolBase):
    def test_program_timeout_ends_test(self):
        h = SimHarness(max_real_s=10.0)
        h.start_run([src(1, 0.0, 0.0, 1500.0)])
        enter = h.enter()
        self.assertEqual(enter["remaining_real_duration_s"], 10)
        h.measure(0.0, 0.0, 1)
        h.fake_clock.advance(11.0)
        status, body = h.post("/measure", h.action(0.0, 0.0, 1))
        self.assertFalse(body["accepted"])
        self.assertEqual(h.engine.run.stats.end_reason, "program_timeout")
        h.close()

    def test_window_timeout_ends_test(self):
        h = SimHarness(window_s=30.0, max_real_s=1200.0)
        h.start_run([src(1, 0.0, 0.0, 1500.0)])
        enter = h.enter()
        self.assertEqual(enter["remaining_real_duration_s"], 30)
        h.fake_clock.advance(31.0)
        h.server.tick()
        self.assertEqual(h.engine.run.stats.end_reason, "window_timeout")
        h.close()

    def test_remaining_real_duration_shrinks_late_enter(self):
        h = SimHarness(window_s=8 * 60.0, max_real_s=1200.0)
        h.start_run([src(1, 0.0, 0.0, 1500.0)])
        h.fake_clock.advance(120.0)  # 晚 2 分钟调用 /enter
        enter = h.enter()
        # 窗口 480 s, 晚 120 s 进入 -> 实际可用现实时间 360 s
        self.assertEqual(enter["remaining_real_duration_s"], 360)
        h.close()

    def test_remaining_never_negative(self):
        h = SimHarness(window_s=60.0, max_real_s=1200.0)
        h.start_run([src(1, 0.0, 0.0, 1500.0)])
        h.fake_clock.advance(59.9)
        enter = h.enter()
        self.assertGreaterEqual(enter["remaining_real_duration_s"], 0)
        self.assertLessEqual(enter["remaining_real_duration_s"], 1)
        h.close()

    def test_virtual_timeout(self):
        h = SimHarness(max_virtual_s=100.0)
        h.start_run([src(1, 0.0, 0.0, 1500.0)])
        h.enter()
        r = h.measure(2000.0, 0.0, 1)  # 需要 400 s 移动, 超过虚拟上限
        self.assertTrue(r["accepted"])
        self.assertTrue(h.engine.run.stats.virtual_timeout_flagged)
        h.server.tick()
        self.assertEqual(h.engine.run.stats.end_reason, "virtual_timeout")
        h.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
