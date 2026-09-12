"""物理规则与虚拟计时测试 (附件1 表1/表2, 附件2 第4节)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.harness import SimHarness, src  # noqa: E402


class TimingTest(unittest.TestCase):
    """附件1 表2 / 附件2 第10节 的完整计时示例。"""

    def setUp(self):
        self.h = SimHarness()
        self.h.start_run([src(3, 1000.0, 0.0, 1500.0)])
        self.h.enter()
        self.base = self.h.engine.run.virtual_seconds()

    def tearDown(self):
        self.h.close()

    def test_full_timing_example(self):
        r = self.h.measure(300, 400, 1)
        self.assertEqual(r["virtual_time_s"], 105.0)  # 500 m / 5 = 100 s, 检测 5 s, 无切换
        r = self.h.measure(300, 400, 2)
        self.assertEqual(r["virtual_time_s"], 111.0)  # + 切换 1 s + 检测 5 s
        r = self.h.clear(300, 0, 3)
        self.assertEqual(r["virtual_time_s"], 194.0)  # 400 m -> 80 s, 未发现 3 s
        r = self.h.measure(300, 0, 2)
        self.assertEqual(r["virtual_time_s"], 199.0)  # 无移动, 频道仍为 2, 无切换
        r = self.h.exit()
        self.assertEqual(r["virtual_time_s"], 199.0)
        self.assertEqual(r["exit_reason"], "user_exit")

    def test_channel_switch_only_on_measure(self):
        before = self.h.engine.run.virtual_seconds()
        self.h.clear(0, 0, 5)  # /clear 不切换频道
        self.assertEqual(self.h.engine.run.channel, 1)
        self.assertAlmostEqual(self.h.engine.run.virtual_seconds() - before, 3.0, places=6)

    def test_move_distance_uses_euclidean(self):
        self.h.measure(3, 4, 1)
        self.assertAlmostEqual(self.h.engine.run.virtual_seconds() - self.base, 6.0, places=6)  # 5 m -> 1 s + 5 s

    def test_movement_outside_arena_allowed(self):
        r = self.h.measure(5000, 0, 1)
        self.assertTrue(r["accepted"])
        self.assertAlmostEqual(r["virtual_time_s"] - self.base, 1005.0, places=6)


class MeasurePhysicsTest(unittest.TestCase):
    def setUp(self):
        self.h = SimHarness()
        self.h.start_run(
            [
                src(1, 300.0, 400.0, 1500.0),
                src(2, 0.0, 0.0, 1000.0),
                src(5, 0.0, 900.0, 1200.0, kind="directional", direction=90.0),
            ]
        )
        self.h.enter()

    def tearDown(self):
        self.h.close()

    def test_direction_bearing(self):
        r = self.h.measure(0, 0, 1)
        self.assertEqual(r["measure_result"], "direction")
        self.assertAlmostEqual(r["svd_deg"], 53.13, places=2)  # atan2(400,300)

    def test_svd_format(self):
        r = self.h.measure(123.456, -77.7, 1)
        self.assertEqual(round(r["svd_deg"], 2), r["svd_deg"])
        self.assertTrue(0.0 <= r["svd_deg"] < 360.0)

    def test_normalization_quadrants(self):
        # 干扰源在西南方向 -> 示向度应在 (180, 270)
        h = SimHarness()
        h.start_run([src(4, -500.0, -500.0, 1500.0)])
        h.enter()
        r = h.measure(0, 0, 4)
        self.assertEqual(r["measure_result"], "direction")
        self.assertAlmostEqual(r["svd_deg"], 225.0, places=2)
        h.close()

    def test_near_within_5m(self):
        r = self.h.measure(3.0, 4.0, 2)  # 距 (0,0) 恰好 5 m
        self.assertEqual(r["measure_result"], "near")
        self.assertNotIn("svd_deg", r)

    def test_just_outside_5m_is_direction(self):
        r = self.h.measure(5.01, 0.0, 2)
        self.assertEqual(r["measure_result"], "direction")

    def test_out_of_range_no_signal(self):
        r = self.h.measure(0.0, 1000.1, 2)  # 有效接收半径 1000 m
        self.assertEqual(r["measure_result"], "no_signal")
        self.assertNotIn("svd_deg", r)

    def test_exactly_on_radius_has_signal(self):
        r = self.h.measure(0.0, 1000.0, 2)
        self.assertEqual(r["measure_result"], "direction")

    def test_directional_coverage_inside(self):
        # 源 (0,900) 定向方向 90 度 -> 覆盖 [0,180] 半平面; (0,1300) 方位 90 -> 可检测
        r = self.h.measure(0.0, 1300.0, 5)
        self.assertEqual(r["measure_result"], "direction")
        self.assertAlmostEqual(r["svd_deg"], 270.0, places=2)  # 从检测点指向源 (正南)

    def test_directional_coverage_boundary_inclusive(self):
        # (500,900) 相对 (0,900) 方位 0 度 = 边界, 含边界
        r = self.h.measure(500.0, 900.0, 5)
        self.assertEqual(r["measure_result"], "direction")

    def test_directional_outside_coverage(self):
        # (0,500) 位于源的南侧, 方位 270, 超出 [0,180] 覆盖范围
        r = self.h.measure(0.0, 500.0, 5)
        self.assertEqual(r["measure_result"], "no_signal")

    def test_near_requires_coverage_for_directional(self):
        # 距离 1 m 但在覆盖范围外 -> 既非 near 也非 direction
        r = self.h.measure(0.0, 899.0, 5)
        self.assertEqual(r["measure_result"], "no_signal")
        # 覆盖范围内的 1 m -> near
        r = self.h.measure(0.0, 901.0, 5)
        self.assertEqual(r["measure_result"], "near")

    def test_cleared_channel_returns_no_signal(self):
        self.h.clear(0.0, 0.0, 2)  # 清除频道 2
        r = self.h.measure(0.0, 0.0, 2)
        self.assertEqual(r["measure_result"], "no_signal")

    def test_empty_channel_returns_no_signal(self):
        r = self.h.measure(0.0, 0.0, 19)
        self.assertEqual(r["measure_result"], "no_signal")


class ClearPhysicsTest(unittest.TestCase):
    def setUp(self):
        self.h = SimHarness()
        self.h.start_run([src(1, 100.0, 0.0, 1500.0), src(7, -800.0, 0.0, 1500.0, kind="directional", direction=180.0)])
        self.h.enter()

    def tearDown(self):
        self.h.close()

    def test_clear_exactly_20m_success(self):
        base = self.h.engine.run.virtual_seconds()
        r = self.h.clear(80.0, 0.0, 1)  # (0,0) -> (80,0): 80 m = 16 s; 清除 5 s
        self.assertEqual(r["clear_result"], "success")
        self.assertAlmostEqual(r["virtual_time_s"] - base, 16.0 + 5.0, places=6)

    def test_clear_just_outside_20m_fails(self):
        base = self.h.engine.run.virtual_seconds()
        r = self.h.clear(79.9, 0.0, 1)  # 79.9 m = 15.98 s; 未发现 3 s
        self.assertEqual(r["clear_result"], "no_target_in_range")
        self.assertAlmostEqual(r["virtual_time_s"] - base, 15.98 + 3.0, places=6)

    def test_clear_directional_ignores_coverage(self):
        # 定向方向 180 度, (0,0) 不在覆盖范围内, 但 20 m 内仍可清除
        r = self.h.clear(-799.0, 0.0, 7)
        self.assertEqual(r["clear_result"], "success")

    def test_clear_cost_recorded_in_stats(self):
        """成功清除 5 s、未发现 3 s, 统计与动作日志必须一致 (虚拟时钟同一口径)。"""
        self.h.clear(100.0, 0.0, 1)  # 成功
        self.h.clear(0.0, 0.0, 20)  # 未发现
        run = self.h.engine.run
        self.assertAlmostEqual(run.stats.clear_time_s, 5.0 + 3.0, places=6)
        clears = [a for a in run.actions if a["action"] == "clear"]
        self.assertAlmostEqual(clears[0]["action_duration_s"], 5.0, places=6)
        self.assertAlmostEqual(clears[1]["action_duration_s"], 3.0, places=6)
        total = sum(a["total_duration_s"] for a in run.actions)
        self.assertAlmostEqual(total, run.virtual_seconds(), places=6)
        parts = run.stats.travel_time_s + run.stats.switch_time_s + run.stats.detect_time_s + run.stats.clear_time_s
        self.assertAlmostEqual(parts, run.virtual_seconds(), places=6)

    def test_double_clear_returns_no_target(self):
        self.h.clear(100.0, 0.0, 1)
        before = self.h.engine.run.virtual_seconds()
        r = self.h.clear(100.0, 0.0, 1)
        self.assertEqual(r["clear_result"], "no_target_in_range")
        self.assertAlmostEqual(self.h.engine.run.virtual_seconds() - before, 3.0, places=6)

    def test_clear_not_existing_channel(self):
        r = self.h.clear(0.0, 0.0, 20)
        self.assertEqual(r["clear_result"], "no_target_in_range")

    def test_clear_does_not_change_channel_or_switch_time(self):
        self.h.measure(0.0, 0.0, 9)
        self.assertEqual(self.h.engine.run.channel, 9)
        self.h.clear(100.0, 0.0, 1)
        self.assertEqual(self.h.engine.run.channel, 9)


class SvDErrorTest(unittest.TestCase):
    """示向度误差必须落在 [-1, 1] 度内, 且同一地点重复检测读数稳定。"""

    def test_error_bounds(self):
        h = SimHarness(error_enabled=True)
        h.start_run([src(1, 1000.0, 0.0, 1500.0)])
        h.enter()
        worst = 0.0
        distinct = set()
        for i in range(120):
            r = h.measure(float(i), 0.0, 1)
            offset = (r["svd_deg"] - 0.0 + 180.0) % 360.0 - 180.0
            worst = max(worst, abs(offset))
            distinct.add(round(r["svd_deg"], 2))
        self.assertLessEqual(worst, 1.0 + 1e-9, "示向度误差超出 [-1,1] 度")
        self.assertGreater(len(distinct), 10, "不同检测点的示向度误差应呈现统计差异")
        h.close()

    def test_error_repeatable_at_same_point(self):
        """同一地点电磁环境固定: 同一位置重复检测读数一致。"""
        h = SimHarness(error_enabled=True)
        h.start_run([src(1, 0.0, 500.0, 1500.0)])
        h.enter()
        first = h.measure(0.0, 0.0, 1)["svd_deg"]
        second = h.measure(0.0, 0.0, 1)["svd_deg"]
        third = h.measure(0.0, 0.0, 1)["svd_deg"]
        self.assertEqual(first, second)
        self.assertEqual(second, third)
        h.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
