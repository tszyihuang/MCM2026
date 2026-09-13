"""问题3 求解器的自检: 与 simulator.core 的物理等价性 + 信念 / 原语正确性.

运行:
    python -m unittest tools.tests.test_solver_equiv -v
"""

from __future__ import annotations

import math
import os
import random
import unittest

import numpy as np

from robotdog.solver import ops                                  # noqa: E402
from robotdog.solver.world import World, generate_case           # noqa: E402
from simulator.core import CaseGenerator, SimulatorEngine       # noqa: E402


class TestCaseGeneration(unittest.TestCase):
    def test_identical_to_simulator(self):
        for seed in (1, 2, 1001, 424242):
            a = generate_case(seed)
            b = CaseGenerator(seed).generate(3).sources
            self.assertEqual(len(a), len(b))
            for sa, sb in zip(a, b):
                self.assertEqual(sa.channel, sb.channel)
                self.assertAlmostEqual(sa.x, sb.x, places=12)
                self.assertAlmostEqual(sa.y, sb.y, places=12)
                self.assertAlmostEqual(sa.radius_m, sb.radius_m, places=12)

    def test_source_count_range(self):
        for seed in range(50):
            srcs = generate_case(seed)
            self.assertTrue(10 <= len(srcs) <= 16)
            self.assertEqual(len({s.channel for s in srcs}), len(srcs))
            for s in srcs:
                self.assertLessEqual(math.hypot(s.x, s.y), 1800.0 + 1e-9)
                self.assertTrue(1000.0 <= s.radius_m <= 1500.0)


class TestPhysicsEquivalence(unittest.TestCase):
    """同一动作序列在 robotdog.solver.World 与 simulator.core 上必须逐步一致。"""

    def _run(self, seed: int, n: int = 240):
        w = World(seed=seed, run_id="EQ")
        eng = SimulatorEngine(team_id="T", countdown_s=0.0)
        eng.new_run(3, seed=seed, code="EQ")
        eng.run.run_id = "EQ"
        now = eng.clock.now()
        eng.run.phase = "armed"
        eng.run.window_start = now
        eng.run.window_deadline = now + 1500.0      # 对齐 baseline_ref.build_engine
        eng.handle("enter", {"arena_id": "default", "robot_id": "T", "request_id": "enter"})
        rng = random.Random(seed ^ 0x5EED)
        for step in range(n):
            x = rng.uniform(-2200, 2200)
            y = rng.uniform(-2200, 2200)
            c = rng.randint(1, 20)
            base = {"arena_id": "default", "robot_id": "T", "request_id": "r%d" % step,
                    "position": {"x": x, "y": y}, "channel": c}
            if rng.random() < 0.22:
                mine = w.clear(x, y, c)
                resp = eng.handle("clear", base)
                self.assertEqual(resp["clear_result"], mine.result)
            else:
                mine = w.measure(x, y, c)
                resp = eng.handle("measure", base)
                self.assertEqual(resp["measure_result"], mine.result)
                if mine.result == "direction":
                    self.assertAlmostEqual(resp["svd_deg"], mine.svd_deg, places=9)
            self.assertAlmostEqual(eng.run.virtual_seconds(), w.virtual_t, delta=1e-5)
        self.assertEqual(eng.run.cleared_count, w.cleared_count)

    def test_three_seeds(self):
        for seed in (1001, 1050, 1111):
            with self.subTest(seed=seed):
                self._run(seed)

    def test_near_and_clear_radius_semantics(self):
        w = World(seed=7)
        s = w.sources[0]
        # 距离 3 m 且全向 ⇒ near
        got = w.measure(s.x + 3.0, s.y, s.channel)
        self.assertEqual(got.result, "near")
        # 距离 25 m ⇒ 超出清除半径, 应"未发现"
        got2 = w.clear(s.x + 25.0, s.y, s.channel)
        self.assertEqual(got2.result, "no_target_in_range")
        # 距离 19 m ⇒ 命中
        got3 = w.clear(s.x + 19.0, s.y, s.channel)
        self.assertEqual(got3.result, "success")


class TestBelief(unittest.TestCase):
    def test_two_bearings_localize(self):
        errs = []
        for seed in range(3001, 3041):
            w = World(seed=seed)
            s = None
            for cand in w.sources:
                # 两个检测点都要在**该源自己的有效接收半径**内, 否则第二条示向度拿不到
                if (math.hypot(cand.x, cand.y) <= cand.radius_m - 30.0
                        and math.hypot(cand.x - 1200, cand.y - 400) <= cand.radius_m - 30.0):
                    s = cand
                    break
            if s is None:
                continue
            w.measure(0.0, 0.0, s.channel)
            w.measure(1200.0, 400.0, s.channel)
            tr = w.belief.tracks[s.channel - 1]
            self.assertEqual(tr.mode, 2)
            self.assertIsNotNone(tr.est)
            errs.append(math.hypot(tr.est[0] - s.x, tr.est[1] - s.y))
        self.assertGreater(len(errs), 20)
        self.assertLess(float(np.mean(errs)), 60.0)
        self.assertLess(float(np.max(errs)), 200.0)
        self.assertTrue(all(e == e for e in errs))       # 无 NaN

    def test_pi_not_destroyed_by_far_no_signal(self):
        """已确认存在的源, 在远于 R 上界的点收到 no_signal 不应摧毁其存在概率。"""
        w = World(seed=1074)
        tr = None
        for s in w.sources:
            if s.channel == 6:
                tr = s
        self.assertIsNotNone(tr)
        w.measure(-50.0, 321.0, 6)
        w.measure(350.0, 1014.0, 6)
        b = w.belief.tracks[5]
        self.assertEqual(b.mode, 2)
        self.assertGreater(b.pi, 0.9)
        lo, hi = b.r_bounds()
        self.assertGreater(lo, 1000.0)          # 收到过信号 ⇒ R 有下界
        # 在估计点 1600 m 外做 no_signal: 该处必然收不到, pi 不应显著下降
        est = b.est
        far = (est[0] + 1600.0, est[1])
        w.measure(far[0], far[1], 6)
        self.assertGreater(b.pi, 0.9)

    def test_duplicate_measurement_is_ignored(self):
        w = World(seed=11)
        s = w.sources[0]
        w.measure(0.0, 0.0, s.channel)
        tr = w.belief.tracks[s.channel - 1]
        n0 = tr.n_dir
        w.measure(0.0, 0.0, s.channel)          # 同点同频道: 读数不变, 无新信息
        self.assertEqual(tr.n_dir, n0)


class TestOps(unittest.TestCase):
    def test_pursue_clears_from_single_bearing(self):
        ok = 0
        tot = 0
        for seed in range(2001, 2031):
            w = World(seed=seed)
            cands = [s for s in w.sources if math.hypot(s.x, s.y) <= 1300]
            if not cands:
                continue
            s = cands[0]
            if w.measure(0.0, 0.0, s.channel).result != "direction":
                continue
            tot += 1
            if ops.pursue(w, s.channel)[0]:
                ok += 1
        self.assertGreater(tot, 20)
        self.assertGreaterEqual(ok / tot, 0.9)

    def test_measure_point_is_perpendicular(self):
        """只有一条射线时, 下一个检测点必须偏离视线 (否则纵向距离永远定不下来)。"""
        w = World(seed=2001)
        s = [x for x in w.sources if math.hypot(x.x, x.y) <= 1300][0]
        w.measure(0.0, 0.0, s.channel)
        tr = w.belief.tracks[s.channel - 1]
        p = ops.choose_measure_point(w, s.channel, ops.PursueConfig())
        est = tr.est
        u = (est[0], est[1])
        n = math.hypot(*u)
        u = (u[0] / n, u[1] / n)
        v = (p[0] - est[0], p[1] - est[1])
        cross = abs(u[0] * v[1] - u[1] * v[0])
        self.assertGreater(cross, 200.0)


if __name__ == "__main__":
    unittest.main()
