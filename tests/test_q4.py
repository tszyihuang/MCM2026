"""问题4 环境的回归用例: 案例生成逐位一致 / 动作等价 / 发布版求解器不变量。

这些用例是**问题4 环境**（``world.py``，问题3 与问题4 共用）的护栏, 任何一处物理
写错都会在这里失败。问题4 求解器是 ``sweeper4``（集合覆盖 + 锚点射线 + 边界环）。

  * ``world._draw_case(seed, 4)`` 必须与 ``simulator.core.CaseGenerator`` **逐位**一致
    （随机数消耗顺序错一步, 后续全部错位）;
  * 同一动作序列在 ``World(problem=4)`` 与 ``SimulatorEngine`` 上必须逐步一致
    （含定向源的覆盖角、near、清除）;
  * ``sweeper4``: 确定性、时间口径、**边界环真的把漏源补上**、双口径一致。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from robotdog.solver import sweeper4 as sc
from robotdog.solver.belief4 import Belief4
from robotdog.solver.world import World, _draw_case
from simulator.core import CaseGenerator, SimulatorEngine


class CaseGenerationTest(unittest.TestCase):
    def test_case_generation_bitwise_matches_simulator(self):
        for seed in (9500, 9501, 9502, 12345, 777):
            a = CaseGenerator(seed).generate(4)
            b = _draw_case(seed, 4)
            self.assertEqual(a.total, len(b), "seed %d 源个数不一致" % seed)
            for sa, sb in zip(a.sources, b):
                self.assertEqual(sa.channel, sb.channel)
                self.assertAlmostEqual(sa.x, sb.x, places=9)
                self.assertAlmostEqual(sa.y, sb.y, places=9)
                self.assertAlmostEqual(sa.radius_m, sb.radius_m, places=9)
                self.assertEqual(sa.kind, sb.kind)
                if sa.direction_deg is None:
                    self.assertIsNone(sb.direction_deg)
                else:
                    self.assertAlmostEqual(sa.direction_deg, float(sb.direction_deg), places=9)

    def test_problem3_path_unchanged(self):
        """问题3 的案例生成不能被问题4 的改动影响 (随机数消耗顺序必须保持)。"""
        from robotdog.solver.world import generate_case
        for seed in (9500, 9501):
            a = CaseGenerator(seed).generate(3)
            b = generate_case(seed)
            self.assertEqual(a.total, len(b))
            for sa, sb in zip(a.sources, b):
                self.assertEqual(sa.channel, sb.channel)
                self.assertAlmostEqual(sa.x, sb.x, places=9)
                self.assertAlmostEqual(sa.y, sb.y, places=9)


class ActionEquivalenceTest(unittest.TestCase):
    """同一动作序列在 World(problem=4) 与 SimulatorEngine 上逐步一致。"""

    def _run(self, seed: int, steps: int = 200) -> int:
        w = World(seed=seed, run_id="T4", problem=4)
        eng = SimulatorEngine(team_id="T4", countdown_s=0.0)
        eng.new_run(4, seed=seed, code="T4-%d" % seed)
        run = eng.run
        run.run_id = "T4"
        now = eng.clock.now()
        run.phase = "armed"
        run.window_start = now
        run.window_deadline = now + 1500.0
        self.assertTrue(eng.handle("enter", {"arena_id": "default", "robot_id": "T4",
                                             "request_id": "e0"}).get("accepted"))
        rng = np.random.default_rng(seed)
        # 动作序列一次性生成: 边跑边判会让两个引擎的分支错开
        plan = []
        for _ in range(steps):
            s = w.sources[int(rng.integers(0, w.n_sources))]
            kind = "clear" if rng.random() < 0.25 else "measure"
            r = float(rng.uniform(1.0, 1200.0))
            a = float(rng.uniform(0.0, 2.0 * math.pi))
            plan.append((kind, round(s.x + r * math.cos(a), 6),
                         round(s.y + r * math.sin(a), 6), s.channel))
        n_bad = 0
        for i, (kind, x, y, ch) in enumerate(plan):
            if kind == "clear":
                ra = w.clear(x, y, ch).result
                rb = eng.handle("clear", {"arena_id": "default", "robot_id": "T4",
                                          "request_id": "c%d" % i,
                                          "position": {"x": x, "y": y},
                                          "channel": ch}).get("clear_result")
            else:
                info = w.measure(x, y, ch)
                resp = eng.handle("measure", {"arena_id": "default", "robot_id": "T4",
                                              "request_id": "m%d" % i,
                                              "position": {"x": x, "y": y},
                                              "channel": ch})
                ra, rb = info.result, resp.get("measure_result")
                if ra == "direction" and rb == "direction":
                    self.assertAlmostEqual(info.svd_deg, float(resp["svd_deg"]), places=9)
            self.assertEqual(ra, rb, "step %d (%s ch%d) 结果不一致" % (i, kind, ch))
            # 时钟: World 浮点累加 vs 引擎微秒整数累加, 只允许 1e-4 级舍入差
            self.assertLess(abs(w.virtual_t - float(resp.get("virtual_time_s", 0.0))), 1e-4) \
                if kind == "measure" else None
        return n_bad

    def test_equivalence_seed_9500(self):
        self._run(9500)

    def test_equivalence_clears_hit_and_miss(self):
        self._run(9503, steps=150)

    def test_belief4_is_problem4_default(self):
        """``World(problem=4)`` 的默认信念是 ψ 分箱信念 (环境语义)。"""
        self.assertIsInstance(World(seed=9500, problem=4).belief, Belief4)
        self.assertNotIsInstance(World(seed=9500, problem=3).belief, Belief4)


class BoundaryRingTest(unittest.TestCase):
    """边界环: 发布版"100% 清除"的唯一来源。"""

    def test_ring_is_on_by_default(self):
        cfg = sc.build_cfg()
        self.assertTrue(cfg.boundary_ring)
        self.assertEqual(len(sc.build_pool(cfg)),
                         len(sc.SC_COVER_POINTS) + cfg.boundary_ring_n)

    def test_ring_points_are_inside_arena(self):
        from robotdog.consts import ARENA_RADIUS_M
        for (x, y) in sc.boundary_ring_points(1785.0, 24):
            self.assertLessEqual(math.hypot(x, y), ARENA_RADIUS_M + 1e-9)

    def test_ring_covers_the_edge_sliver_source(self):
        """seed 9588 的 ch9: 离边界 9.4 m 的定向源, 原本 24 个点一个都够不到。

        它的可检测区域是贴着边界的一条弧带, 因此**必须**有贴边的补扫点。
        这条用例是回归护栏: 谁把边界环关掉/改小, 这里就会红。
        """
        w = World(seed=9588, problem=4)
        s = [x for x in w.sources if x.channel == 9][0]
        self.assertFalse(s.is_omni)
        self.assertGreater(math.hypot(s.x, s.y), 1780.0, "该源应当是贴边源")

        def reachable(pool):
            for (qx, qy) in pool:
                if math.hypot(qx, qy) > 1800.0:
                    continue
                if math.hypot(qx - s.x, qy - s.y) <= s.radius_m and s.in_coverage(qx, qy):
                    return True
            return False

        self.assertFalse(reachable(sc.SC_COVER_POINTS), "基线 24 点本应覆盖不到它")
        self.assertTrue(reachable(sc.build_pool(sc.build_cfg())), "边界环应当覆盖到它")

    def test_edge_source_gets_cleared(self):
        w = World(seed=9588, problem=4, **sc.world_kwargs())
        sc.run_sweeper4(w, sc.build_cfg())
        self.assertEqual(w.cleared_count, w.n_sources)


class StrategyTest(unittest.TestCase):
    """发布版求解器的不变量: 确定性、时间口径、机制真的在跑。"""

    def test_deterministic(self):
        a = World(seed=9507, problem=4, **sc.world_kwargs())
        sc.run_sweeper4(a, sc.build_cfg())
        b = World(seed=9507, problem=4, **sc.world_kwargs())
        sc.run_sweeper4(b, sc.build_cfg())
        self.assertEqual(a.cleared_count, b.cleared_count)
        self.assertEqual(a.virtual_t, b.virtual_t)
        self.assertEqual(a.moved_m, b.moved_m)
        self.assertEqual(a.measures, b.measures)

    def test_timing_and_clearing_sanity(self):
        w = World(seed=9500, problem=4, **sc.world_kwargs())
        cfg = sc.build_cfg()
        sc.run_sweeper4(w, cfg)
        self.assertLessEqual(w.virtual_t, cfg.max_virtual)
        self.assertGreater(w.cleared_count, 0)
        self.assertLessEqual(w.moved_m / 5.0, w.virtual_t + 1e-6)

    def test_deploy_mode_shares_clearing_decision(self):
        """``knows_total`` 对本求解器无影响 —— 它从不使用真值总数。

        因此进程内与部署口径应当**逐位相同**, 这正是它在正式测试里的优势。
        """
        a = World(seed=9502, problem=4, **sc.world_kwargs())
        sc.run_sweeper4(a, sc.build_cfg(), knows_total=False)
        b = World(seed=9502, problem=4, **sc.world_kwargs())
        sc.run_sweeper4(b, sc.build_cfg(), knows_total=True)
        self.assertEqual(a.cleared_count, b.cleared_count)
        self.assertEqual(a.virtual_t, b.virtual_t)
        self.assertEqual(a.moved_m, b.moved_m)

    def test_no_truth_access(self):
        """求解器不得读取真值: 它的世界装配给的是**空信念**, 真值只能靠 /measure。"""
        w = World(seed=9500, problem=4, **sc.world_kwargs())
        self.assertEqual(w.belief.tracks, [])
        w2 = World(seed=9500, problem=4, **sc.world_kwargs())
        sc.run_sweeper4(w2, sc.build_cfg())
        self.assertGreater(w2.cleared_count, 0)

    def test_ray_sweep_and_anchor_search_are_used(self):
        """射线盲清 / 锚点搜索必须真的被触发 —— 它们是"看见过却清不掉"那一类的解药。"""
        calls = {"ray": 0, "anchor": 0}
        orig_ray = sc.SetCoverSolver._ray_sweep
        orig_anc = sc.SetCoverSolver._anchor_search

        def ray(self, ch):
            calls["ray"] += 1
            return orig_ray(self, ch)

        def anc(self, x, y, ch):
            calls["anchor"] += 1
            return orig_anc(self, x, y, ch)

        sc.SetCoverSolver._ray_sweep, sc.SetCoverSolver._anchor_search = ray, anc
        try:
            for seed in (9500, 9501, 9502):
                w = World(seed=seed, problem=4, **sc.world_kwargs())
                sc.run_sweeper4(w, sc.build_cfg())
        finally:
            sc.SetCoverSolver._ray_sweep = orig_ray
            sc.SetCoverSolver._anchor_search = orig_anc
        self.assertGreater(calls["ray"], 0, "射线扫描从未被调用")
        self.assertGreater(calls["anchor"], 0, "锚点搜索从未被调用")

    def test_config_defaults(self):
        cfg = sc.build_cfg()
        self.assertEqual(cfg.step_ladder, (700.0, 420.0, 240.0, 130.0, 70.0, 35.0, 16.0))
        self.assertEqual(cfg.boundary_ring_r, 1785.0)
        self.assertEqual(cfg.boundary_ring_n, 24)
        self.assertAlmostEqual(cfg.ray_dist_step, 35.0)
        self.assertLess(cfg.ray_lateral_rad * cfg.ray_dist_max, 20.0,
                        "横向摆动必须小于清除半径 20 m, 否则两侧采样会跨过源")


if __name__ == "__main__":
    unittest.main()
