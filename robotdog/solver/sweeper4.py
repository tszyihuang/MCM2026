"""问题4 求解器: 集合覆盖补扫 + 锚点射线盲清 + 贴边补扫环。

机制移植自 ``2MCM2026/solution``, 已按本仓库的进程内 ``World`` 重写并纳入统一评测。

来源与身份
----------
本方案**不做贝叶斯推断**: 不回答"源在哪", 而是用几何采样的方式去撞 ——
离线贪心集合覆盖给出 24 个补扫点, 对历史示向度做射线盲清, 再用贴边补扫环兜底。

两个"困难"与两个专用手段 (原实现的论证, 逐条保留)
------------------------------------------------
定向源只在 ±90° 扇区内辐射, 带来两个新困难:

  困难一：站在扇区之外时完全测不到信号 ⇒ 固定扫描点必须足够密, 否则源"从未被发现"。
  困难二：沿示向度逼近途中可能走出扇区而丢失信号 ⇒ 源"被发现但清不掉"。

**手段 1 · 加密补扫点集**。原实现实测 (``common/tools/scan_sets.py``):
2 环×8=16 点定向覆盖 90% / 路程 10490 m; 3 环×16=48 点覆盖 96% / 路程 19255 m;
菱形网格 61 点覆盖 99% / 路程 25157 m。取 48 点覆盖率接近上限而路程可接受。
随后又用**贪心集合覆盖**把点集压到 **24 点**: 建模
``扫描点 Q 覆盖 (P,φ) ⟺ |Q−P| ≤ r_min 且 Q 落在 φ 的 ±90° 扇区内``,
全集 ``U = {(P,φ)}`` 取 P 的 150 m 网格 × φ 的 20° 网格 (共 7938 个),
24 个点即达 **99.94%** 覆盖、巡游 20099 m。端到端实测 (60 局):
48 点 97.89% / 2052 s, 61 点 99.21% / 2340 s, **24 点 99.74% / 1602 s**
—— 点数、清除率、时间三项同时最优。

**手段 2 · 锚点射线搜索**。原实现对 92 个漏失定向源归因:
48 个「从未靠近」由手段1 解决, **44 个「曾走到 50 m 内却没清掉」**由本手段解决。
原理: ``/clear`` 只判距离、与覆盖角无关, 所以既然在 P 点曾测到该频道的示向度 b,
源就必然在过 P、方向 b 的射线上且距离不超过 P 处的接收半径 —— 沿该射线按距离
采样并盲调 ``/clear`` 即可。采样点再做 ±0.3° 的横向摆动, 抵消 ±1° 角误差在远距离的放大
(偏移量必须**小于清除半径 20 m**, 否则两侧采样点会跨过源)。

实测出处 (2MCM2026 自己的口径, 与 MCMRL 标定集不同, 不要混用)
------------------------------------------------------------
原实现报 **清除比例 87.35%、全清 40/200 (20.0%)、平均定位清除 2422 s**
(200 局, 见 ``2MCM2026/solution/common/out/FINAL_p4_200.json``)。
理论上限由 ``common/tools/p4_ceiling.py`` 给出: 48 点扫描集对定向源可发现率 96%,
剩余差距来自"发现后在逼近途中丢失信号"。

> 本模块在标定集 (seed 9500-9799) 上的成绩由 ``tools/eval_q4.py`` 实测给出。

相对原实现的三处**工程性改动** (不改算法, 但必须声明)
------------------------------------------------------
1. **可调参数从模块全局改为配置对象**。原实现把 ``MAX_TRIES`` / ``SCHED_BEARING_WEIGHT``
   / ``RAY_*`` 放在 ``common.core_solver`` 的模块全局, 由 ``problem4/solver.py`` 在
   **import 时改写** —— 这会静默重调同进程内**所有** ``BaseSolver`` 子类, 且
   ``STEP_LADDER`` 是在类体求值时绑定的, 改模块常量对它无效。本模块把它们全部
   搬进 :class:`SetCoverConfig`, 既消除进程级副作用, 也让 ``eval_q4 --set k=v`` 可用。
2. **加了虚拟时间上限**。原内核**完全不看时间**: 停止条件只有动作数 ``GUARD_LIMIT``、
   轮数 ``LOOP_ROUNDS`` 和"补扫无新发现"。而收尾的射线扫描**不受 guard 约束**
   (guard 只在主循环每轮开头检查一次), 单频道最坏可发出约
   ``6 obs × 37 距离 × 5 横向 = 1110`` 次 ``/clear``。原模拟器靠虚拟/现实上限兜底,
   MCMRL 的 ``World`` 不做任何限制。由于本题报的是**平均定位清除时间**
   (= 虚拟总时间 / 已清除数), 没有上限的收尾会直接把成绩毁掉, 故加 ``max_virtual``。
3. **收尾阶段额外计数** (``q4_hunt_found``), 仅为对照表取数, 不影响策略。

新增的边界环: 把清除比例从 0.9996 推到 **1.0000** (本仓库最重要的改动)
----------------------------------------------------------------------
原实现的 24 点贪心集合覆盖对「位置 × 定向方向」空间的覆盖率是 **99.94%**,
剩下那 0.06% 在标定集 300 局里就体现为 **1 个漏源** (seed 9588 的 ch9)。
把那个源拆开看, 它的形状说明了全部问题:

    ch9: pos=(-132.8, 1785.7) |pos|=1790.6  R=1035.5  psi=73.1°  (定向源)

源**离竞技场边界只有 9.4 m**。它的可检测区域
``{P : |P|≤1800, |P−S|≤R, P 在 psi±90° 扇区内}`` 因此被压成**贴着边界的一条弧带**
(约 2450 m²), 而 24 个补扫点**一个都没落进去** —— 于是该源整局从未被检测到。
这不是"策略不好", 是**纯几何的覆盖缺口**: 越靠边的源, 可检测区域越薄。

补法很直接: **在贴边处再加一圈补扫点**。本仓库新增
``boundary_ring`` (默认开, r = 1785 m, 24 点 = 每 15°)。

**静态覆盖审计** (2000 局 / 12941 个定向源, 纯几何、不跑模拟器):

  ====================  ======  ==========  ========
  补扫点集               点数    漏定向源    漏源率
  ====================  ======  ==========  ========
  原实现 24 点            24      567        4.38%
  24 + r1730×24          48       64        0.49%
  24 + r1760×24          48       25        0.19%
  **24 + r1785×24**      48        6        0.031%
  24 + r1785×48          72        4        0.031%
  ====================  ======  ==========  ========

(该审计只数「补扫点集本身」能否覆盖, 是**上界** —— 实际跑时机器狗在逼近/锚点/
螺旋阶段还会顺带测到一些源。所以它用来**排序**候选点集, 不用来报成绩。)

**端到端实测** (seed 20000-21999, 2000 局全新种子, 从未参与任何标定):

  ====================  ======  ==========  ==========  ==========
  补扫点集               点数    清除比例    全清局      漏源
  ====================  ======  ==========  ==========  ==========
  原实现 24 点            24     0.99962    1990/2000     10
  **24 + r1785×24**      48     **1.00000** **2000/2000**  **0**
  24 + r1785×36          60     1.00000    2000/2000      0
  24 + r1785×48          72     1.00000    2000/2000      0
  ====================  ======  ==========  ==========  ==========

代价: 平均定位清除 1323.5 → **1787.8 s/源** (+35%), 移动 69 485 → 93 792 m。

**三档预设** (都可用 ``tools/eval_q4.py --set k=v`` 切换):

  ================  ==========================================  ==========  ==========
  档                 参数                                        清除比例     s/源
  ================  ==========================================  ==========  ==========
  极速 (不保证)       ``boundary_ring=False``                     0.99962     1323.5
  **全清 (默认)**     ``boundary_ring=True``, r=1785, n=24        **1.00000** **1787.8**
  更冗余             ``boundary_ring_n=36``                      1.00000     2039.3
  ================  ==========================================  ==========  ==========

> **为什么把全清档设为默认**: 题目写的是「**在确保所有干扰源被清除的前提下**,
> 完成任务的时间越短越好」—— 清除是**前置条件**, 时间只是次要目标。
> 极速档的 0.9996 好看在时间上, 但它把"是否满足题目前提"交给了运气。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..consts import N_CHANNELS
from .sweeper import SweepConfig, SweepResult
from .world import World

Vec = Tuple[float, float]

#: 贪心集合覆盖得到的 24 点补扫点集 (覆盖「位置 × 定向方向」空间 99.94%)。
#: 逐位照抄 ``2MCM2026/solution/problem4/solver.py`` 的 ``COVER_POINTS``;
#: 生成脚本是 ``2MCM2026/solution/common/tools/setcover_p4.py``。
SC_COVER_POINTS: List[Vec] = [
    (-750.0, -250.0), (750.0, 250.0), (-250.0, 750.0), (250.0, -750.0),
    (-750.0, -1250.0), (750.0, 1250.0), (-1250.0, 750.0), (1250.0, -750.0),
    (-1750.0, -250.0), (1750.0, 250.0), (-500.0, 1750.0), (250.0, -1750.0),
    (1000.0, 1750.0), (1250.0, -1500.0), (-1500.0, -1250.0),
    (-1500.0, 1250.0), (0.0, 0.0), (1500.0, 750.0), (-500.0, -2000.0),
    (0.0, 1500.0), (2000.0, -500.0), (-2000.0, 250.0), (250.0, -1500.0),
    (-1000.0, 250.0),
]

#: 原实现的粗距离阶梯 (``RAY_DIST_STEP = 0`` 时的历史行为)。
SC_RAY_LEGACY_LADDER = (200.0, 400.0, 650.0, 950.0, 1300.0)


class SetCoverConfig(SweepConfig):
    """集合覆盖方案的配置。

    继承 :class:`SweepConfig` 是为了让 ``tools/eval_q4.py`` 的 ``--set k=v`` 与
    ``build_cfg()`` 约定统一; 本方案**不读** ``SweepConfig`` 的任何信念/收尾旋钮
    (``hunt_mass_min`` / ``probe_*`` / ``joint_*`` 等对它都没有意义),
    只读 ``max_virtual``。

    全部数值默认 = 2MCM2026 ``problem4/solver.py`` 调优后的**生效值**
    (注意: 该文件先设 ``step_ladder = STEP_LADDER_DIRECTED`` 又覆盖成
    ``(700, 420, 240, 130, 70, 35, 16)``, 后者才生效; 本配置取后者)。
    """

    # --- 步长阶梯: 定向源用小步, 大步会冲出 ±90° 扇区导致信号丢失 ---
    step_ladder: Tuple[float, ...] = (700.0, 420.0, 240.0, 130.0, 70.0, 35.0, 16.0)
    # --- 调度: 候选频道按 距离 + w·方位偏差 排序 ---
    sched_bearing_weight: float = 3.0
    # --- 主循环护栏 ---
    max_tries: int = 2            # 单频道连续逼近失败多少次后暂停主动逼近
    guard_limit: int = 3000       # 单局动作数上限
    loop_rounds: int = 300        # 主循环轮数上限
    rescan_k: int = 0             # 补扫连续 K 点无收获即停; 0 = 不停 (优先清除率)
    # --- 补扫点集 ---
    cover_points: Tuple[Vec, ...] = tuple(SC_COVER_POINTS)
    # --- **边界环**: 把清除比例从 0.9996 推到 1.0000 的那一圈 ---------------
    #: 是否在 ``cover_points`` 之外再加一圈贴边的补扫点。
    #: ``False`` = 极速档 (2000 局全新种子实测 0.99962); ``True`` = 全清档 (**1.00000**)。
    boundary_ring: bool = True
    #: 环半径 (m)。竞技场半径 1800, 取 1785 留 15 m 余量。
    #: **越贴边越好**: 静态覆盖审计 (2000 局 / 12941 个定向源) 里 r=1730 还剩 0.49%
    #: 的定向源覆盖不到, r=1785 只剩 0.031% —— 残余漏源的形状就是贴着边界的一条弧带。
    boundary_ring_r: float = 1785.0
    #: 环上点数 (24 点 = 每 15°, 相邻点弦长约 467 m)。
    #: 24 点已实测 2000 局**零遗漏**; 调大只是更冗余、更慢 (36 点 → 约 +14% 时间)。
    boundary_ring_n: int = 24
    use_anchor_search: bool = True   # 是否启用收尾锚点 + 射线搜索
    # --- 射线扫描 (收尾专用) ---
    ray_on_fail: bool = True         # 主循环逼近失败时是否也做射线扫描
    ray_dist_step: float = 35.0      # 距离采样步长; 0 = 退回粗阶梯
    ray_dist_min: float = 30.0
    ray_dist_max: float = 1320.0
    ray_lateral_rad: float = 0.005235987755982988   # 0.3° ⇒ 远距离偏移 < 20 m
    # --- 本移植新增: 虚拟时间上限 (见模块 docstring 改动 2) ---
    max_virtual: float = 90000.0


def boundary_ring_points(radius: float, n: int) -> List[Vec]:
    """贴边均匀分布在半径 ``radius`` 上的 ``n`` 个补扫点 (每 ``360/n`` 度一个)。"""
    n = int(n)
    if n <= 0:
        return []
    return [(radius * math.cos(2.0 * math.pi * i / n),
             radius * math.sin(2.0 * math.pi * i / n)) for i in range(n)]


def build_pool(cfg: SetCoverConfig) -> List[Vec]:
    """实际使用的补扫点集 = ``cover_points`` (+ 边界环)。"""
    pts: List[Vec] = list(cfg.cover_points)
    if cfg.boundary_ring:
        pts += boundary_ring_points(cfg.boundary_ring_r, cfg.boundary_ring_n)
    return pts


def build_cfg(**kw: Any) -> SetCoverConfig:
    """发布配置 (默认 = 原实现的生效档 **+ 边界环**, 即全清档)。

    支持 ``build_cfg(boundary_ring=False)`` 形式的覆盖, 便于切到极速档。
    """
    cfg = SetCoverConfig()
    for k, v in kw.items():
        if not hasattr(SetCoverConfig, k):
            raise AttributeError("SetCoverConfig has no attribute %r" % k)
        setattr(cfg, k, v)
    return cfg


class _NullBelief:
    """占位信念: 本方案**不做任何贝叶斯推断**。

    路线分歧使然 —— 集合覆盖方案把"源在哪"交给几何采样回答 (补扫点集 + 射线盲清),
    没有后验可更新。``World.measure`` / ``World.clear`` 会照常调用
    ``on_measure`` / ``on_clear``, 这里用空实现接住: 既不维护一份没人会读的信念,
    也省掉每局约 500 次观测更新 (20 频道 × 约 1000 格点网格) 的开销。
    """

    def __init__(self) -> None:
        self.tracks: List[Any] = []

    def on_measure(self, mx: float, my: float, channel: int, kind: str,
                   svd: Optional[float], t: float) -> None:
        return None

    def on_clear(self, mx: float, my: float, channel: int, hit: bool) -> None:
        return None


def world_kwargs() -> Dict[str, Any]:
    """``World(...)`` 的额外关键字: 本方案不需要信念, 装一个空实现即可。"""
    return {"belief_cls": _NullBelief}


class _OutOfBudget(Exception):
    """虚拟时间超预算; 由 :meth:`SetCoverSolver.solve` 捕获后正常收尾。"""


class _WorldCtx:
    """把 MCMRL 的 ``World`` 适配成原内核期望的 ``ctx`` 形状。

    原内核与环境的唯一契约是 ``enter / measure / clear / exit`` 四个方法加一个
    ``position`` **属性**, 响应是**字典**。``World`` 返回的是 ``StepInfo``, 因此
    这一层只做形状翻译, 不含任何策略逻辑。

    两个容易踩空的点, 都按原运行的语义严格照做:

    * ``position`` 必须在 ``measure`` **和** ``clear`` 之后都刷新。原内核的
      ``pos()`` 优先读 ``ctx.position``; 一旦退化成它自己的 ``_last_pos`` 兜底,
      而那个字段只由 ``measure`` 写 (``clear`` 不写), 航位推算就会在每次
      ``/clear`` 之后变陈旧 —— 这是这类移植最典型的静默失效模式。
    * 结果里有 ``direction`` 时必须带 ``svd_deg`` 键 (原内核是无保护索引)。
    """

    def __init__(self, world: World, cfg: SetCoverConfig) -> None:
        self.w = world
        self.cfg = cfg
        self.position: Vec = world.pos
        self.actions = 0

    def _tick(self) -> None:
        self.actions += 1
        if self.w.virtual_t > self.cfg.max_virtual:
            raise _OutOfBudget("virtual_t %.0f > %.0f" % (self.w.virtual_t, self.cfg.max_virtual))

    def enter(self) -> Dict[str, Any]:
        return {}

    def exit(self) -> Dict[str, Any]:
        return {}

    def measure(self, x: float, y: float, ch: int) -> Dict[str, Any]:
        self._tick()
        si = self.w.measure(x, y, ch)
        self.position = self.w.pos
        out: Dict[str, Any] = {"measure_result": si.result}
        if si.result == "direction":
            out["svd_deg"] = si.svd_deg
        return out

    def clear(self, x: float, y: float, ch: int) -> Dict[str, Any]:
        self._tick()
        si = self.w.clear(x, y, ch)
        self.position = self.w.pos
        return {"clear_result": si.result}


class SetCoverSolver:
    """问题4 内核: 全区域补扫 + 示向度阶梯逼近 + 锚点/射线盲清。

    频道三态 (照抄原实现, 每一态都有实测依据):

      * ``_cleared_ok`` 已成功清除, 不再处理;
      * ``alive``       曾测到过信号;
      * ``_abandoned``  重试耗尽, 暂停主动逼近, 但**仍参与补扫测量**, 换位置后
        重新测到信号即复活。原实现注明: 旧版把这一态并入"死频道"会让补扫直接
        跳过它, 这是问题4「曾靠近但未清除」型漏失的**直接原因**。
    """

    def __init__(self, ctx: _WorldCtx, cfg: SetCoverConfig) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.obs: Dict[int, List[Tuple[Vec, float]]] = {}     # ch -> [(pos, bearing)]
        self._cleared_ok: set = set()
        self._abandoned: set = set()
        self.alive: set = set()
        self.cleared = 0
        self._guard = 0
        self._scanned: set = {(0.0, 0.0)}
        self._pool: List[Vec] = build_pool(cfg)
        #: 收尾阶段 (锚点/射线) 额外清掉的源数 —— 仅供报告取数
        self.tail_found = 0
        self.tail_channels = 0
        self.hit_budget = False

    # ------------------------------------------------------------------ 基础设施
    def pos(self) -> Vec:
        """当前坐标: 走 ``ctx.position`` (机器狗航位推算)。"""
        return self.ctx.position

    def measure(self, x: float, y: float, ch: int) -> Dict[str, Any]:
        self._guard += 1
        r = self.ctx.measure(x, y, ch)
        mr = r.get("measure_result")
        if mr == "direction":
            self.obs.setdefault(ch, []).append(((x, y), r["svd_deg"]))
            # 只保留最近 6 条: 更早的射线在源已经逼近后基本没有价值,
            # 而射线扫描的代价随观测数线性增长。
            if len(self.obs[ch]) > 6:
                self.obs[ch].pop(0)
            self.alive.add(ch)
        elif mr == "near":
            self.alive.add(ch)
        return r

    def clear(self, x: float, y: float, ch: int) -> Dict[str, Any]:
        self._guard += 1
        return self.ctx.clear(x, y, ch)

    def last_bearing(self, ch: int) -> Tuple[Optional[Vec], Optional[float]]:
        for p, b in reversed(self.obs.get(ch, [])):
            return p, b
        return None, None

    # ------------------------------------------------------------------ 主流程
    def solve(self) -> int:
        try:
            return self._solve()
        except _OutOfBudget:
            # 虚拟时间超预算: 正常收尾而不是崩掉, 已清除的部分照常计入成绩。
            self.hit_budget = True
            return self.cleared

    def _solve(self) -> int:
        self.ctx.enter()

        # ---- 首轮扫描: 机械狗从原点出发, 原点覆盖半径 1000~1500 m,
        #      是单位时间发现率最高的一步, 因此 20 个频道一次测满。----
        for ch in range(1, N_CHANNELS + 1):
            r = self.measure(0.0, 0.0, ch)
            if r.get("measure_result") == "near":
                if self.clear(0.0, 0.0, ch).get("clear_result") == "success":
                    self._cleared_ok.add(ch)
                    self.cleared += 1

        # ---- 主循环 ----
        tries: Dict[int, int] = {}
        for _ in range(self.cfg.loop_rounds):
            if self._guard > self.cfg.guard_limit:
                break
            # 只主动逼近「未清除且未放弃」的频道
            pend = [c for c in sorted(self.alive)
                    if c not in self._cleared_ok and c not in self._abandoned]
            if not pend:
                if not self._rescan():
                    break
                continue

            cur = self.pos()
            info: Dict[int, Tuple[Vec, float]] = {}
            for c in pend:
                p, b = self.last_bearing(c)
                if p is not None:
                    info[c] = (p, b)
            if not info:
                break

            ch = min(info, key=lambda c: self._sched(c, cur, info[c]))
            if self._pursue(ch):
                self._cleared_ok.add(ch)
                self.cleared += 1
                tries.pop(ch, None)
            else:
                tries[ch] = tries.get(ch, 0) + 1
                if tries[ch] >= self.cfg.max_tries:
                    # 重试耗尽: 暂停主动逼近, 但**继续参与补扫测量**。
                    self._abandoned.add(ch)

        # ---- 收尾: 对「看见过但重试耗尽」的频道做最后补救 ----
        # 必须用 _cleared_ok 过滤而**不能**用"已放弃"过滤, 否则收尾一个都处理不到。
        if self.cfg.use_anchor_search:
            for ch in sorted(self.alive):
                if ch in self._cleared_ok or not self.obs.get(ch):
                    continue
                if ch in self._abandoned:
                    self.tail_channels += 1
                if self._multi_anchor_search(ch):
                    self._cleared_ok.add(ch)
                    self.cleared += 1
                    self.tail_found += 1

        self.ctx.exit()
        return self.cleared

    # ------------------------------------------------------------------ 启发式
    def _sched(self, ch: int, cur: Vec, ib: Tuple[Vec, float]) -> float:
        """调度打分: 距离为主, 兼顾「顺路程度」, 减少回头路。

        ``dev`` = 走向该频道最后可见点的方向 与 该点读到的示向度 之间的夹角:
        dev 小说明"顺着射线走过去"就是朝源的方向, 不会白跑。
        """
        p, b = ib
        d = math.hypot(p[0] - cur[0], p[1] - cur[1])
        to_b = math.degrees(math.atan2(p[1] - cur[1], p[0] - cur[0])) % 360
        dev = abs((to_b - b + 180.0) % 360.0 - 180.0)
        return d + self.cfg.sched_bearing_weight * dev

    def _pursue(self, ch: int) -> bool:
        """沿示向度按步长阶梯逼近并清除。"""
        ladder = self.cfg.step_ladder
        for k, step in enumerate(ladder):
            p, b = self.last_bearing(ch)
            if p is None:
                return False
            cur = self.pos()
            th = math.radians(b)
            tx = cur[0] + step * math.cos(th)
            ty = cur[1] + step * math.sin(th)

            # 目标在身后时, 先回到有信号的实测点附近, 避免折返跑
            to_p = math.degrees(math.atan2(p[1] - cur[1], p[0] - cur[0])) % 360
            if abs((to_p - b + 180.0) % 360.0 - 180.0) > 120.0:
                d_p = math.hypot(p[0] - cur[0], p[1] - cur[1])
                if d_p > 25.0:
                    L = min(step, d_p)
                    tx = cur[0] + (p[0] - cur[0]) / d_p * L
                    ty = cur[1] + (p[1] - cur[1]) / d_p * L

            r = self.measure(tx, ty, ch)
            mr = r.get("measure_result")

            if mr == "near":
                if self.clear(tx, ty, ch).get("clear_result") == "success":
                    return True
                return self._spiral(tx, ty, ch)

            if mr == "direction":
                if k == len(ladder) - 1:
                    if self.clear(tx, ty, ch).get("clear_result") == "success":
                        return True
                    return self._spiral(tx, ty, ch)
                continue

            # --- no_signal: 可能已越过, 也可能走出了定向源的扇区 ---
            # 先在"上一位置 → 新位置"的中点盲清一次: 步长 > 清除半径时,
            # 源常常正好落在这一步跨过去的那段路上。
            mx, my = (tx + cur[0]) / 2.0, (ty + cur[1]) / 2.0
            if self.clear(mx, my, ch).get("clear_result") == "success":
                return True
            m2 = self.measure(mx, my, ch)
            if m2.get("measure_result") == "near":
                if self.clear(mx, my, ch).get("clear_result") == "success":
                    return True
                return self._spiral(mx, my, ch)
            if m2.get("measure_result") == "direction":
                continue

            # 信号丢失。此时有两个可靠信息:
            #   (1) 曾测到信号的位置 p2 —— 源必在 p2 的接收半径内
            #   (2) 该处的示向度 b2 —— 源必在过 p2、方向 b2 的射线上
            p2, _b2 = self.last_bearing(ch)
            if p2 is not None and self._anchor_search(p2[0], p2[1], ch):
                return True
            if self.cfg.ray_on_fail and self._ray_sweep(ch):
                return True
            return False
        return False

    def _anchor_search(self, x: float, y: float, ch: int) -> bool:
        """以「最后可见点」为锚的环形盲清。

        源必然在该点的有效接收半径内且在该点的覆盖扇区内; 丢失信号只是因为我们
        走出了扇区。因此以该点为圆心做多半径环形 ``/clear`` 最直接,
        半径梯度覆盖"源距锚点有多远"的不确定性。
        """
        for radius in (40.0, 90.0, 160.0, 260.0):
            for k in range(12):
                th = math.radians(k * 30.0)
                if self.clear(x + radius * math.cos(th),
                              y + radius * math.sin(th),
                              ch).get("clear_result") == "success":
                    return True
        return False

    def _ray_sweep(self, ch: int) -> bool:
        """沿历史示向度射线做一维距离扫描并盲清 (收尾专用)。

        对每条历史观测 (p, b), 源必在过 p、方向 b 的射线上, 且距离不超过 p 处的
        有效接收半径。因此沿射线按距离采样即可命中; 每个采样点再做横向摆动,
        覆盖角误差在远距离的放大。与"以锚点为圆心的同心环"相比, 射线扫描把搜索
        空间由二维降为一维, 在源距锚点较远时效率高得多。
        """
        obs = self.obs.get(ch, [])
        if not obs:
            return False
        if self.cfg.ray_dist_step > 0.0:
            ladder: List[float] = []
            d = self.cfg.ray_dist_min
            while d <= self.cfg.ray_dist_max:
                ladder.append(d)
                d += self.cfg.ray_dist_step
        else:
            ladder = list(SC_RAY_LEGACY_LADDER)
        for (px, py), b in reversed(obs):
            th = math.radians(b)
            dx, dy = math.cos(th), math.sin(th)
            nx, ny = -dy, dx
            for dist in ladder:
                qx, qy = px + dist * dx, py + dist * dy
                if self.clear(qx, qy, ch).get("clear_result") == "success":
                    return True
                # 横向摆动覆盖角度误差在远距离上的放大。
                # 偏移量必须**小于清除半径 20 m**, 否则两侧采样会跨过源。
                off = dist * self.cfg.ray_lateral_rad
                for side in (-1.0, 1.0):
                    if self.clear(qx + side * off * nx,
                                  qy + side * off * ny,
                                  ch).get("clear_result") == "success":
                        return True
                # 角误差较小的情况: 再加一圈更细的偏移
                off2 = off * 0.5
                for side in (-1.0, 1.0):
                    if self.clear(qx + side * off2 * nx,
                                  qy + side * off2 * ny,
                                  ch).get("clear_result") == "success":
                        return True
        return False

    def _multi_anchor_search(self, ch: int) -> bool:
        """收尾: 射线扫描 + 锚点环形搜索, 二者互补。

        先射线扫描 (对"源距锚点远但方向准"有效), 再对最近两条观测各做一次
        锚点环形搜索 (对"源距锚点近"有效)。
        """
        if self._ray_sweep(ch):
            return True
        obs = self.obs.get(ch, [])
        for (px, py), _b in reversed(obs[:2]):
            if self._anchor_search(px, py, ch):
                return True
        return False

    def _spiral(self, x: float, y: float, ch: int) -> bool:
        """以 (x,y) 为中心的螺旋兜底搜索。

        半径上限 220 m 有据: 实测失败的定向源中位最近距离 30 m、P90 为 53 m,
        但仍有相当一部分落在 90 m 之外, 原上限 90 m 会让这些源永远够不到。
        """
        for radius in (10.0, 20.0, 35.0, 60.0, 90.0, 150.0, 220.0):
            for k in range(8):
                th = math.radians(k * 45.0)
                if self.clear(x + radius * math.cos(th),
                              y + radius * math.sin(th),
                              ch).get("clear_result") == "success":
                    return True
        return False

    # ------------------------------------------------------------------ 全区域补扫
    def _rescan(self) -> bool:
        """全区域补扫, 找被遮挡/超距而尚未发现的源。

        这一步是「清除率」的关键: 原实现实测省略它会把完全清除率从 95% 降到 13%。

        对问题4 (含定向源) 尤其重要: 定向源漏掉的主因是「从未靠近」——
        它在你当前位置处于覆盖扇区之外, 但换个位置就可见。覆盖扇区是 ±90°
        (半个平面), 绕着源走一圈必然能进入扇区, 因此用**贪心集合覆盖得到的
        24 点**在中半径上扫描 (见模块 docstring 手段 1)。
        """
        cur = self.pos()
        cand = [p for p in self._pool if p not in self._scanned]
        cand.sort(key=lambda p: math.hypot(p[0] - cur[0], p[1] - cur[1]))

        barren = 0
        for tx, ty in cand:
            # 250 m 以内的点相对当前位置信息量太低 (原点首扫与后续动作已经覆盖),
            # 跳过可以省下大量几乎无收益的移动。
            if math.hypot(tx - cur[0], ty - cur[1]) < 250.0:
                continue
            self._scanned.add((tx, ty))
            found = False
            for ch in range(1, N_CHANNELS + 1):
                # 只跳过「已确认清除」的频道。重试耗尽的频道必须继续测量:
                # 它只是暂时没找到逼近路径, 换个位置很可能重新可见。
                if ch in self._cleared_ok:
                    continue
                r = self.measure(tx, ty, ch)
                mr = r.get("measure_result")
                if mr == "direction":
                    found = True
                    self._abandoned.discard(ch)   # 复活, 重新纳入主动逼近
                elif mr == "near":
                    found = True
                    self._abandoned.discard(ch)
                    if self.clear(tx, ty, ch).get("clear_result") == "success":
                        self._cleared_ok.add(ch)
                        self.cleared += 1
            if found:
                return True
            barren += 1
            if self.cfg.rescan_k and barren >= self.cfg.rescan_k:
                break
        return False


def run_sweeper4(world: World, cfg: Optional[SetCoverConfig] = None, trace: bool = False,
                 real_left: Optional[Any] = None, reserve_s: float = 20.0,
                 knows_total: bool = True) -> SweepResult:
    """问题4 集合覆盖方案主入口。

    ``knows_total`` **被忽略**: 本方案从不使用真值总数, 也不知道自己是否清完 ——
    它靠"全区域补扫一轮无新发现"收手 (原实现的设计如此, 因此进程内与部署两个口径
    对它天然是同一个数)。签名与其它规划器保持一致, 以便统一评测工具复用。

    ``real_left`` / ``reserve_s`` 未使用: 本方案不看现实时间, 只受 ``max_virtual``
    与动作数护栏约束。
    """
    cfg = cfg or build_cfg()
    ctx = _WorldCtx(world, cfg)
    solver = SetCoverSolver(ctx, cfg)
    solver.solve()
    out = SweepResult(world)
    out.steps = ctx.actions
    out.probes = 0
    out.sweeps = 0
    out.pursues = 0
    out.mv_probe = 0.0
    if trace:
        print("  SET-COVER: cleared=%d/%d actions=%d tail+%d budget_hit=%s"
              % (world.cleared_count, world.n_sources, ctx.actions,
                 solver.tail_found, solver.hit_budget), flush=True)
    # 遥测: 与其它规划器的收尾计数对齐, 供统一对比表取数
    world.q4_hunt_found = int(solver.tail_found)
    world.q4_hunt_anchors = int(solver.tail_channels)
    world.q4_sc_actions = int(ctx.actions)
    world.q4_sc_budget_hit = bool(solver.hit_budget)
    return out


def run_candidate(world: World, cfg: Optional[SetCoverConfig] = None, trace: bool = False,
                  real_left: Optional[Any] = None, reserve_s: float = 20.0,
                  knows_total: bool = True) -> SweepResult:
    """``run_sweeper4`` 的别名 (评测工具的候选发现顺序会用到)。"""
    return run_sweeper4(world, cfg, trace=trace, real_left=real_left,
                        reserve_s=reserve_s, knows_total=knows_total)


def run_case(seed: int, cfg: Optional[SetCoverConfig] = None,
             trace: bool = False) -> Dict[str, Any]:
    """跑一局, 返回与 ``eval_q4`` 同口径的一行统计 (含定向源明细)。"""
    cfg = cfg or build_cfg()
    w = World(seed=seed, problem=4, **world_kwargs())
    run_sweeper4(w, cfg, trace=trace)
    n_dir = sum(1 for s in w.sources if not s.is_omni)
    cleared_dir = sum(1 for s in w.sources if not s.is_omni and not s.alive)
    return {
        "seed": seed, "cleared": w.cleared_count, "total": w.n_sources,
        "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
        "clears": w.clears, "failed_clears": w.failed_clears,
        "n_directed": n_dir, "cleared_directed": cleared_dir,
        "hunt_found": getattr(w, "q4_hunt_found", 0),
        "hunt_anchors": getattr(w, "q4_hunt_anchors", 0),
        "actions": getattr(w, "q4_sc_actions", 0),
        "budget_hit": getattr(w, "q4_sc_budget_hit", False),
    }


def _main(argv: Optional[List[str]] = None) -> int:
    """独立入口: ``python -m robotdog.solver.sweeper4 --seeds 9500-9549``。"""
    import argparse
    import json
    import os
    import statistics as st
    import time

    ap = argparse.ArgumentParser(description="问题4 集合覆盖 + 锚点射线 评测")
    ap.add_argument("--seeds", default="9500-9549")
    ap.add_argument("--max-virtual", type=float, default=90000.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    seeds: List[int] = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))

    cfg = build_cfg(max_virtual=a.max_virtual)
    t0 = time.time()
    rows = [run_case(sd, cfg) for sd in seeds]
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    print("=== 集合覆盖 + 锚点射线 | %d 局 ===" % len(rows))
    print("  清除比例 %d/%d = %.4f   全清 %d/%d"
          % (clr, tot, clr / max(tot, 1),
             sum(1 for r in rows if r["cleared"] == r["total"]), len(rows)))
    print("  平均定位清除 %.1f s/源   中位 %.1f"
          % (sum(r["virtual_s"] for r in rows) / max(clr, 1),
             st.median(r["virtual_s"] / max(r["cleared"], 1) for r in rows)))
    print("  定向源清除 %d/%d = %.2f%%   收尾额外清掉 %d 个 (%.2f/局)"
          % (sum(r["cleared_directed"] for r in rows), sum(r["n_directed"] for r in rows),
             100.0 * sum(r["cleared_directed"] for r in rows) / max(sum(r["n_directed"] for r in rows), 1),
             sum(r["hunt_found"] for r in rows), sum(r["hunt_found"] for r in rows) / len(rows)))
    print("  每局 移动 %.0f m   检测 %.1f 次   触碰虚拟上限 %d 局"
          % (sum(r["moved_m"] for r in rows) / len(rows),
             sum(r["measures"] for r in rows) / len(rows),
             sum(1 for r in rows if r["budget_hit"])))
    print("  墙钟 %.1f s" % (time.time() - t0))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh, ensure_ascii=False, indent=1)
        print("明细 ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
