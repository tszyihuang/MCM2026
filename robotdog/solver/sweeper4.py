"""问题4（含定向源）发布策略: 测量经济学 + 中继测量 + 在线路线重优化。

本模块就是 ``python -m robotdog.solver.deploy --problem 4`` 与
``tools/eval_q4.py --planners sweeper4`` **实际执行**的策略。决策层
（:data:`PARAMS` + :func:`strategy` + :class:`_Solver`）与单文件版
``Desktop/solver_q4.py`` 逐字一致；文件下半部分是主工程适配层
（``build_cfg`` / ``world_kwargs`` / ``_Ctx`` / ``run_sweeper4``），只做
“配置对象 / 环境装配 / 统一入口”三件事，不改变任何决策。

策略要点（相对基线 agent5 只换了“任务排序”这一层，外加中继测量）:
基线用“贪心最近任务”（纯 NN）挑下一个任务，本版把「剩余站点」+
「已可靠交会定位的待清目标」放进**同一条路线**，并用 2-opt / Or-opt /
cheapest-insertion 在线重优化；中继测量把“第 2 条示向度”顺路插在腿内
（``|A−M|+|M−B| = |A−B|``，零额外移动），让交会定位提前。

代价模型（“米本位”，沿用基线）：
  /measure(x,y,c) = |Δp|/5 + [c≠当前频道]×1 + 5   → 一次检测 ≈ 30 m 路程
  /clear(x,y,c)   = |Δp|/5 + (5 成功 / 3 失败)     → 一次失败盲清 ≈ 15 m 路程

离线审计（``analyze.py``，seeds 1..150）说明基线 23.6 km 的去向：
  17 860 m  24 站最优回路（站点骨架，刚性）
   1 917 m  站点**拜访顺序**浪费（贪心 NN 的拜访序 19 777 m vs 最优 17 860 m）
   1 855 m  拜访全部源所需的最小绕行（离线理想）
   1 977 m  清除搜索的额外绕行
本模块吃回的就是那 1 917 m 顺序浪费（以及部分插入浪费）。

发布成绩（**16 站**布局，部署口径 ``knows_total=False``，明细见 ``README.md`` 与
``data/reports/q4_ship16_*.json``）：
  * seed 9500-9999（500 局）     清除比例 0.9992，424.9 s/源
  * seed 20000-20999（1000 局，未参与标定） 0.9992，422.3 s/源
  * seed 30000-30999 + 40000-40999（2000 局） 0.9992，421.6 s/源

布局取舍（``README.md``「精度 ⇄ 速度」一节，3500 局实测）：16 站比 22 站快 6.3%，
代价是 0.08% 的源（≈1% 的局）漏检。22 站（``ring_def`` = 985@7 + 1850@14）满足
"最坏角隙 ≤ 180°"的**零漏检几何判据**（178.99°），16 站不满足（360°）——
两者 3 局平均清除比例跌破 98% 的概率分别是 0% 与 3.1%。
上一版（18 站环骨架，0.9994 / 461.9 s）留档为 ``sweeper4_ring18.py``。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .world import World

# ===========================================================================
# 可调参数
# ===========================================================================
PARAMS = {
    # ---- 布局 ----
    'ring_spec': ((0, 1), (970.0, 8), (1850.0, 15)),
    'ring_phase': 1.0,         # 每环固定相位偏移（单位 π/3）
    #: 显式环定义 [[半径, 点数, 相位(度)], ...]；给出时覆盖 ring_spec/ring_phase
    #: 定稿的 16 站 = 圆心 + 985@5 + 1850@10（外环相对内环相位 +4.286°）—— 这是**速度档**：
    #: 3500 局实测 0.9992 / 421.6 s/源，比 22 站档快 6.3%。
    #: 代价：最坏角隙 360°（22 站档才是 178.99° ≤ 180° 的可证零漏检），
    #: 抽样 26 万个源里 236 个（0.091%）"任何站点都看不见"，3 局测试跌破 98% 的概率 3.1%。
    #: 22 站零漏检档 = ((0, 1, 0.0), (985.0, 7, 0.0), (1850.0, 14, 4.286))，本行一行之差。
    #: 取舍见 README「精度 ⇄ 速度」一节。
    'ring_def': ((0, 1, 0.0), (985.0, 5, 0.0), (1850.0, 10, 4.286)),
    'grid_d': 0.0,             # >0 时改为用六边形栅格（与 ring_spec 叠加）
    'grid_rmax': 0.0,
    'ring_r': 0.0,             # 兼容旧参数
    'ring_n': 0,
    # ---- 每站测哪些频道 ----
    'max_chan_station': 20,
    'min_obs': 2,              # 攒够几条示向度才算"解决"（>=2 才能交会）
    'fix_sigma_ok': 50.0,      # 认为"已定位"的误差阈值 (m)
    'zero_inner': 99,          # 前 k 个内层站点测"零观测"频道（99=全部都测）
    'zero_ring_stride': 1,     # 外圈每 stride 个点测一次"零观测"频道
    'ring_outer_min': 1500.0,  # 半径 >= 此值的环算作"外圈"
    # ---- 任务调度 ----
    'route_mode': 'nearest',   # nearest | tour（planner='A' 时生效）
    'lookahead': 0.0,
    'sta_bonus': 0.9,          # 站点距离权重（<1 更倾向先去站点）
    'opp_max': 1e18,           # 巡游期允许的顺路清除半径（1e18=总是允许）
    'opp_try': 2,              # 同一源最多尝试清除几次（防止反复扑空）
    'enroute_cap': 0,          # 清除点顺带测几个活跃频道
    'early_k': 99,             # 探测到 k 个频道后提前收工（>=10 才有意义）
    # ---- 在线路线重优化（新增）----
    'planner': 'E',            # A=基线贪心最近任务；B/C/D/E 见 _plan_route
    'plan_ls': 3,              # 0=NN 1=+2opt 2=+Or-opt(1..3) 3=+Or-opt(1..5) 4=多起点
    'plan_stability': 1,       # 1=保留上一版路线顺序，新节点用 cheapest insertion
    'plan_urgent_r': 0.0,      # >0：目标离当前位置小于此值就直接插到队首
    'plan_seed': 'nn',         # 首版路线的起手：nn | tour | tour_rev
    # ---- 「物理上不可能应答」的补测剪枝（只作用于已发现频道）----
    'skip_far': 1500.0,        # >0 启用；≤此距离内才补测（源接收半径上限 = 1500）
    'skip_far_k': 2.5,         # 估计点还要按 k·sigma 放余量
    # ---- 可行域（示向度楔形交集）剪枝：严格可证明安全 ----
    'region_prune': 1,         # 1=启用
    'region_disk': 1,          # 1=同时用"接收半径 ≤1500"的圆盘裁剪
    'region_margin': 1.05 * math.pi / 180.0,
    'ray_prune': 1,            # 1=启用"定向源被包围即不可能"的射线排除
    'tri_diam': 2600.0,        # 三角形直径上限（仅用于限制枚举量）
    # ---- 计数约束早停 ----
    'count_stop': 1,           # 1=启用
    'source_max': 16,          # 源总数上界
    # ---- 中继测量：在"当前腿的中点"补测，移动成本为 0 ----
    'relay': 1,                # 1=启用
    'relay_sin': 0.08,         # 交角阈值 |sinψ|
    'relay_cap': 8,            # 每条腿最多补测几个频道
    'relay_min_leg': 150.0,    # 腿太短就不补测
    'relay_reach': 900.0,      # 中点到"可行源域"的距离上限
    'relay_tref': 900.0,       # 源距离的参考值（用于估交角）
    'relay_poor': 0,           # 1=对"定位差(σ>fix_sigma_ok)"的频道也做中继
    'relay_maxobs': 1,         # 允许中继的频道最大已有示向度条数
    'relay_retry': 1,          # 1=中继未换来定位时允许下条腿重试
    'relay_rank': 'dist',      # sin=按交角排 | dist=按"中点到可行源域距离"排（命中率优先）
    'relay_point': 'best',     # mid=腿中点 | best=腿上"离可行射线最近"的点（同样零移动）
    'relay_multi': 1,          # 1=中继成功后继续在同一腿多做一个（成功=替换，边际成本≈0）
    # ---- 中继测量 v2：后验预测命中概率 + 腿上自由选点（自 c2 移植；**定稿默认**）----
    'relay_mode': 2,           # 1=c4 几何版（reach+sin+best point）；2=后验模型版
    'relay_pmin': 0.40,        # 预测命中概率下限（低于此不测）
    'relay_pts': 5,            # 腿上候选点数（严格在腿内部，含中点）
    'relay_keep': 8,           # 每条腿最多补测几个频道
    'relay_omni': 0.3,         # 先验全向源占比（用于定向覆盖因子）
    'relay_pre': 600.0,        # 预筛：腿段到可行射线段的距离上限（c2 定稿值；原 1500 恒真）
    'relay_sin2': 0.08,        # v2 的硬交角下限 |sinψ|（= c4 的 relay_sin）
    'relay_qmode': 1,          # 0=硬下限；1=按 min(1, sinψ/sin2) 折算分数
    'relay_tmean': 0.0,        # 源距离后验均值（0=自动算）
    'relay_stop': 0,           # 1=本腿拿到一个定位就停
    'relay_adv_min': 0.0,      # >0：只中继"提前量 >= 此米数"的频道
    'relay_adv_w': 0,          # 1=按 p·提前量 排序；0=只按 p 排序
    'relay_adv_scan': 14,      # 沿路线向后最多看多少个节点
    'relay_value': 12.0,       # 一次中继"命中并交会"折算的秒数（用于权衡绕路）
    'relay_off_max': 0.0,      # >0：中继点允许偏离腿的垂直距离上限 (m)
    'relay_pmin2': 0.0,        # >0：同一条腿上第 2 个及以后频道要更高的命中概率
    # ---- 逼近与清除 ----
    'creep_step': 36.0,
    'creep_max_it': 10,
    'ray_step': 26.0,          # 沿射线盲清步长（保证 < 20m 覆盖）
    'ray_max_it': 40,
    'ray_find_it': 62,
    'ray_from_o': 1,           # 档 2：先走到观测点再沿射线走（保底）
    'ray_recheck': 1,          # 1=进点后立刻测一次向（防止"进点已越过源"时照直走）
    'sweep_gap': 26.0,
    'sweep_rmin': 60.0,
    'sweep_rmax': 300.0,
    'sweep_grow': 1.7,
    'sweep_cap': 40,
    'sweep_cap_max': 110,
    # ---- 单观测兜底 ----
    'hunt_mode': 'perp',       # ray | perp | both
    'hunt_probe': 450.0,
}

ARENA_R = 1800.0
CH_ALL = tuple(range(1, 21))
SOURCE_MAX = 16

#: 诊断计数器（只写不读，不参与决策）
STATS = {}


def _bump(k, v=1.0):
    STATS[k] = STATS.get(k, 0.0) + v


def _hyp(a, b):
    return math.sqrt(a * a + b * b)


# ===========================================================================
# 中继测量的后验命中概率模型（自 c2 移植）
# ---------------------------------------------------------------------------
# 第 1 条示向度到达时，源 S 必在射线 O + t·u 上。给定"在 O 测到信号"这一事件，
# 沿射线的后验密度为
#       p(t) ∝ t · q(t),    q(t) = P(R_s ≥ t),  R_s ~ U[1000, 1500]
# （t 来自场地内面元密度；q 来自"接收半径必须够得着 O"）。
# 若在腿上的 M 点再测一次，两点都收到信号要求同一个 R_s ≥ max(t, |M−S|)，
# 于是
#       P(命中 | M) = Σ_k t_k·q(max(t_k, d_k))·cf_k  /  Σ_k t_k·q(t_k)
# 其中 cf_k 是"定向源同时覆盖 O 与 M"的先验概率
#       cf = w_omni + (1−w_omni)·max(0, 1 − ∠(S→O, S→M)/180°).
# 这个量同时给出（a）该不该测（阈值）与（b）腿上的哪一点最好（选点）。
# ===========================================================================
RX_LO = 1000.0
RX_HI = 1500.0
_TGRID = (150.0, 400.0, 620.0, 800.0, 950.0, 1060.0, 1160.0, 1260.0,
          1360.0, 1450.0)


def _q(t):
    if t <= RX_LO:
        return 1.0
    if t >= RX_HI:
        return 0.0
    return (RX_HI - t) / (RX_HI - RX_LO)


def _seg_dist(px, py, ax, ay, bx, by):
    """点 (px,py) 到线段 AB 的距离。"""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return _hyp(px - ax, py - ay)
    s = ((px - ax) * vx + (py - ay) * vy) / L2
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    return _hyp(px - ax - s * vx, py - ay - s * vy)


def _closest_on_seg(ax, ay, bx, by, px, py):
    """线段 AB 上离 (px,py) 最近的点。"""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 <= 1e-12:
        return ax, ay
    s = ((px - ax) * vx + (py - ay) * vy) / L2
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    return ax + s * vx, ay + s * vy


_WSUM = sum(t * _q(t) for t in _TGRID)


def _post_mean(n=1500):
    """后验 p(t) ∝ t·q(t) 的均值（源沿射线的期望距离 ≈ 855 m）。"""
    num = den = 0.0
    dt = 1500.0 / n
    for i in range(n):
        t = (i + 0.5) * dt
        p = t * _q(t)
        num += p * t
        den += p
    return num / den if den > 0 else 800.0


T_MEAN = _post_mean()


# ===========================================================================
# 可行域几何：所有"已发现频道"的源位置一定落在一个可证明的凸区域里
#   R(c) = 场地 ⊇ 由每条示向度 (±1° 楔形) 与"接收半径 ≤1500"圆盘交出来的集合
# 对任意站点 P：min_{X∈R} |X−P| > 1500  ⇒  该站物理上不可能收到 c 的信号，
# 这一次 /measure 必然 no_signal，可以安全跳过。
#
# 与基线 skip_far 的差别：基线对"已交会"的频道用「估计点 ± k·σ」，σ 大时
# （交角差、或只剩 1 条射线导致 σ 达数百米）判据几乎恒不触发；本实现直接
# 用**示向度楔形的交集**，对 2 条及以上的示向度都能给出一个很窄的条带。
# 所有近似一律取**外近似**（超集），因此判据是保守的、可证明安全的。
# ===========================================================================
_WEDGE_HALF = 1.05 * math.pi / 180.0       # 读数 ±1° + 两位小数舍入余量
_ARENA_N = 16                               # 场地外接正 16 边形
_ARENA_RC = ARENA_R / math.cos(math.pi / _ARENA_N)
_DISK_N = 16                                # 接收半径圆盘的外接正 16 边形
_DISK_RC = 1500.0 / math.cos(math.pi / _DISK_N)


def _arena_poly():
    return [(_ARENA_RC * math.cos(2.0 * math.pi * k / _ARENA_N),
             _ARENA_RC * math.sin(2.0 * math.pi * k / _ARENA_N))
            for k in range(_ARENA_N)]


def _clip_hp(poly, a, b, cc):
    """裁剪凸多边形：保留 a·x + b·y >= cc 的一侧。"""
    n = len(poly)
    if n == 0:
        return poly
    out = []
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        d1 = a * x1 + b * y1 - cc
        d2 = a * x2 + b * y2 - cc
        if d1 >= 0.0:
            out.append((x1, y1))
        if (d1 >= 0.0) != (d2 >= 0.0):
            t = d1 / (d1 - d2)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return out


def _pt_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L = dx * dx + dy * dy
    if L < 1e-12:
        return _hyp(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / L
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return _hyp(px - x1 - t * dx, py - y1 - t * dy)


def _hp_t_interval(nx, ny, cc, ox, oy, ux, uy, tmax):
    """射线 X(t)=O+t·d 上满足 n·(X−…) >= cc 的 t 区间（与 [0,tmax] 相交）。"""
    k = nx * ux + ny * uy
    rhs = cc - (nx * ox + ny * oy)
    if k > 1e-12:
        return (rhs / k, tmax)
    if k < -1e-12:
        return (0.0, rhs / k)
    return (0.0, tmax) if rhs <= 0.0 else None


def _ray_in_shrunk_tri(ox, oy, ux, uy, tri, m, tmax):
    """射线落在"必无源"区域里的 t 区间。

    判据（可证明）：若源 X 落在三个 no_signal 站 v1v2v3 构成的三角形**内部**，
    且 X 到三站都 <=1000 m（<= rx 下界），则三站都必须"定向背对 X"才能
    同时 no_signal；但 X 在三角形内 ⟹ 从 X 看三站的方向不被任何开半平面包含
    ⟹ 不存在同时背对三站的 u ⟹ 矛盾。故该三角形内部必无源。
    """
    (ax, ay), (bx, by), (cx, cy) = tri
    if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0.0:
        (bx, by), (cx, cy) = (cx, cy), (bx, by)
    lo, hi = 0.0, tmax
    for (px, py, qx, qy) in ((ax, ay, bx, by), (bx, by, cx, cy),
                             (cx, cy, ax, ay)):
        ex, ey = qx - px, qy - py
        L = math.hypot(ex, ey)
        if L < 1e-9:
            return None
        nx, ny = -ey / L, ex / L          # CCW 内侧法向
        iv = _hp_t_interval(nx, ny, nx * px + ny * py + m,
                            ox, oy, ux, uy, tmax)
        if iv is None:
            return None
        if iv[0] > lo:
            lo = iv[0]
        if iv[1] < hi:
            hi = iv[1]
        if hi <= lo:
            return None
    # 还必须三个顶点都在 <=1000−m 之内（rx 下界 1000）
    rr = (1000.0 - m) ** 2
    for (px, py) in ((ax, ay), (bx, by), (cx, cy)):
        vx, vy = ox - px, oy - py
        B = ux * vx + uy * vy
        C = vx * vx + vy * vy - rr
        disc = B * B - C
        if disc <= 0.0:
            return None
        s = math.sqrt(disc)
        if -B + s < lo or -B - s > hi:
            return None
        if -B - s > lo:
            lo = -B - s
        if -B + s < hi:
            hi = -B + s
    return (lo, hi) if hi > lo else None


def _pt_poly_dist(px, py, poly):
    n = len(poly)
    if n == 0:
        return float('inf')
    if n == 1:
        return _hyp(px - poly[0][0], py - poly[0][1])
    if n == 2:
        return _pt_seg_dist(px, py, poly[0][0], poly[0][1],
                            poly[1][0], poly[1][1])
    inside = True
    best = float('inf')
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) < 0.0:
            inside = False
            break
    if inside:
        return 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        d = _pt_seg_dist(px, py, x1, y1, x2, y2)
        if d < best:
            best = d
    return best


def _prj(x, y, rmax=1860.0):
    d = _hyp(x, y)
    if d > rmax:
        k = rmax / d
        return x * k, y * k
    return x, y


def strategy(ctx, scenario=None):
    STATS.clear()
    try:
        _Solver(ctx, scenario).run()
    except Exception:
        try:
            ctx.exit()
        except Exception:
            pass


class _Solver:
    def __init__(self, ctx, scenario=None):
        self.ctx = ctx
        self.phase = 'init'
        self.pos = (0.0, 0.0)
        self.chan = 1
        self.obs = {}        # ch -> [(px,py,theta)]
        self.nos = {}        # ch -> [(px,py)]
        self.cleared = set()
        self.fix = {}        # ch -> (x,y,sigma)
        self.seen = set()    # 曾探测到信号的频道
        self.attempt = {}    # ch -> 已尝试清除次数（避免反复扑空）
        self.route = []      # 当前路线（node 列表）
        self.relay_done = set()   # 中继测量已试过的频道（每个只试一次）
        self.relay_fixed = set()  # 靠中继测量完成交会的频道（仅诊断）
        self._visit_no = -1
        self.where = 'other'
        self._reg_cache = {}
        self._tri_cache = {}
        self._feas_cache = {}
        if scenario is not None:
            self.chan = int(getattr(scenario, 'initial_channel', 1) or 1)
            ip = getattr(scenario, 'initial_position', None)
            if ip is not None:
                self.pos = (float(ip[0]), float(ip[1]))

    # ---------------- 基本动作 ----------------
    def do_measure(self, c, x, y):
        d = _hyp(x - self.pos[0], y - self.pos[1])
        _bump('mv_' + self.phase, d)
        _bump('n_' + self.phase)
        # ---- 微观经济学诊断：测量分类 + 时间拆分 ----
        sw = 1.0 if c != self.chan else 0.0
        _bump('t_sw', sw)
        _bump('t_mv_m', d / 5.0)
        _bump('t_act_m', 5.0)
        blank = c not in self.seen
        nobs0 = len(self.obs.get(c, ()))
        hasfix = c in self.fix
        W = self.where
        _bump('w_n_' + W)
        _bump('w_t_' + W, d / 5.0 + sw + 5.0)
        r = self.ctx.measure(x, y, c)
        self.chan = c
        self.pos = (float(x), float(y))
        res = r.get('measure_result')
        if res == 'direction':
            self.obs.setdefault(c, []).append((float(x), float(y),
                                               float(r['svd_deg'])))
            self.seen.add(c)
            if blank:
                _bump('m_blank_live')
            elif nobs0 == 0:
                _bump('m_new')
            else:
                _bump('m_add')
        elif res == 'near':
            self.seen.add(c)
            _bump('m_blank_live' if blank else 'm_near')
        else:
            self.nos.setdefault(c, []).append((float(x), float(y)))
            _bump('m_dead')
            _bump('m_dead_blank' if blank else 'm_dead_live')
            if not blank:
                _bump('w_dead_' + W)
                _bump('m_dead_live_fix' if hasfix else 'm_dead_live_1obs')
                if W == 'sta':
                    nb = 1 if nobs0 <= 1 else (2 if nobs0 == 2 else 3)
                    _bump('dl_sta_n%d' % nb)
                    fx = self.fix.get(c)
                    if fx is not None:
                        sg = fx[2]
                        _bump('dl_sig_%d' % (0 if sg <= 70 else
                                             (1 if sg <= 150 else
                                              (2 if sg <= 400 else 3))))
                    ro = self._region(c)
                    if ro:
                        _bump('dl_regmin_%d' % min(3, int(
                            _pt_poly_dist(x, y, ro) // 500.0)))
            _bump('t_dead', d / 5.0 + sw + 5.0)
            _bump('td_mv', d / 5.0)
        if blank:
            _bump('m_blank')
            _bump('t_blank', d / 5.0 + sw + 5.0)
            _bump('m_blank_' + self.phase)
            if res != 'no_signal':
                _bump('disc_at_%d' % self._visit_no)
        else:
            _bump('m_live')
            _bump('m_live_' + self.phase)
            if nobs0 >= 2:
                _bump('m_add_red')
        return res, r

    def try_clear(self, c, x, y):
        d = _hyp(x - self.pos[0], y - self.pos[1])
        _bump('mv_' + self.phase, d)
        _bump('c_' + self.phase)
        _bump('t_mv_c', d / 5.0)
        r = self.ctx.clear(x, y, c)
        self.pos = (float(x), float(y))
        if r.get('clear_result') == 'success':
            self.cleared.add(c)
            self.fix.pop(c, None)
            _bump('cok_' + self.phase)
            _bump('t_act_c', 5.0)
            return True
        _bump('t_act_c', 3.0)
        return False

    # ---------------- 定位 ----------------
    def compute_fix(self, c):
        obs = self.obs.get(c)
        if not obs or len(obs) < 2:
            return None
        sx = sy = 0.0
        n = 0
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                p = self._cross(obs[i], obs[j])
                if p is not None:
                    sx += p[0]
                    sy += p[1]
                    n += 1
        if n == 0:
            return None
        x, y = sx / n, sy / n
        sig_th = math.radians(1.0)
        a00 = a01 = a11 = b0 = b1 = 0.0
        for _ in range(5):
            a00 = a01 = a11 = b0 = b1 = 0.0
            for (px, py, th) in obs:
                t = math.radians(th)
                nx, ny = -math.sin(t), math.cos(t)
                d = _hyp(x - px, y - py)
                if d < 1.0:
                    d = 1.0
                w = 1.0 / d
                cc = nx * px + ny * py
                a00 += w * w * nx * nx
                a01 += w * w * nx * ny
                a11 += w * w * ny * ny
                b0 += w * w * nx * cc
                b1 += w * w * ny * cc
            det = a00 * a11 - a01 * a01
            if abs(det) < 1e-18:
                return None
            x = (b0 * a11 - b1 * a01) / det
            y = (a00 * b1 - a01 * b0) / det
        det = a00 * a11 - a01 * a01
        cxx = a11 / det
        cyy = a00 / det
        lam = 0.5 * (cxx + cyy) + math.sqrt(0.25 * (cxx - cyy) ** 2
                                            + (a01 / det) ** 2)
        sigma = sig_th * math.sqrt(max(lam, 0.0))
        x, y = _prj(x, y)
        return (x, y, sigma)

    @staticmethod
    def _cross(o1, o2):
        p1x, p1y, a1 = o1
        p2x, p2y, a2 = o2
        t1 = math.radians(a1)
        t2 = math.radians(a2)
        d1x, d1y = math.cos(t1), math.sin(t1)
        d2x, d2y = math.cos(t2), math.sin(t2)
        den = d1x * d2y - d1y * d2x
        if abs(den) < 1e-9:
            return None
        dx, dy = p2x - p1x, p2y - p1y
        t = (dx * d2y - dy * d2x) / den
        px, py = p1x + t * d1x, p1y + t * d1y
        if t < 0.0 or t > 4000.0:
            return None
        u = (px - p2x) * d2x + (py - p2y) * d2y
        if u < 0.0 or u > 4000.0:
            return None
        return (px, py)

    def update_fixes(self):
        for c in CH_ALL:
            if c in self.cleared:
                continue
            f = self.compute_fix(c)
            if f is not None:
                self.fix[c] = f

    # ---------------- 布局 ----------------
    def make_plan(self):
        P = PARAMS
        pts = []
        outer_min = P.get('ring_outer_min', 1500.0)
        rd = P.get('ring_def')
        if rd:
            for item in rd:
                r, n = float(item[0]), int(item[1])
                phd = float(item[2]) if len(item) > 2 and item[2] is not None \
                    else 0.0
                if r <= 1e-9:
                    pts.append([0.0, 0.0, 'inner', -1])
                    continue
                kind = 'ring' if r >= outer_min else 'inner'
                off = math.radians(phd)
                for k in range(n):
                    a = off + 2.0 * math.pi * k / n
                    pts.append([r * math.cos(a), r * math.sin(a), kind, k])
            spec = ()
        else:
            spec = P.get('ring_spec') or ()
        ph = P.get('ring_phase', 0.0)
        for ri, (r, n) in enumerate(spec):
            if r <= 1e-9:
                pts.append([0.0, 0.0, 'inner', -1])
                continue
            kind = 'ring' if r >= outer_min else 'inner'
            n = int(n)
            off = ph * ri * math.pi / 3.0      # 每环一个固定相位偏移
            for k in range(n):
                a = off + 2.0 * math.pi * k / n
                pts.append([r * math.cos(a), r * math.sin(a), kind, k])
        d = P['grid_d']
        rmax = P['grid_rmax']
        if d > 0 and rmax > 0:
            dy = d * math.sqrt(3.0) / 2.0
            jmax = int(rmax / dy) + 2
            imax = int(rmax / d) + 2
            for j in range(-jmax, jmax + 1):
                y = j * dy
                off = (d * 0.5) if (j % 2) else 0.0
                for i in range(-imax, imax + 1):
                    x = i * d + off
                    if _hyp(x, y) <= rmax + 1e-9:
                        pts.append([x, y, 'inner', -1])
        if P['ring_r'] > 0 and P['ring_n'] > 0:
            n = int(P['ring_n'])
            for k in range(n):
                a = 2 * math.pi * k / n
                pts.append([P['ring_r'] * math.cos(a),
                            P['ring_r'] * math.sin(a), 'ring', k])
        # 去重
        seen = set()
        out = []
        for p in pts:
            key = (round(p[0], 6), round(p[1], 6))
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        if P['route_mode'] == 'tour' and P.get('planner') == 'A':
            return self._tsp_order(out)
        return out

    @staticmethod
    def _tsp_order(pts, restarts=8):
        """多起点 NN + 2-opt + Or-opt 的开链 TSP（起点固定为原点，终点自由）。"""
        n = len(pts)
        if n <= 1:
            return list(pts)
        best_l, best_o = None, None
        seen0 = set()
        for s0 in range(n):
            r0 = round(_hyp(pts[s0][0], pts[s0][1]), 3)
            if r0 in seen0:
                continue
            seen0.add(r0)
            rem = list(range(n))
            rem.remove(s0)
            cur = s0
            order = [s0]
            while rem:
                k = min(rem, key=lambda i: (pts[i][0] - pts[cur][0]) ** 2
                        + (pts[i][1] - pts[cur][1]) ** 2)
                rem.remove(k)
                order.append(k)
                cur = k
            order = _Solver._ls_order(order, [tuple(p[:2]) for p in pts],
                                      (0.0, 0.0), 3)
            ln = _Solver._order_len(order, [tuple(p[:2]) for p in pts],
                                    (0.0, 0.0))
            if best_l is None or ln < best_l - 1e-9:
                best_l, best_o = ln, order
            if len(seen0) >= restarts:
                break
        return [pts[i] for i in best_o]

    # ---- 通用开链 TSP 局部搜索（node 用坐标列表表示）----
    @staticmethod
    def _order_len(order, pts, start):
        cur = start
        tot = 0.0
        for i in order:
            p = pts[i]
            tot += _hyp(p[0] - cur[0], p[1] - cur[1])
            cur = p
        return tot

    @staticmethod
    def _ls_order(order, pts, start, depth=2):
        """depth: 1=2-opt, 2=+Or-opt(1..3), 3=+Or-opt(1..5)。"""
        order = list(order)
        n = len(order)
        if n < 3:
            return order
        # ---- 2-opt ----
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                a = start if i == 0 else pts[order[i - 1]]
                b = pts[order[i]]
                for j in range(i + 1, n - 1):
                    cc = pts[order[j]]
                    dd = pts[order[j + 1]]
                    old = _hyp(b[0] - a[0], b[1] - a[1]) + \
                        _hyp(dd[0] - cc[0], dd[1] - cc[1])
                    new = _hyp(cc[0] - a[0], cc[1] - a[1]) + \
                        _hyp(dd[0] - b[0], dd[1] - b[1])
                    if new < old - 1e-9:
                        order[i:j + 1] = order[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
        if depth < 2:
            return order
        # ---- Or-opt ----
        for Lmax in ((3, 5) if depth >= 3 else (3,)):
            improved = True
            while improved:
                improved = False
                pos = [start] + [pts[i] for i in order]
                for L in range(1, min(Lmax, n - 1) + 1):
                    for i in range(0, n - L + 1):
                        s1 = pos[i + 1]
                        sL = pos[i + L]
                        p = pos[i]
                        q = pos[i + L + 1] if i + L + 1 <= n else None
                        if q is None:
                            gain = _hyp(s1[0] - p[0], s1[1] - p[1])
                        else:
                            gain = (_hyp(s1[0] - p[0], s1[1] - p[1])
                                    + _hyp(q[0] - sL[0], q[1] - sL[1])
                                    - _hyp(q[0] - p[0], q[1] - p[1]))
                        if gain <= 1e-9:
                            continue
                        rpos = pos[:i + 1] + pos[i + L + 1:]
                        nr = n - L
                        bestk, bestc = None, None
                        for k in range(nr + 1):
                            if k == nr:
                                c2 = _hyp(rpos[nr][0] - s1[0],
                                          rpos[nr][1] - s1[1])
                            else:
                                u = rpos[k]
                                v = rpos[k + 1]
                                c2 = (_hyp(s1[0] - u[0], s1[1] - u[1])
                                      + _hyp(v[0] - sL[0], v[1] - sL[1])
                                      - _hyp(v[0] - u[0], v[1] - u[1]))
                            if bestc is None or c2 < bestc:
                                bestc, bestk = c2, k
                        if bestc is not None and bestc < gain - 1e-9:
                            seg = order[i:i + L]
                            rest = order[:i] + order[i + L:]
                            order = rest[:bestk] + seg + rest[bestk:]
                            improved = True
                            break
                    if improved:
                        break
        return order

    # ---------------- 频道选择 ----------------
    def channels_at(self, st):
        """返回在该站点值得测的频道列表。"""
        P = PARAMS
        _, _, kind, ridx = st
        if kind == 'inner' and self.inner_visited >= P['zero_inner']:
            allow_zero = False
        elif kind == 'ring':
            allow_zero = (ridx % max(1, int(P['zero_ring_stride'])) == 0)
        else:
            allow_zero = True
        sx, sy = st[0], st[1]
        sf = P.get('skip_far', 0.0)
        kk = P.get('skip_far_k', 2.5)
        rmode = int(P.get('region_prune', 0))
        out = []
        full_seen = (P.get('count_stop', 1)
                     and len(self.seen) >= int(P.get('source_max', SOURCE_MAX)))
        for c in CH_ALL:
            if c in self.cleared:
                continue
            # ---- 计数约束早停：源总数 <=16 ⇒ 已有 16 个频道应答过，
            #      其余频道可**证明**为空，不必再扫 ----
            if full_seen and c not in self.seen:
                _bump('cc_skip')
                continue
            f = self.fix.get(c)
            nobs = len(self.obs.get(c, ()))
            if f is not None:
                if f[2] <= P['fix_sigma_ok'] and nobs >= int(P['min_obs']):
                    continue
                # 源接收半径 <= 1500：站点离"估计点 ± k·σ"之外，物理上不可能应答
                if sf > 0.0 and _hyp(sx - f[0], sy - f[1]) > sf + kk * f[2]:
                    continue
            else:
                obs = self.obs.get(c)
                # 只有示向度没有定位点：源必在射线 [O, O+1500u] 上，
                # 站点到该线段的最短距离 > 1500 时同样不可能应答
                if sf > 0.0 and obs and self._ray_seg_dist(sx, sy, obs) > sf:
                    continue
            # ---- 可行域（示向度楔形交集）剪枝：严格可证明安全 ----
            if rmode and nobs:
                if nobs == 1 and P.get('ray_prune', 1) \
                        and self.ray_prune(sx, sy, c):
                    _bump('skip_ray')
                    continue
                dm = self.region_min_dist(sx, sy, c)
                if dm >= 0.0 and dm > sf:
                    _bump('skip_region')
                    continue
            if nobs == 0 and not allow_zero:
                continue
            out.append(c)
        lim = int(P['max_chan_station'])
        if len(out) > lim:
            out.sort(key=lambda c: (0 if self.obs.get(c) else 1, c))
            out = out[:lim]
        return out

    # ---------------- 可行域剪枝（新） ----------------
    def _region(self, c):
        """频道 c 的可行域外近似凸多边形（带缓存）。None=无信息。"""
        obs = self.obs.get(c)
        if not obs:
            return None
        key = len(obs)
        hit = self._reg_cache.get(c)
        if hit is not None and hit[0] == key:
            return hit[1]
        P = PARAMS
        margin = P.get('region_margin', _WEDGE_HALF)
        poly = _arena_poly()
        use_disk = bool(P.get('region_disk', 1))
        for (px, py, th) in obs:
            a1 = math.radians(th) - margin
            a2 = math.radians(th) + margin
            c1, s1 = math.cos(a1), math.sin(a1)
            c2, s2 = math.cos(a2), math.sin(a2)
            # cross(d1, X−p) >= 0
            poly = _clip_hp(poly, -s1, c1, -s1 * px + c1 * py)
            # cross(d2, X−p) <= 0
            poly = _clip_hp(poly, s2, -c2, s2 * px - c2 * py)
            if len(poly) < 3:
                break
            if use_disk:
                step = 2.0 * math.pi / _DISK_N
                for k in range(_DISK_N):
                    ang = k * step + 0.5 * step
                    na, nb = math.cos(ang), math.sin(ang)
                    # 外接正 N 边形：na·(X−p) <= _DISK_RC
                    poly = _clip_hp(poly, -na, -nb,
                                    -(na * px + nb * py) - _DISK_RC)
                    if len(poly) < 3:
                        break
            if len(poly) < 3:
                break
        if len(poly) < 3:
            # 交为空（读数舍入导致的不一致）→ 视为无信息，绝不据此剪枝
            poly = None
        self._reg_cache[c] = (key, poly)
        return poly

    def region_min_dist(self, px, py, c):
        """站点 P 到频道 c 可行域的最小距离；无信息时返回 inf 语义的 -1。"""
        poly = self._region(c)
        if poly is None:
            return -1.0
        return _pt_poly_dist(px, py, poly)

    # ---------------- 定向源"被包围"排除（严格可证明安全，新） ----------------
    def _small_tris(self, c):
        pts = self.nos.get(c)
        if not pts or len(pts) < 3:
            return None
        key = len(pts)
        hit = self._tri_cache.get(c)
        if hit is not None and hit[0] == key:
            return hit[1]
        dmax = PARAMS.get('tri_diam', 2600.0)
        if len(pts) > 20:
            pts = pts[-20:]
        n = len(pts)
        tris = []
        for i in range(n - 2):
            xi, yi = pts[i]
            for j in range(i + 1, n - 1):
                xj, yj = pts[j]
                if _hyp(xi - xj, yi - yj) > dmax:
                    continue
                for k in range(j + 1, n):
                    xk, yk = pts[k]
                    if _hyp(xi - xk, yi - yk) > dmax:
                        continue
                    if _hyp(xj - xk, yj - yk) > dmax:
                        continue
                    tris.append(((xi, yi), (xj, yj), (xk, yk)))
        if len(tris) > 400:
            tris = tris[-400:]
        self._tri_cache[c] = (key, tris)
        return tris

    def _feasible_t(self, c):
        """只有 1 条示向度时，射线 X(t)=O+t·d 上"仍可能藏着源"的 t 区间表。

        排除依据：源若落在某个"三顶点两两 <=1000 m"的 no_signal 三角形内，
        需要存在方向 u 同时背对三个站 —— 不可能。故该三角形内部必无源。
        返回 None 表示无信息（不剪枝）；返回 [] 表示射线全程已被排除。
        """
        obs = self.obs.get(c)
        if not obs or len(obs) != 1:
            return None
        nnos = len(self.nos.get(c, ()))
        key = (len(obs), nnos)
        hit = self._feas_cache.get(c)
        if hit is not None and hit[0] == key:
            return hit[1]
        ox, oy, th = obs[0]
        a = math.radians(th)
        ux, uy = math.cos(a), math.sin(a)
        b = ox * ux + oy * uy
        cc = ox * ox + oy * oy
        disc = b * b - cc + ARENA_R * ARENA_R
        tmax = min(1500.0, -b + math.sqrt(disc)) if disc > 0.0 else 0.0
        if tmax <= 0.0:
            self._feas_cache[c] = (key, [])
            return []
        marg = PARAMS.get('region_margin', 1.05 * math.pi / 180.0)
        m = math.tan(marg) * 1500.0 + 6.0        # 楔形侧向半宽上界 + 余量
        ex = []
        for tri in (self._small_tris(c) or ()):
            iv = _ray_in_shrunk_tri(ox, oy, ux, uy, tri, m, tmax)
            if iv is not None:
                ex.append(iv)
        if not ex:
            self._feas_cache[c] = (key, [(0.0, tmax)])
            return [(0.0, tmax)]
        ex.sort()
        merged = [list(ex[0])]
        for iv in ex[1:]:
            if iv[0] <= merged[-1][1]:
                if iv[1] > merged[-1][1]:
                    merged[-1][1] = iv[1]
            else:
                merged.append(list(iv))
        out = []
        cur = 0.0
        for lo, hi in merged:
            if lo > cur:
                out.append((cur, lo))
            if hi > cur:
                cur = hi
        if cur < tmax:
            out.append((cur, tmax))
        self._feas_cache[c] = (key, out)
        _bump('feas_seg', len(out))
        return out

    def _sta_reach_t(self, px, py, ox, oy, ux, uy, tmax):
        """射线 X(t) 落在以站点 P 为心、1500 m 为半径的圆内的 t 区间。"""
        vx, vy = ox - px, oy - py
        B = ux * vx + uy * vy
        C = vx * vx + vy * vy - 1500.0 * 1500.0
        disc = B * B - C
        if disc <= 0.0:
            return None
        s = math.sqrt(disc)
        lo = max(0.0, -B - s)
        hi = min(tmax, -B + s)
        return (lo, hi) if hi > lo else None

    def ray_prune(self, px, py, c):
        """1 条示向度频道：站点 P 是否物理上不可能收到（严格安全）。"""
        _bump('rp_call')
        obs = self.obs.get(c)
        if not obs or len(obs) != 1:
            return False
        feas = self._feasible_t(c)
        if feas is None:
            return False
        if not feas:
            return True
        ox, oy, th = obs[0]
        a = math.radians(th)
        ux, uy = math.cos(a), math.sin(a)
        tmax = feas[-1][1]
        win = self._sta_reach_t(px, py, ox, oy, ux, uy, tmax)
        if win is None:
            return True
        for lo, hi in feas:
            if min(hi, win[1]) > max(lo, win[0]):
                return False
        return True

    @staticmethod
    def _ray_seg_dist(px, py, obs):
        """(px,py) 到该频道"仍可能应答的源位置集合"的最短距离（取最小）。

        只有一条/几条示向度时，源必在轨迹
            T = {O + t·u : t ∈ [0,1500]} ∩ 场地(半径 1800)
        上（接收半径 ≤1500）。此外源要对 (px,py) 应答还需
            (O−S)·(P−S) ≥ 0   ⟺   t ≥ u·(P−O) =: t_c
        —— 因为定向扇区是以源为顶点的 ±90° 半平面，而 O 必在扇区内，
        所以"从 O 看过去更远的那一半"不可能覆盖 P。
        两个条件都不满足 ⇒ 该站物理上不可能收到 c 的信号，可安全跳过补测。
        """
        best = None
        for (ox, oy, th) in obs:
            a = math.radians(th)
            ux, uy = math.cos(a), math.sin(a)
            b = ox * ux + oy * uy
            c = ox * ox + oy * oy
            disc = b * b - c + ARENA_R * ARENA_R
            if disc <= 0.0:
                continue                       # 整条射线都在场地外
            rt = math.sqrt(disc)
            t0 = max(0.0, -b - rt)
            t1 = min(1500.0, -b + rt)
            # 扇区条件 t >= t_c
            tc = (px - ox) * ux + (py - oy) * uy
            if tc > t0:
                t0 = tc
            if t1 < t0:
                continue
            t = (px - ox) * ux + (py - oy) * uy
            if t < t0:
                t = t0
            elif t > t1:
                t = t1
            qx, qy = ox + t * ux, oy + t * uy
            dd = _hyp(px - qx, py - qy)
            if best is None or dd < best:
                best = dd
        return best if best is not None else 1e18

    # =====================================================================
    # 在线路线重优化
    # =====================================================================
    def _node_xy(self, node):
        if node[0] == 'S':
            st = self.plan[node[1]]
            return (st[0], st[1])
        f = self.fix.get(node[1])
        if f is None:
            return self.pos
        return (f[0], f[1])

    def pending_nodes(self):
        """当前还值得跑一趟的任务节点。"""
        out = []
        for i, st in enumerate(self.plan):
            if self.plan_done[i]:
                continue
            if self.channels_at(st):
                out.append(('S', i))
        maxt = int(PARAMS['opp_try'])
        for c in self.fix:
            if c in self.cleared:
                continue
            if self.attempt.get(c, 0) >= maxt:
                continue
            out.append(('C', c))
        return out

    def _plan_route(self, nodes):
        P = PARAMS
        mode = P.get('planner', 'A')
        ls = int(P.get('plan_ls', 2))
        stab = int(P.get('plan_stability', 0))
        if not nodes:
            self.route = []
            return []
        # ---- 起点顺序 ----
        order = None
        if stab and self.route:
            keyset = set(nodes)
            surv = [n for n in self.route if n in keyset]
            have = set(surv)
            new = [n for n in nodes if n not in have]
            if surv:
                order = self._cheapest_insert(surv, new)
        seed = P.get('plan_seed', 'nn')
        if order is None and not self.route and seed != 'nn':
            order = self._seed_order(nodes, seed)
        if order is None:
            order = self._nn_order(nodes)
        if ls <= 0:
            return order
        # ---- 通用局部搜索 ----
        pts_all = [self._node_xy(n) for n in nodes]
        km = {n: i for i, n in enumerate(nodes)}
        idx = [km[n] for n in order]
        if ls >= 4:
            best_i, best_l = None, None
            for s0 in self._multi_starts(nodes, 6):
                cand = [s0] + [x for x in idx if x != s0]
                cand = self._ls_order(cand, pts_all, self.pos, 3)
                ln = self._order_len(cand, pts_all, self.pos)
                if best_l is None or ln < best_l - 1e-9:
                    best_l, best_i = ln, cand
            idx = best_i
            out = [nodes[i] for i in idx]
        else:
            depth = 1 if ls == 1 else (2 if ls == 2 else 3)
            idx = self._ls_order(idx, pts_all, self.pos, depth)
            out = [nodes[i] for i in idx]
        # ---- 紧急插入：目标离得很近就直接排到最前 ----
        ur = P.get('plan_urgent_r', 0.0)
        if ur > 0.0:
            for k, n in enumerate(out):
                if n[0] == 'C' and k > 0:
                    f = self.fix.get(n[1])
                    if f is not None and _hyp(f[0] - self.pos[0],
                                              f[1] - self.pos[1]) <= ur:
                        out.insert(0, out.pop(k))
                        break
        return out

    def _seed_order(self, nodes, seed):
        """首版路线用"预计算的站点最优回路"起手（可选反向）。

        站点集合是静态已知的，所以一开始就能把 24 站的骨架排到最优；
        剩下的问题只是"站序的行走方向会不会影响源的定位延迟"。
        """
        pos_of = {}
        for n in nodes:
            x, y = self._node_xy(n)
            pos_of[(round(x, 6), round(y, 6))] = n
        tour = self._tsp_order(self.plan)          # 返回 [x,y,kind,ridx]
        seq = []
        for p in tour:
            key = (round(p[0], 6), round(p[1], 6))
            n = pos_of.get(key)
            if n is not None:
                seq.append(n)
        if seed == 'tour_rev':
            seq = seq[::-1]
        got = set(seq)
        seq += [n for n in nodes if n not in got]
        return seq

    def _ray_sin(self, mx, my, obs, tref):
        """中继点 M 到各观测射线的交角 |sinψ| 的最大值（估算，用于避免退化交会）。"""
        best = None
        for (ox, oy, th) in obs:
            a = math.radians(th)
            ux, uy = math.cos(a), math.sin(a)
            p = (mx - ox) * ux + (my - oy) * uy
            qx, qy = ox + p * ux, oy + p * uy
            h = _hyp(mx - qx, my - qy)
            d = _hyp(tref - p, h)
            if d < 1.0:
                continue
            s = h / d
            if best is None or s > best:
                best = s
        return best

    def _relay_point(self, ax, ay, bx, by, obs):
        """在腿 A→B 上选最优点做中继。

        **腿上任一点都是零额外移动**：|A−M| + |M−B| = |A−B|。所以不必固定在中点，
        可以挑"离可行射线最近"的点来提高命中率（失败=追加 6 s，成功=替换≈0）。
        返回 (t, 到射线的距离, 交角 sin)，全部不合格则 None。
        """
        P = PARAMS
        tref = P.get('relay_tref', 800.0)
        reach = P.get('relay_reach', 1200.0)
        sin_min = P.get('relay_sin', 0.05)
        best = None
        for i in range(11):
            t = i / 10.0
            px = ax + t * (bx - ax)
            py = ay + t * (by - ay)
            s = self._ray_sin(px, py, obs, tref)
            if s is None or s < sin_min:
                continue
            d = self._ray_seg_dist(px, py, obs)
            if d > reach:
                continue
            if best is None or d < best[1]:
                best = (t, d, s)
        return best

    def _relay(self, order):
        """中继测量入口：mode 1 = c4 几何版；mode 2 = c2 后验模型版。"""
        if int(PARAMS.get('relay_mode', 1)) >= 2:
            return self._relay_pred(order)
        return self._relay_geom(order)

    def _relay_geom(self, order):
        """**中继测量**：在"当前腿的中点"对一个"只有 1 条示向度"的频道补测一次。

        移动成本为 0 —— 中点本来就在 A→B 的直线上，两次动作的位移之和
        `|A−M| + |M−B| = |A−B|` 不变，只多花一次检测（5 s + 1 s 切频道）。
        目标是把"首条示向度 → 可交会定位"的延迟削掉一段。

        两道过滤（避免白测）：
          a) 中点到该频道"可行源域"（射线 ∩ 场地 ∩ 扇区）的距离 ≤ reach；
          b) 交角 |sinψ| = h / |S−M| ≥ relay_sin（h = 中点到射线的垂距，
             |S−M| 用参考距离 tref 估），保证交会不退化。
        """
        P = PARAMS
        if not P.get('relay') or not order:
            return False
        ax, ay = self.pos
        bx, by = self._node_xy(order[0])
        leg = _hyp(bx - ax, by - ay)
        if leg < P.get('relay_min_leg', 300.0):
            return False
        mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
        reach = P.get('relay_reach', 1500.0)
        tref = P.get('relay_tref', 800.0)
        sin_min = P.get('relay_sin', 0.35)
        best_pt = (P.get('relay_point', 'mid') == 'best')
        cands = []
        poor = bool(P.get('relay_poor', 0))
        for c in CH_ALL:
            if c in self.cleared or c in self.relay_done:
                continue
            f = self.fix.get(c)
            if f is not None:
                if not poor or (f[2] <= P['fix_sigma_ok']
                                and len(self.obs.get(c, ())) >= 2):
                    continue
            obs = self.obs.get(c)
            if not obs or len(obs) > int(P.get('relay_maxobs', 1)):
                continue
            if best_pt:
                bp = self._relay_point(ax, ay, bx, by, obs)
                if bp is None:
                    continue
                cands.append((-bp[1], bp[2], bp[0], c))
                continue
            dreg = self.region_min_dist(mx, my, c)
            if dreg < 0.0:
                dreg = self._ray_seg_dist(mx, my, obs)
            if dreg > reach:
                continue
            best = self._ray_sin(mx, my, obs, tref)
            if best is None or best < sin_min:
                continue
            # 命中率优先：中点到可行源域越近越可能真的应答
            if P.get('relay_rank', 'sin') == 'dist':
                cands.append((-dreg, best, 0.5, c))
            else:
                cands.append((best, best, 0.5, c))
        if not cands:
            return False
        cands.sort(reverse=True)
        sel = cands[:max(1, int(P.get('relay_cap', 1)))]
        if best_pt:
            # 必须在腿上按 t 递增执行，才能保证总位移仍为 |A−B|
            sel.sort(key=lambda x: x[2])
        got = False
        tried = []
        # 中继成功 = **替换**（后面本来要测该频道的站点会被跳过），边际成本≈0；
        # 中继失败 = **追加**（白花 6 s）。所以：命中就继续多做一个，失败就停。
        multi = bool(P.get('relay_multi', 0))
        for _nd, _s, _t, c in sel:
            px = ax + _t * (bx - ax)
            py = ay + _t * (by - ay)
            self.relay_done.add(c)
            tried.append(c)
            self.where = 'relay'
            self.do_measure(c, px, py)
            self.update_fixes()
            if c in self.fix:
                got = True
                if not multi:
                    break
        if not got and P.get('relay_retry', 0):
            # 中继没换来定位就允许下一条腿再试（否则只能等站点补测）
            for c in tried:
                self.relay_done.discard(c)
        return got

    # ------------------------------------------------------------------
    # 中继测量 v2：后验命中概率模型（自 c2 移植）
    # ------------------------------------------------------------------
    def _phit(self, c, mx, my):
        """在 (mx,my) 处对频道 c 补测一次、返回 direction 的后验预测概率。"""
        obs = self.obs.get(c)
        if not obs:
            return 0.0
        ox, oy, th = obs[0]
        a = math.radians(th)
        ux, uy = math.cos(a), math.sin(a)
        om = PARAMS.get('relay_omni', 0.5)
        num = 0.0
        for t in _TGRID:
            sx = ox + t * ux
            sy = oy + t * uy
            dx = mx - sx
            dy = my - sy
            d = math.sqrt(dx * dx + dy * dy)
            dm = d if d > t else t
            if dm <= RX_LO:
                q = 1.0
            elif dm >= RX_HI:
                continue
            else:
                q = (RX_HI - dm) / 500.0
            if om < 1.0:
                cs = ((ox - sx) * dx + (oy - sy) * dy) / (t * d) \
                    if d > 1e-9 else 0.0
                if cs > 1.0:
                    cs = 1.0
                elif cs < -1.0:
                    cs = -1.0
                ang = math.degrees(math.acos(cs))
                cf = om + (1.0 - om) * (1.0 - ang / 180.0)
            else:
                cf = 1.0
            num += t * q * cf
        return num / _WSUM

    def _leg_ray_dist(self, ax, ay, bx, by, c):
        """腿段 [A,B] 到"该频道可行射线段"的最短距离（粗筛用）。"""
        obs = self.obs.get(c)
        if not obs:
            return 1e18
        ox, oy, th = obs[0]
        a = math.radians(th)
        ux, uy = math.cos(a), math.sin(a)
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        best = 1e18
        for t in _TGRID:
            sx = ox + t * ux
            sy = oy + t * uy
            if L2 > 1e-9:
                s = ((sx - ax) * vx + (sy - ay) * vy) / L2
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
            else:
                s = 0.0
            qx = ax + s * vx
            qy = ay + s * vy
            dd = _hyp(sx - qx, sy - qy)
            if dd < best:
                best = dd
        return best

    def _relay_pred(self, order):
        """中继测量 v2：用后验命中概率同时决定「测不测 / 测哪个 / 在哪测」。

        腿上的任意点都满足 |A−M| + |M−B| = |A−B|，**移动成本恒为 0**，
        所以选点完全由"命中概率"决定；多个频道只要按沿腿参数递增顺序测，
        移动成本仍然是 0。
        """
        P = PARAMS
        if not P.get('relay') or not order:
            return False
        ax, ay = self.pos
        bx, by = self._node_xy(order[0])
        leg = _hyp(bx - ax, by - ay)
        if leg < P.get('relay_min_leg', 150.0):
            _bump('rl_shortleg')
            return False
        K = max(1, int(P.get('relay_pts', 13)))
        ks = range(1, K + 1)
        pxs = [ax + (bx - ax) * k / (K + 1.0) for k in ks]
        pys = [ay + (by - ay) * k / (K + 1.0) for k in ks]
        pre = P.get('relay_pre', 900.0)
        pmin = P.get('relay_pmin', 0.40)
        retry = int(P.get('relay_retry', 0))
        sin2 = P.get('relay_sin2', 0.08)
        qmode = int(P.get('relay_qmode', 0))
        tmean = P.get('relay_tmean', 0.0) or T_MEAN
        rval = P.get('relay_value', 12.0)
        off_max = P.get('relay_off_max', 0.0)
        adv_w = int(P.get('relay_adv_w', 0))
        adv_min = P.get('relay_adv_min', 0.0)
        maxobs = int(P.get('relay_maxobs', 1))
        poor = bool(P.get('relay_poor', 0))
        scored = []          # (rank, c, k, p, mx, my, 沿腿参数)
        for c in CH_ALL:
            if c in self.cleared or (c in self.relay_done and not retry):
                continue
            f = self.fix.get(c)
            if f is not None:
                if not poor or (f[2] <= P['fix_sigma_ok']
                                and len(self.obs.get(c, ())) >= 2):
                    continue
            obs = self.obs.get(c)
            if not obs or len(obs) > maxobs:
                continue
            if self._leg_ray_dist(ax, ay, bx, by, c) > pre:
                _bump('rl_pre')
                continue
            ox, oy, th = obs[0]
            a = math.radians(th)
            ux, uy = math.cos(a), math.sin(a)
            # 源的后验估计点，用来估"第二条射线与第一条的交角"
            sx, sy = ox + tmean * ux, oy + tmean * uy
            # ---- 候选点：腿上 K 个（移动成本 0） + 朝源估计点的偏移族 ----
            cand_pts = [(pxs[k], pys[k], 0.0) for k in range(K)]
            if off_max > 0.0:
                qx, qy = _closest_on_seg(ax, ay, bx, by, sx, sy)
                vx, vy = sx - qx, sy - qy
                if _hyp(vx, vy) > 1e-6:
                    for lam in (0.25, 0.5, 0.75, 1.0):
                        tx = qx + lam * vx
                        ty = qy + lam * vy
                        et = _seg_dist(tx, ty, ax, ay, bx, by)
                        if et > off_max:
                            k2 = off_max / et
                            tx = qx + (tx - qx) * k2
                            ty = qy + (ty - qy) * k2
                        ex = (_hyp(tx - ax, ty - ay)
                              + _hyp(bx - tx, by - ty) - leg)
                        cand_pts.append((tx, ty, ex if ex > 0.0 else 0.0))
            bobj, bp, bk = -1e18, 0.0, -1
            for k, (mx, my, ex) in enumerate(cand_pts):
                p = self._phit(c, mx, my)
                if p <= 0.0:
                    continue
                d = _hyp(mx - sx, my - sy)
                if d < 1.0:
                    d = 1.0
                h = abs((mx - ox) * uy - (my - oy) * ux)
                sn = h / d
                if qmode == 0:
                    if sn < sin2:
                        continue
                else:
                    p *= (sn / sin2) if sn < sin2 else 1.0
                obj = p * rval - 0.2 * ex      # 收益(s) − 绕路代价(s)
                if obj > bobj:
                    bobj, bp, bk = obj, p, k
            if bk < 0 or bp < pmin:
                continue
            mx, my = cand_pts[bk][0], cand_pts[bk][1]
            rank = bobj
            if adv_min > 0.0 or adv_w:
                adv = self._advance(c, order, ax, ay, mx, my, leg)
                if adv < adv_min:
                    _bump('rl_advcut')
                    continue
                if adv_w:
                    rank = bobj * adv
            s_along = (((mx - ax) * (bx - ax) + (my - ay) * (by - ay))
                       / (leg * leg)) if leg > 1e-9 else 0.0
            scored.append((rank, c, bk, bp, mx, my, s_along))
        _bump('rl_calls')
        if not scored:
            _bump('rl_nocand')
            return False
        scored.sort(reverse=True)
        keep = max(1, int(P.get('relay_keep', 1)))
        pmin2 = P.get('relay_pmin2', 0.0)
        sel = []
        for z in scored:
            if len(sel) >= keep:
                break
            if sel and pmin2 > 0.0 and z[3] < pmin2:
                continue
            sel.append(z)
        sel.sort(key=lambda z: z[6])          # 沿腿顺序，移动成本仍为 0
        stop = int(P.get('relay_stop', 0))
        got = False
        for _r, c, _k, _p, mx, my, _s in sel:
            self.relay_done.add(c)
            _bump('rl_meas')
            _bump('rl_phit', _p)
            _res, _ = self.do_measure(c, mx, my)
            if _res != 'direction':
                _bump('rl_waste')
            else:
                self.relay_fixed.add(c)
            self.update_fixes()
            if c in self.fix:
                got = True
                _bump('rl_hit')
                if stop:
                    break
        return got

    def _advance(self, c, order, ax, ay, mx, my, leg):
        """中继把"第 2 条射线"提前了多少米（默认关闭，仅诊断/负面复现用）。"""
        obs = self.obs.get(c)
        if not obs:
            return 1e18
        sf = PARAMS.get('skip_far', 0.0) or 1500.0
        d_m = _hyp(mx - ax, my - ay)
        cum = 0.0
        px, py = ax, ay
        lim = int(PARAMS.get('relay_adv_scan', 14))
        for i, n in enumerate(order[:lim]):
            nx, ny = self._node_xy(n)
            cum += _hyp(nx - px, ny - py)
            px, py = nx, ny
            if n[0] != 'S':
                continue
            if self._ray_seg_dist(nx, ny, obs) <= sf:
                return cum - d_m
        return 1e18

    def _multi_starts(self, nodes, k):
        pts = [self._node_xy(n) for n in nodes]
        idx = sorted(range(len(nodes)),
                     key=lambda i: (pts[i][0] - self.pos[0]) ** 2
                     + (pts[i][1] - self.pos[1]) ** 2)
        return idx[:k]

    def _nn_order(self, nodes):
        pts = [self._node_xy(n) for n in nodes]
        rem = list(range(len(nodes)))
        cur = self.pos
        order = []
        while rem:
            k = min(rem, key=lambda i: (pts[i][0] - cur[0]) ** 2
                    + (pts[i][1] - cur[1]) ** 2)
            rem.remove(k)
            order.append(nodes[k])
            cur = pts[k]
        return order

    def _cheapest_insert(self, route, new):
        route = list(route)
        pending = list(new)
        while pending:
            pos = self._coords_of(route)
            best = None
            for n in pending:
                p = self._node_xy(n)
                bl, bk = None, 0
                for k in range(len(route) + 1):
                    if k == len(route):
                        c = _hyp(pos[k][0] - p[0], pos[k][1] - p[1])
                    else:
                        u, v = pos[k], pos[k + 1]
                        c = (_hyp(p[0] - u[0], p[1] - u[1])
                             + _hyp(v[0] - p[0], v[1] - p[1])
                             - _hyp(v[0] - u[0], v[1] - u[1]))
                    if bl is None or c < bl:
                        bl, bk = c, k
                if best is None or bl < best[0] - 1e-12:
                    best = (bl, n, bk)
            route.insert(best[2], best[1])
            pending.remove(best[1])
        return route

    def _coords_of(self, order):
        out = [self.pos]
        for n in order:
            out.append(self._node_xy(n))
        return out

    # ---------------- 基线调度（planner='A'）----------------
    def choose_task(self, pending):
        P = PARAMS
        px, py = self.pos
        best = None
        look = P.get('lookahead', 0.0)
        if look > 0.0 and len(pending) > 1:
            coords = [(self.plan[i][0], self.plan[i][1]) for i in pending]
            for k, i in enumerate(pending):
                st = coords[k]
                d = _hyp(st[0] - px, st[1] - py)
                nb = min((_hyp(coords[j][0] - st[0], coords[j][1] - st[1])
                          for j in range(len(coords)) if j != k), default=0.0)
                sc = d - look * nb
                if best is None or sc < best[0]:
                    best = (sc, 'sta', i)
        else:
            for i in pending:
                st = self.plan[i]
                if P['route_mode'] == 'tour':
                    sc = float(i)
                else:
                    sc = _hyp(st[0] - px, st[1] - py) * P['sta_bonus']
                if best is None or sc < best[0]:
                    best = (sc, 'sta', i)
        lim = P['opp_max'] if pending else 1e18
        maxt = int(P['opp_try'])
        for c, f in self.fix.items():
            if c in self.cleared:
                continue
            if self.attempt.get(c, 0) >= maxt:
                continue
            d = _hyp(f[0] - px, f[1] - py)
            if d > lim:
                continue
            if best is None or d < best[0]:
                best = (d, 'opp', c)
        return None if best is None else (best[1], best[2])

    # ---------------- 主循环 ----------------
    def run(self):
        self.ctx.enter()
        self.plan = self.make_plan()
        self.plan_done = [False] * len(self.plan)
        self.inner_visited = 0
        self.update_fixes()
        try:
            self.main_loop()
        except Exception as exc:                                # noqa: BLE001
            _bump('err_main')
            STATS['msg_main'] = f'{type(exc).__name__}: {exc}'
        for nm, stage in (('hunt', self.final_hunt),
                          ('clear', self.final_clear)):
            try:
                stage()
            except Exception as exc:                            # noqa: BLE001
                _bump('err_' + nm)
                STATS['msg_' + nm] = f'{type(exc).__name__}: {exc}'
        self.ctx.exit()

    def main_loop(self):
        planner = PARAMS.get('planner', 'A')
        for it in range(600):
            _bump('loop_it')
            if planner == 'A':
                pending = []
                for i, st in enumerate(self.plan):
                    if self.plan_done[i]:
                        continue
                    if self.channels_at(st):
                        pending.append(i)
                if len(self.seen) >= int(PARAMS['early_k']) or not pending:
                    pending = []
                self.phase = 'survey' if pending else 'clear'
                task = self.choose_task(pending)
                if task is None:
                    break
                kind, payload = task
                if kind == 'sta':
                    self.visit_station(payload)
                else:
                    self.clear_source(payload)
                self.update_fixes()
                continue

            # ---- 在线重优化调度 ----
            nodes = self.pending_nodes()
            if len(self.seen) >= int(PARAMS['early_k']):
                nodes = [n for n in nodes if n[0] == 'C']
            self.phase = ('survey' if any(n[0] == 'S' for n in nodes)
                          else 'clear')
            if not nodes:
                self.route = []
                break
            order = self._plan_route(nodes)
            self.route = order
            if not order:
                break
            if self._relay(order):
                nodes = self.pending_nodes()
                order = self._plan_route(nodes)
                self.route = order
                if not order:
                    break
            n0 = order[0]
            if n0[0] == 'S':
                self.visit_station(n0[1])
            else:
                self.clear_source(n0[1])
            self.update_fixes()
        else:
            _bump('loop_full')

    def final_clear(self):
        """收尾：把所有"已定位但还没清掉"的源再清一遍。"""
        self.phase = 'clear'
        for _ in range(2):
            self.update_fixes()
            todo = [c for c in self.fix if c not in self.cleared]
            if not todo:
                return
            todo.sort(key=lambda c: _hyp(self.fix[c][0] - self.pos[0],
                                         self.fix[c][1] - self.pos[1]))
            for c in todo:
                if c in self.cleared:
                    continue
                f = self.fix.get(c)
                if f is None:
                    continue
                self.do_clear(c, f[0], f[1], f[2])

    def visit_station(self, idx):
        st = self.plan[idx]
        self.plan_done[idx] = True
        if st[2] == 'inner':
            self.inner_visited += 1
        x, y = st[0], st[1]
        act = self.channels_at(st)
        if not act:
            return
        self.where = 'sta'
        _bump('n_visit')
        STATS.setdefault('vorder', []).append(idx)
        nb = sum(1 for c in act if c not in self.seen)
        _bump('sum_blank_visit', nb)
        _bump('sum_chan_visit', len(act))
        _bump('blank_at_%d' % self._visit_no, nb)
        self._visit_no += 1
        act.sort(key=lambda c: 0 if c == self.chan else 1)
        for c in act:
            self.do_measure(c, x, y)

    def clear_source(self, c):
        f = self.fix.get(c)
        if f is None or c in self.cleared:
            return
        self.attempt[c] = self.attempt.get(c, 0) + 1
        self.do_clear(c, f[0], f[1], f[2])
        cap = int(PARAMS['enroute_cap'])
        if cap > 0:
            act = [c2 for c2 in CH_ALL if c2 not in self.cleared
                   and (self.fix.get(c2) is None
                        or self.fix[c2][2] > PARAMS['fix_sigma_ok'])]
            if act:
                act.sort(key=lambda c: (0 if c == self.chan else 1, c))
                for c2 in act[:cap]:
                    self.do_measure(c2, self.pos[0], self.pos[1])
                    self.update_fixes()

    # ---------------- 逼近 + 清除 ----------------
    def do_clear(self, c, tx, ty, sigma):
        """定位点盲清 → 近距测向步进盲清 → 沿射线长搜索 → 不确定域密集盲清。"""
        P = PARAMS
        self.where = 'creep'
        if self.try_clear(c, tx, ty):
            return True
        step = P['creep_step']
        for _ in range(int(P['creep_max_it'])):
            res, r = self.do_measure(c, self.pos[0], self.pos[1])
            if res == 'near':
                if self.try_clear(c, self.pos[0], self.pos[1]):
                    return True
                break
            if res != 'direction':
                break
            a = math.radians(float(r['svd_deg']))
            nx = self.pos[0] + step * math.cos(a)
            ny = self.pos[1] + step * math.sin(a)
            if self.try_clear(c, nx, ny):
                return True
        if self._ray_search(c, int(P['ray_max_it'])):
            return True
        return self._disc_sweep(c, tx, ty, sigma)

    def _pick_ray(self, c, ref=None):
        obs = self.obs.get(c)
        if not obs:
            return None
        px, py = self.pos if ref is None else ref
        best = None
        for (ox, oy, th) in obs:
            a = math.radians(th)
            ux, uy = math.cos(a), math.sin(a)
            t = (px - ox) * ux + (py - oy) * uy
            if t < 0.0:
                t = 0.0
            qx, qy = ox + t * ux, oy + t * uy
            key = _hyp(ox - px, oy - py)
            dd = _hyp(qx - self.pos[0], qy - self.pos[1])
            if best is None or key < best[0]:
                best = (key, ux, uy, qx, qy, ox, oy, dd)
        return None if best is None else (best[7],) + best[1:7]

    def _rank_rays(self, c, ref=None):
        obs = self.obs.get(c)
        if not obs:
            return []
        px, py = self.pos if ref is None else ref
        out = []
        for (ox, oy, th) in obs:
            a = math.radians(th)
            ux, uy = math.cos(a), math.sin(a)
            t = (px - ox) * ux + (py - oy) * uy
            if t < 0.0:
                t = 0.0
            qx, qy = ox + t * ux, oy + t * uy
            out.append((_hyp(ox - px, oy - py), ux, uy, qx, qy, ox, oy,
                        _hyp(qx - self.pos[0], qy - self.pos[1])))
        out.sort(key=lambda r: r[0])
        return [(r[7],) + r[1:7] for r in out]

    def _ray_search(self, c, max_it):
        f = self.fix.get(c)
        ref = (f[0], f[1]) if f is not None else None
        rays = self._rank_rays(c, ref)
        if not rays:
            return False
        for r in rays[:2]:
            if self._walk_ray(c, r, max_it, False):
                return True
        if PARAMS['ray_from_o']:
            for r in rays[:2]:
                if self._walk_ray(c, r, max_it, True):
                    return True
        return False

    def _walk_ray(self, c, ray, max_it, from_o):
        self.where = 'ray'
        _, ux, uy, qx, qy, ox, oy = ray
        step = PARAMS['ray_step']
        if from_o:
            if _hyp(self.pos[0] - ox, self.pos[1] - oy) > 1.0:
                if self.try_clear(c, ox, oy):
                    return True
        elif _hyp(self.pos[0] - qx, self.pos[1] - qy) > 1.0:
            if self.try_clear(c, qx, qy):
                return True
        for k in range(max_it):
            if self.try_clear(c, self.pos[0] + step * ux,
                              self.pos[1] + step * uy):
                return True
            if k % 2 == 1 or (k == 0 and PARAMS.get('ray_recheck')):
                res, r = self.do_measure(c, self.pos[0], self.pos[1])
                if res == 'near':
                    if self.try_clear(c, self.pos[0], self.pos[1]):
                        return True
                    break
                if res == 'direction':
                    a = math.radians(float(r['svd_deg']))
                    ux, uy = math.cos(a), math.sin(a)
                else:
                    vx, vy = ox - self.pos[0], oy - self.pos[1]
                    n = _hyp(vx, vy)
                    if n < step:
                        break
                    ux, uy = vx / n, vy / n
        return False

    def _disc_sweep(self, c, cx, cy, sigma):
        P = PARAMS
        gap = P['sweep_gap']
        att = max(1, int(self.attempt.get(c, 1)))
        base = max(1.7 * sigma, P['sweep_rmin'])
        rmax = min(base * (P['sweep_grow'] ** (att - 1)), P['sweep_rmax'])
        cap = min(int(P['sweep_cap']) * att, int(P['sweep_cap_max']))
        tot = 0
        r = gap * 0.7
        while r <= rmax + 1e-9:
            k = max(6, int(round(2.0 * math.pi * r / gap)))
            for i in range(k):
                a = 2.0 * math.pi * i / k + 0.31 * r
                if self.try_clear(c, cx + r * math.cos(a), cy + r * math.sin(a)):
                    return True
                tot += 1
                if tot >= cap:
                    return False
            r += gap
        return False

    # ---------------- 兜底 ----------------
    def final_hunt(self):
        mode = PARAMS['hunt_mode']
        for _ in range(30):
            self.update_fixes()
            pend = [c for c in CH_ALL
                    if c not in self.cleared and c not in self.fix]
            one = [c for c in pend if self.obs.get(c)]
            if not one:
                return
            one.sort(key=lambda c: len(self.obs[c]))
            c = one[0]
            self.phase = 'hunt'
            if mode in ('ray', 'both'):
                if self._ray_search(c, int(PARAMS['ray_find_it'])):
                    continue
            if mode in ('perp', 'both'):
                self._perp_probe(c)
                continue
            return

    def _perp_probe(self, c):
        self.where = 'perp'
        best = self._pick_ray(c)
        if best is None:
            return
        _, ux, uy, qx, qy, _ox, _oy = best
        W = PARAMS['hunt_probe']
        cand = None
        for sgn in (1.0, -1.0):
            px2 = qx - W * sgn * uy
            py2 = qy + W * sgn * ux
            dd = _hyp(px2 - self.pos[0], py2 - self.pos[1])
            if cand is None or dd < cand[0]:
                cand = (dd, px2, py2)
        self.do_measure(c, cand[1], cand[2])
        self.update_fixes()
        if c in self.fix:
            self.clear_source(c)
        else:
            self._ray_search(c, int(PARAMS['ray_find_it']))


# ===========================================================================
# 主工程适配层（robotdog 约定）
# ---------------------------------------------------------------------------
# 以上是**决策层**（PARAMS + strategy + _Solver），与单文件版逐字一致；下面这层
# 只做三件事，不改变任何决策：
#   1) ``build_cfg`` / ``Q4Config`` —— 评测与部署共用的**唯一**参数入口；
#   2) ``world_kwargs`` / ``_NullBelief`` —— 进程内 ``World`` 的空信念装配；
#   3) ``_Ctx`` + ``run_sweeper4`` —— 把 ``World`` / ``RemoteWorld`` 适配成
#      opt1 形态的四方法接口，并保留动作数 / 虚拟时间 / 现实时间三个护栏。
# ===========================================================================

#: 不参与决策的护栏旋钮（只在失控时让程序正常收尾，见 ``_Ctx._tick``）
_GUARD_KEYS = ("max_actions", "max_virtual")


class _NullBelief:
    """占位信念: 本方案只用几何 + 示向度，不做贝叶斯推断。"""

    def __init__(self) -> None:
        self.tracks: List[Any] = []

    def on_measure(self, mx: float, my: float, channel: int, kind: str,
                   svd: Optional[float], t: float) -> None:
        return None

    def on_clear(self, mx: float, my: float, channel: int, hit: bool) -> None:
        return None


def world_kwargs() -> Dict[str, Any]:
    """``World(seed=..., problem=4, **world_kwargs())`` 用空信念只留物理账本。"""
    return {"belief_cls": _NullBelief}


class Q4Config:
    """问题4 策略的全部旋钮: 默认值逐条取自 :data:`PARAMS`。

    ``tools/eval_q4.py --set k=v`` 与 ``deploy.py --cfg-module`` 改的都是这一份；
    另有 ``max_actions`` / ``max_virtual`` 两个护栏（不参与决策）。
    """

    #: 动作数护栏（每局实测约 270 次动作，留 20 倍余量）
    max_actions: int = 6000
    #: 虚拟时间护栏（秒），仅防死循环（题目上限 360000 s）
    max_virtual: float = 90000.0

    def __init__(self, **kw: Any) -> None:
        for k, v in PARAMS.items():
            setattr(self, k, v)
        for k in _GUARD_KEYS:
            setattr(self, k, getattr(type(self), k))
        for k, v in kw.items():
            if k not in PARAMS and k not in _GUARD_KEYS:
                raise AttributeError("Q4Config has no attribute %r" % k)
            setattr(self, k, v)


def build_cfg(**kw: Any) -> Q4Config:
    """发布配置: 评测与部署**共用**的唯一一份参数入口。"""
    return Q4Config(**kw)


def _params_of(cfg: Optional[Any] = None) -> Dict[str, Any]:
    out = dict(PARAMS)
    if cfg is not None:
        for k in PARAMS:
            if hasattr(cfg, k):
                out[k] = getattr(cfg, k)
    return out


def _install(cfg: Optional[Any]) -> Dict[str, Any]:
    """把 cfg 的参数写进模块级 ``PARAMS``（策略运行时读的就是它），返回原值。"""
    saved = dict(PARAMS)
    if cfg is not None:
        for k, v in _params_of(cfg).items():
            PARAMS[k] = v
    return saved


def _restore(saved: Dict[str, Any]) -> None:
    PARAMS.clear()
    PARAMS.update(saved)


def _as_points(rows: Sequence[Sequence[Any]]) -> List[Tuple[float, float]]:
    return [(float(r[0]), float(r[1])) for r in rows]


def _layout_rows(P: Optional[Dict[str, Any]] = None) -> List[List[Any]]:
    """按参数生成站点行 ``[x, y, kind, ridx]``（kind: ``inner`` / ``ring``）。

    直接复用 :meth:`_Solver.make_plan` 的布局代码 —— 唯一真相来源，因此探针
    看到的站点集与策略实际巡游的站点集永远一致。
    """
    saved = dict(PARAMS)
    if P:
        PARAMS.update(P)
    try:
        return _Solver(None, None).make_plan()
    finally:
        _restore(saved)


def _tour_index(xy: Sequence[Sequence[float]], start: Sequence[float],
                restarts: int = 8) -> List[int]:
    """多起点 NN + 2-opt + Or-opt(1..3) 的开链巡游；返回下标序。"""
    n = len(xy)
    if n <= 1:
        return list(range(n))
    best_l: Optional[float] = None
    best_o: List[int] = list(range(n))
    seen0 = set()
    for s0 in range(n):
        r0 = round(_hyp(xy[s0][0], xy[s0][1]), 3)
        if r0 in seen0:
            continue
        seen0.add(r0)
        rem = list(range(n))
        rem.remove(s0)
        cur = s0
        order = [s0]
        while rem:
            k = min(rem, key=lambda i: (xy[i][0] - xy[cur][0]) ** 2
                    + (xy[i][1] - xy[cur][1]) ** 2)
            rem.remove(k)
            order.append(k)
            cur = k
        order = _Solver._ls_order(order, xy, start, 3)
        ln = _Solver._order_len(order, xy, start)
        if best_l is None or ln < best_l - 1e-9:
            best_l, best_o = ln, order
        if len(seen0) >= restarts:
            break
    return best_o


def order_points(points: Sequence[Sequence[float]], start: Sequence[float],
                 restarts: int = 8) -> List[Tuple[float, float]]:
    """开放巡游排序（从 ``start`` 出发，不返回）；供插图与设计探针取数。"""
    xy = [(float(p[0]), float(p[1])) for p in points]
    return [xy[i] for i in _tour_index(xy, start, restarts)]


def _order_len(order: Sequence[int], pts: Sequence[Sequence[float]],
               start: Sequence[float]) -> float:
    """开链路径长度（与 :meth:`_Solver._order_len` 同一份实现）。"""
    return _Solver._order_len(order, pts, start)


def _ls_order(order: Sequence[int], pts: Sequence[Sequence[float]],
              start: Sequence[float], depth: int = 2) -> List[int]:
    """2-opt / Or-opt 局部搜索（与 :meth:`_Solver._ls_order` 同一份实现）。"""
    return _Solver._ls_order(order, pts, start, depth)


def build_points(cfg: Optional[Q4Config] = None) -> List[Tuple[float, float]]:
    """按配置生成站点集（按从原点出发的巡游排序）。

    部署/评测**不使用**这个排序（在线路线由 ``_Solver._plan_route`` 决定），
    它只给论文插图与设计探针提供“这批站点按什么顺序走、骨架多长”的口径。
    """
    rows = _layout_rows(_params_of(cfg if cfg is not None else build_cfg()))
    return order_points(_as_points(rows), (0.0, 0.0))


_SKELETON_CACHE: Dict[Any, float] = {}


def _layout_key() -> Tuple[Any, ...]:
    def tup(v: Any) -> Tuple[Any, ...]:
        return tuple(tuple(x) for x in (v or ()))

    return (tup(PARAMS.get('ring_def')), tup(PARAMS.get('ring_spec')),
            float(PARAMS.get('ring_phase', 0.0)),
            float(PARAMS.get('grid_d', 0.0)),
            float(PARAMS.get('grid_rmax', 0.0)),
            float(PARAMS.get('ring_r', 0.0)),
            int(PARAMS.get('ring_n', 0)))


def skeleton_length_m() -> float:
    """站点骨架的开链巡游长度（静态几何；同一布局只算一次）。"""
    key = _layout_key()
    if key not in _SKELETON_CACHE:
        xy = _as_points(_layout_rows(PARAMS))
        _SKELETON_CACHE[key] = _Solver._order_len(_tour_index(xy, (0.0, 0.0)),
                                                  xy, (0.0, 0.0))
    return _SKELETON_CACHE[key]


class _OutOfBudget(Exception):
    """护栏触发；由 :meth:`_Solver.run` 的分阶段 ``except`` 捕获后正常收尾。"""


class _Ctx:
    """把 ``World`` / ``RemoteWorld`` 适配成策略期望的四方法接口。

    与 opt1 沙盒口径逐字一致: ``measure`` 返回 ``{'measure_result', 'svd_deg'}``，
    ``clear`` 返回 ``{'clear_result'}``；另加动作数 / 虚拟时间 / 现实时间护栏。
    """

    def __init__(self, world: World, cfg: Optional[Q4Config] = None,
                 real_left: Optional[Any] = None, reserve_s: float = 20.0) -> None:
        self.w = world
        self.cfg = cfg or build_cfg()
        self.real_left = real_left
        self.reserve_s = float(reserve_s)
        self.actions = 0
        self.budget_hit = False

    def _tick(self) -> None:
        self.actions += 1
        if (self.actions > int(self.cfg.max_actions)
                or self.w.virtual_t > float(self.cfg.max_virtual)):
            self.budget_hit = True
            raise _OutOfBudget("动作数/虚拟时间护栏")
        if self.real_left is not None:
            try:
                left = float(self.real_left())
            except Exception:                               # noqa: BLE001
                left = float("inf")
            if left <= self.reserve_s:
                self.budget_hit = True
                raise _OutOfBudget("现实时间不足")

    @property
    def pos(self) -> Tuple[float, float]:
        return self.w.pos

    def enter(self) -> Dict[str, Any]:
        return {"accepted": True}

    def exit(self) -> Dict[str, Any]:
        return {"accepted": True}

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        self._tick()
        info = self.w.measure(float(x), float(y), int(channel))
        out: Dict[str, Any] = {"measure_result": info.result}
        if info.result == "direction" and info.svd_deg is not None:
            out["svd_deg"] = float(info.svd_deg)
        return out

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        self._tick()
        info = self.w.clear(float(x), float(y), int(channel))
        return {"clear_result": info.result}


def run_sweeper4(world: World, cfg: Optional[Q4Config] = None, trace: bool = False,
                 real_left: Optional[Any] = None, reserve_s: float = 20.0,
                 knows_total: bool = True):
    """问题4 主入口（签名与其它规划器一致，便于统一评测与正式部署）。

    ``knows_total`` 只保留签名兼容: 本策略**从不读真值总数**（正式测试看不到），
    收手条件完全由自己的观测决定。``real_left`` / ``reserve_s`` 是现实时间护栏:
    正式测试传 ``client.real_left``，进程内评测不传（没有现实时间概念）。
    """
    from .sweeper import SweepResult

    cfg = cfg or build_cfg()
    STATS.clear()
    saved = _install(cfg)
    ctx = _Ctx(world, cfg, real_left=real_left, reserve_s=reserve_s)
    solver = _Solver(ctx, None)
    try:
        try:
            solver.run()
        except Exception as exc:                            # noqa: BLE001
            STATS['msg_run'] = "%s: %s" % (type(exc).__name__, exc)
            try:
                ctx.exit()
            except Exception:                               # noqa: BLE001
                pass
        tour_m = round(skeleton_length_m(), 1)
    finally:
        _restore(saved)
    out = SweepResult(world)
    out.steps = ctx.actions
    out.probes = int(STATS.get('rl_meas', 0.0))
    out.sweeps = 0
    out.pursues = 0
    out.mv_probe = float(STATS.get('mv_survey', 0.0))
    world.q4_hunt_found = int(STATS.get('cok_hunt', 0.0))
    world.q4_stages = {
        "points": len(solver.plan),
        "tour_m": tour_m,
        "actions": ctx.actions,
        "clears": int(world.clears),
        "failed_clears": int(world.failed_clears),
        "budget_hit": ctx.budget_hit,
        "replans": int(STATS.get('rp_call', 0.0)),
        "relay": int(STATS.get('rl_meas', 0.0)),
        "relay_hit": int(STATS.get('rl_hit', 0.0)),
        "survey_m": round(STATS.get('mv_survey', 0.0), 1),
        "clear_m": round(STATS.get('mv_clear', 0.0), 1),
        "hunt_m": round(STATS.get('mv_hunt', 0.0), 1),
        "error": (STATS.get('msg_run') or STATS.get('msg_main')
                  or STATS.get('msg_hunt') or STATS.get('msg_clear')),
    }
    if trace:
        print("  Q4: cleared=%d/%d actions=%d %s"
              % (world.cleared_count, world.n_sources, ctx.actions,
                 world.q4_stages), flush=True)
    return out


def run_candidate(world: World, cfg: Optional[Q4Config] = None, trace: bool = False,
                  real_left: Optional[Any] = None, reserve_s: float = 20.0,
                  knows_total: bool = True):
    """``tools/candidates.py`` 风格的别名（与 :func:`run_sweeper4` 同一实现）。"""
    return run_sweeper4(world, cfg, trace=trace, real_left=real_left,
                        reserve_s=reserve_s, knows_total=knows_total)


def run_case(seed: int, cfg: Optional[Q4Config] = None,
             trace: bool = False) -> Dict[str, Any]:
    cfg = cfg or build_cfg()
    w = World(seed=seed, problem=4, **world_kwargs())
    run_sweeper4(w, cfg, trace=trace)
    return {
        "seed": seed, "cleared": w.cleared_count, "total": w.n_sources,
        "virtual_s": w.virtual_t, "moved_m": w.moved_m, "measures": w.measures,
        "clears": w.clears, "failed_clears": w.failed_clears,
        "n_directed": w.n_directional,
        "cleared_directed": sum(1 for s in w.sources if not s.is_omni and not s.alive),
        "stages": getattr(w, "q4_stages", None),
    }


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import statistics as st
    import time

    ap = argparse.ArgumentParser(description="问题4 发布策略（中继 + 在线路线重优化）评测")
    ap.add_argument("--seeds", default="9500-9529")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args(argv)
    seeds: List[int] = []
    for part in a.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    cfg = build_cfg()
    pts = build_points(cfg)
    t0 = time.time()
    rows = [run_case(sd, cfg, trace=a.trace) for sd in seeds]
    tot = sum(r["total"] for r in rows)
    clr = sum(r["cleared"] for r in rows)
    print("=== 问题4 | %d 局 | %d 站 | 骨架 %.0f m ==="
          % (len(rows), len(pts), skeleton_length_m()))
    print("  清除比例 %d/%d = %.4f   全清 %d/%d"
          % (clr, tot, clr / max(tot, 1),
             sum(1 for r in rows if r["cleared"] == r["total"]), len(rows)))
    print("  平均定位清除 %.1f s/源   中位 %.1f   最差 %.1f"
          % (st.mean(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
             st.median(r["virtual_s"] / max(r["cleared"], 1) for r in rows),
             max(r["virtual_s"] / max(r["cleared"], 1) for r in rows)))
    print("  每局移动 %.0f m   检测 %.1f 次   墙钟 %.1f s"
          % (st.mean(r["moved_m"] for r in rows),
             st.mean(r["measures"] for r in rows), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
