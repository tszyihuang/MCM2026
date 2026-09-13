"""问题3 "清除即探测"策略 (sweeper): 把探测点放在**刚清除的源**上, 而不是路上随机点。

动机 (逐动作剖析的结论):
  * 12.8 个源均匀分布在 r=1800 的圆域里, 平均最近邻距离只有约 450 m, 前 3 近邻
    都在约 900 m 内 —— 而有效接收半径 R >= 1000 m。
  * 因此**站在一个刚清除的源上测其它频道, 对附近未发现的源几乎是"必中"**
    (d <= 1000 m 时 P(收到) = 1)。
  * 反过来, 早期策略在"去目标路上 60% 处"测 (P(检测) ≈ 0.3) 和专程跑到格点上测,
    实测 104 次 no_signal / 局 (624 s), 外加 4740 m 专程探测移动 (948 s)。

策略:
  1. 起点盲扫一次 (原点覆盖半径 1000~1500 m, 命中约一半源);
  2. 之后永远"就近清除"已知源 (路线 ≈ TSP), 逼近途中只测**目标频道**用于定位;
  3. 每次清除成功后, 原地对"仍未发现 (mode 0) 且存在概率高"的频道补一次检测 ——
     这一步不产生任何额外移动;
  4. 只有当没有任何已知目标时才专程探测 (覆盖导向的集合覆盖点)。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import ops
from .belief import SIGMA_RAD, ChannelTrack, max_eig
from .belief import GRID_STEP
from ..consts import N_CHANNELS, RADIUS_MAX_M, RADIUS_MIN_M, SPEED_MPS
from .world import World

Vec = Tuple[float, float]
LATTICE = ops.build_lattice()
# 覆盖收尾的候选点: 允许站到圆域外(题目只限制坐标绝对值, 不限制活动范围),
# 用同一六角格点铺到 2800 m = 1800 m(圆域) + 1000 m(有效接收半径)。
COVER_LATTICE = ops.build_lattice(ops.LATTICE_SPACING, 2800.0)
# 覆盖判据里的"有效覆盖半径": 占用网格的格点代表 100 m×100 m 的区域, 格心被覆盖
# 不等于圆域内任一点都被覆盖。把圆域按 25 m 采样, 测出"任一点到最近格心"的最大
# 距离 δ_max = 106 m (边界处格心被 r<=1800 裁掉, 所以比格内半对角 71 m 更大)。
# 取 1000 - 106 - 4 = 890 m, 则"格心被 890 m 覆盖 ⇒ 圆域内任一点到该测点距离
# ≤ 996 m ≤ R (题目给定 R>=1000)" —— 这才让"覆盖完成 ⇒ 必被发现"成为严格结论。
COVER_RADIUS_M = RADIUS_MIN_M - 110.0
# 信息点搜索用的粗格点: 全格点约 1000 个, 每点要做 20 次"格点核积分",
# 逐步全算会让单局慢一个数量级; 每 4 个取 1 个 (间距 3.2 km / 4 ≈ 800 m → 2.2 km)
# 对"去哪儿测一轮"这种粗决策完全够用。
INFO_LATTICE = LATTICE[::4]


class SweepConfig:
    def __init__(self, **kw) -> None:
        """支持 ``SweepConfig(sweep_max_ch=10)`` 形式的覆盖, 便于消融。"""
        for k, v in kw.items():
            if not hasattr(type(self), k):
                raise AttributeError("SweepConfig has no attribute %r" % k)
            setattr(self, k, v)

    # 初始盲扫: 起点一次把所有"可能"频道测一轮 (原点覆盖半径大, 单位时间发现率最高)
    initial_all: bool = True
    initial_max_ch: int = 20
    # 清除后原地补测
    sweep_on_clear: bool = True
    sweep_max_ch: int = 18
    sweep_pd_min: float = 0.10        # 只测"这一测很可能收到"的频道
    sweep_pi_min: float = 0.10
    sweep_skip_std: float = 120.0     # 已定位且不确定度小于该值的频道不再补测
    sweep_w_new: float = 1.0          # 扫一轮里"找新源"的权重
    sweep_w_ray: float = 0.6          # "给单射线补第二条基线"的权重
    sweep_w_ref: float = 0.25         # "给已定位的降不确定度"的权重
    # 目标选择
    pi_pursue: float = 0.25
    clear_std: float = 26.0
    pursue_max_steps: int = 14
    require_localized: bool = False   # True: 只追"已定位"的源; 实测 False 更快
    fast_pursue: bool = True          # 用 pursue_fast (带 150 m 垂直偏移的近距检测)
    leg_step_m: float = 500.0         # 推进时每次走多远 (顺路补测)
    standoff_m: float = 420.0         # 距估计点多远时做垂直偏移检测
    lateral_m: float = 30.0           # 垂直偏移量
    fix_step_m: float = 30.0          # 清除失败后的修正步长 (省约 2x 步长的移动)
    # --- 弦式 (bow) 纵深定向检测 (推导见 pursue_fast 的 docstring) ---
    bow: bool = True                  # False: 退化为固定侧移后折返 (消融对照)
    bow_rad_th_m: float = 60.0        # 径向 (纵深) 不确定度大于该值才值得做 bow
    bow_d_bow_m: float = 600.0        # 只在 d <= 该值时 bow (更远时沿视线推进更便宜)
    bow_lat_frac: float = 0.0         # >0: 侧移 delta = bow_lat_frac*d (线性, 推荐)
                                      # <=0: 退回 delta = SIGMA_RAD*(d/2)^2/bow_sigma_target_m
    bow_sigma_target_m: float = 6.0   # 仅 bow_lat_frac<=0 时有意义: 目标纵深标准差
    bow_lat_min_m: float = 10.0
    bow_lat_max_m: float = 400.0
    tsp_order: bool = True            # 目标排序用 2-opt 巡回而不是贪心最近
    # 信息点 (没有已定位目标时去哪)
    info_w_new: float = 1.0           # 首次发现的价值权重
    info_w_ray: float = 0.8           # 给单射线频道补第二条示向度的价值权重
    info_value_s: float = 260.0       # 单位价值折算的虚拟秒 (用于旅行成本比较)
    info_max_ch: int = 8
    info_pd_min: float = 0.05
    # 顺路信息停车 (去清除的路上)
    leg_stop: bool = True
    leg_fracs: Tuple[float, ...] = (0.35, 0.65, 0.9)
    # 专程探测
    probe_max: int = 40
    probe_v_source: float = 1500.0    # 收尾覆盖的"质量→秒"汇率
    probe_max_ch: int = 8
    # --- 收尾覆盖 (见 cover_point 的 docstring) ---
    hunt_max_ch: int = 20             # 一趟收尾探测测满全部频道 (旅行成本不变)
    hunt_here: bool = True            # 当前位置参与收尾探测点竞争 (零移动选项)
    # 覆盖收尾: 题目给定有效接收半径 R>=1000 m, 所以在任一点测过的频道, 只要该点
    # 到源的距离不超过 1000 m 就**必然**收到信号。于是"每个未清除频道都被一组
    # 1000 m 覆盖的测点覆盖"是"源必被找到"的充分条件 —— 这条判据只用题目给的
    # 下界, 不依赖 R~U(1000,1500) 的分布假设, 是"确保全部清除"的真正保障。
    coverage_finish: bool = True      # 信念质量吃完后, 仍按 1000 m 覆盖补齐
    coverage_max_steps: int = 20      # 覆盖收尾的最大补测趟数 (安全网)
    min_sources: int = 10             # 题目给定的源数下界 (10~16), 清不到 10 个不能停
    max_actions: int = 200
    max_virtual: float = 90000.0
    ch_cost_s: float = 6.0            # 单频道检测耗时 (5 s 检测 + 1 s 切换)
    hunt: bool = True                 # 没有"划算"的信息点时, 仍按集合覆盖把残余质量扫掉
    hunt_mass_min: float = 0.00       # 残余期望源数低于该值即停止"信念驱动"的补扫。
                                      # 启用覆盖收尾 (coverage_finish) 之后, 清除比例由覆盖
                                      # 判据保证 (恒为 1), 该门限只影响信念补扫的早晚:
                                      # 9 个取值的平均时间差异 < 3%, 0.00 (扫到信念质量
                                      # 吃完再交给覆盖收尾) 在标定集/独立集上都最快。
    hunt_fail_max: int = 12           # 连续多少次专程探测"什么也没测到"就收手
    # --- 单点联合选择: 一条 leg 上同时决定推进多远与侧移多少 ---------------
    # 把"顺路停车点 (leg_info_stop 的 3 个固定比例) + 弦式侧移 (pursue_fast 的 bow)"
    # 两个手挑决定合并成**一个** (eps, delta) 联合优化:
    #   min  实际旅行秒(经过 q 再走向目标) + 6 s 检测 + E[剩余清除代价 | 后验]
    # 代价函数与 sigma 预测由真实模拟器标定得到。
    joint_choice: bool = True
    joint_eps: Tuple[float, ...] = (0.0, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9)
    joint_dl_fracs: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.35)   # delta = 比例 x d
    joint_delta_max_m: float = 700.0
    joint_sigma_inflate: float = 1.0      # 解析后验 sigma 的保守放大 (mode2 偏乐观)
    joint_sigma_floor_frac: float = 0.15  # 最多只敢声称把当前 sigma 降到该比例
    joint_sig_cap_m: float = 600.0        # 代价函数里 sigma 的截断 (标定范围外不外推)
    joint_g1_a: float = 90.0              # 单射线: 开销秒 ~ a + b*sigma_rad
    joint_g1_b: float = 0.06
    joint_g2_a: float = 31.0              # 已定位: 开销秒 ~ a + b*sigma_rad
    joint_g2_b: float = 0.10
    joint_margin_s: float = 0.0           # >0: 需比保守动作好这么多秒才偏离
    joint_mode1: bool = True              # 单射线目标上生效
    joint_mode2: bool = True              # 已定位目标上生效 (只在 bow 那一步接管)
    joint_mode2_bow_only: bool = True     # mode2 只在本来就要 bow 的状态接管
                                          # (全面接管 mode2 是实测负收益)


class SweepResult:
    __slots__ = ("cleared", "total", "virtual_s", "steps", "measures", "clears",
                 "failed_clears", "moved_m", "probes", "sweeps", "pursues", "mv_probe")

    def __init__(self, w: World) -> None:
        self.cleared = w.cleared_count
        self.total = w.n_sources
        self.virtual_s = w.virtual_t
        self.measures = w.measures
        self.clears = w.clears
        self.failed_clears = w.failed_clears
        self.moved_m = w.moved_m
        self.steps = 0
        self.probes = 0
        self.sweeps = 0
        self.pursues = 0
        self.mv_probe = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "cleared": self.cleared, "total": self.total, "virtual_s": self.virtual_s,
            "avg_clear_s": self.virtual_s / self.cleared if self.cleared else float("inf"),
            "measures": self.measures, "clears": self.clears,
            "failed_clears": self.failed_clears, "moved_m": self.moved_m,
            "steps": self.steps, "probes": self.probes, "sweeps": self.sweeps,
            "pursues": self.pursues, "mv_probe": self.mv_probe,
        }


def build_cfg() -> "SweepConfig":
    """发布配置: 评测与部署**共用的唯一一份**参数 (两处不一致会导致行为不一致)。

    题面的目标不是标量: 清除比例越大越好、平均定位清除时间越小越好, 且前者是前置目标。
    因此参数按这两个目标的取舍标定, 选点取前沿上边际代价开始陡增处。首批 300 局
    (seed 9500-9799) 上的关键取值与理由如下
    (发布配置随后在扩展标定集 seed 9500-11499 / 2000 局上复核):

      * ``sweep_max_ch=18, sweep_pd_min=0.10``: 清除后原地补测的频道预算 ——
        补测**不产生任何移动**, 所以"每次多测几个频道"比"多跑一趟专程探测"便宜得多;
      * ``sweep_skip_std=120``: 已定位到 120 m 以内的频道不再补测 (实测清点时
        位置误差只有 6.4 m, 而清除半径 20 m);
      * ``lateral_m=30``: 近距离侧移检测的偏移量 (再大反而更慢);
      * ``require_localized=False``: 单射线目标也直接去追 (实测更快);
      * ``bow=True``: 弦式纵深定向检测, 在剩余路段中点按 delta 侧移测一次,
        消掉"走到估计点旁固定侧移再折返"的来回;
      * ``fix_step_m=30``: 清除失败后的修正步长 (每次失败省约 2x 步长的移动);
      * ``probe_v_source=1000`` + ``hunt_max_ch=20``: 收尾覆盖的"质量→秒"汇率与
        单趟频道预算, 两者互补 —— 前者把收尾点选在真正覆盖最多残余质量处, 后者让
        一趟就把该处的残余质量吃干净;
      * ``hunt_here=True``: 当前位置参与收尾探测点竞争, 提供零移动选项;
      * **``sweep_pi_min=0.05`` + ``leg_stop=False``**: 2026-09-13 现场复核后收紧的
        两项 —— 降低"顺路补测"的存在概率门限(原来 0.10 漏掉了不少真正该测的频道,
        导致后面要专程跑一趟), 并取消 3 个固定比例的顺路信息停车点(它们与
        ``joint_choice`` 的联合优化重复, 且实测多走 291 m/局、多测 11.5 次/局)。
        见 ``tools/tune_p3.py``: 标定 500 局 394.4 → **375.3 s/源**, 标定 2000 局
        396.4 → **377.5 s/源**, 独立 1000 局 → **375.4 s/源**, 全新 1000 局
        (21000-21999, 调参未用) 396.4 → **378.4 s/源**, 四处清除比例均 1.0000;
      * **收尾口径: ``coverage_finish=False`` + ``hunt_mass_min=0.05``**
        (2026-09-13 现场定为速度优先 λ=0.05)。信念质量扫完之后按 λ 门限收尾
        (**不**再按 1000 m 覆盖判据补齐), 因此**没有"停机即不漏源"的保证**:

        | 口径 | λ | 清除比例 | 全清率 | s/源 | 3 局跌破 98% 的概率 |
        | --- | --- | --- | --- | --- | --- |
        | **速度优先 (当前发布, λ=0.05)** | **0.05** | **0.9988** | **0.985** | **281.6** | **4.6%** |
        | 速度优先 (更快) | 0.10 | 0.9968 | 0.960 | 268.6 | 11.6% |
        | 速度优先 (更快) | 0.30 | 0.9907 | 0.892 | 250.4 | 29.2% |
        | 保障模式 (必不漏源) | 0.00 | 1.0000 | 1.000 | 377.5 | 0% |

        标定集 2000 局实测 (``tools/tune_p3.py --stage speed``)。**注意**: 速度优先
        没有"停机即不漏源"的保证 —— 3 局正式测试只有约 39 个源, 漏 1 个就把清除比例
        压到 0.974 < 0.98, 上表最后一列就是按 2000 局重抽样估出的失败概率。
        现场按速度优先取 λ=0.05; 要换回"必不漏源", 把 ``coverage_finish`` 改回
        True、``hunt_mass_min`` 调到 0.00 即可 (代价是约 34% 的虚拟时间);
      * ``joint_choice=True``: 把"顺路停车点"与"弦式侧移"两个手挑决定合并成一次
        ``(eps, delta)`` 联合优化 (见 :func:`joint_measure_point`);
      * ``max_actions`` 与 ``max_virtual`` 是**纯安全阀**: 实测每局只走约 15 个宏动作、
        约 3300 虚拟秒, 两者都远离上限, 调它们对成绩没有任何影响。
    """
    return SweepConfig(sweep_max_ch=18, sweep_pd_min=0.10, sweep_pi_min=0.05,
                       sweep_skip_std=120, lateral_m=30, leg_stop=False,
                       require_localized=False, hunt=True,
                       hunt_mass_min=0.05, hunt_fail_max=12, probe_max=40,
                       max_virtual=90000.0,
                       bow=True, fix_step_m=30.0, probe_v_source=1000.0,
                       hunt_max_ch=20, hunt_here=True,
                       joint_choice=True, coverage_finish=False, min_sources=10)


def _pd_all(world: World, p: Vec) -> np.ndarray:
    """一次算出**所有频道**在 p 处的检测概率 (矩阵乘法)。

    原写法对每个频道各做一遍"3200 格点求距离 → 核 → 点积", 20 个频道就是 20 遍;
    而核只取决于距离、与频道无关, 所以把各频道的位置分布堆成一个矩阵做一次乘法即可。
    单局 profile 里 ``belief.detect_prob`` 占 79% 的时间, 这里直接消掉大头。

    **已知的一处口径不一致 (T7 复核发现, 未修)**: 这里 mode 0 用的是"格点分布 × 先验核"
    的点积, 而 ``Belief.on_measure`` 在算 ``no_signal`` 的似然时, mode 0 分支用的是
    ``ChannelTrack.detect_prob``。两者在 R 的先验下数值相同, 但一旦把 R 换成由观测
    截断的后验 (T7 方案) 就会分叉 —— 这正是 T7 必须同时改三处的原因。
    当前没有启用截断核 (T7 实测部署口径 +2.0 s/源, 汇率不划算),
    因此这处不一致**不影响**实际行为。
    """
    bel = world.belief
    cells = bel.pts
    d = np.hypot(cells[:, 0] - p[0], cells[:, 1] - p[1])
    kernel = np.clip((RADIUS_MAX_M - d) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0).astype(np.float32)
    out = np.zeros(N_CHANNELS, dtype=np.float64)
    idx: List[int] = []
    mats: List[np.ndarray] = []
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared:
            continue
        if tr.mode == 0 and tr.grid is not None:
            idx.append(i)
            mats.append(tr.grid)
        else:
            out[i] = tr.detect_prob(p, cells, None)
    if idx:
        G = np.stack(mats, axis=0)
        out[idx] = G @ kernel
    return out


def _unresolved(world: World, cfg: SweepConfig) -> List[int]:
    """还没发现 (mode 0) 的频道。"""
    out = []
    for i in range(N_CHANNELS):
        tr = world.belief.tracks[i]
        if tr.cleared or tr.pi < cfg.sweep_pi_min or tr.is_measured((world.x, world.y)):
            continue
        out.append(i + 1)
    return out


def pick_channels(world: World, p: Vec, cfg: SweepConfig, max_ch: int,
                  pd_min: float, pi_min: Optional[float] = None) -> List[int]:
    """在点 p 处"值得测"的频道 (按 存在概率 × 检测概率 排序, 限量)。

    ``pi_min`` 可单独放宽: 收尾清扫时必须用 0, 否则存在概率被 no_signal 压到
    ``sweep_pi_min`` 以下的频道**永远不会再被测**, 那个源就必然漏掉。
    """
    bel = world.belief
    pi_floor = cfg.sweep_pi_min if pi_min is None else pi_min
    scored = []
    pds = _pd_all(world, p)
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared or tr.pi < pi_floor or tr.is_measured(p):
            continue
        # 已定位 (且马上就能清) 的频道不再补测: 实测每源平均拿到 4.28 条示向度,
        # 而清点时位置误差只有 6.4 m (清除半径 20 m) —— 多出来的检测全是浪费。
        if tr.mode == 2 and tr.std_max() <= cfg.sweep_skip_std:
            continue
        pd = float(pds[i])
        if pd < pd_min:
            continue
        # 价值加权: 找新源 > 给单射线补基线 > 给已定位的降误差。
        # 原来统一用 pi*pd 排序会让"已经发现 (pi=1)"的频道挤掉"还没发现 (pi~0.3)"的,
        # 而后者才是不再需要专程探测的关键。
        if tr.mode == 0:
            w = cfg.sweep_w_new
        elif tr.mode == 1:
            w = cfg.sweep_w_ray
        else:
            w = cfg.sweep_w_ref
        scored.append((w * float(tr.pi) * pd, i + 1))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    return ops.order_channels(world, [c for _v, c in scored[:max_ch]])


def sweep_here(world: World, cfg: SweepConfig, max_ch: Optional[int] = None,
               pd_min: Optional[float] = None) -> int:
    """在当前点补测"值得测"的频道 (清除后调用, 无额外移动)。"""
    p = (world.x, world.y)
    chans = pick_channels(world, p, cfg,
                          cfg.sweep_max_ch if max_ch is None else max_ch,
                          cfg.sweep_pd_min if pd_min is None else pd_min)
    if not chans:
        return 0
    ops.probe(world, p, channels=chans)
    return len(chans)


def probe_at(world: World, p: Vec, cfg: SweepConfig, max_ch: int,
             pd_min: float = 0.0, info: bool = False) -> int:
    """移动到 p 并测一轮 (p != 当前点时才会产生移动)。

    ``info=True`` 时用信息探测的判据选频道 (与 ``info_score`` 一致);
    没有值得测的频道时**直接返回 0, 不做空测** —— 空测只花 5 s 不产生任何信息。
    """
    if info:
        chans = ops.order_channels(world, [c for _v, c in info_channels(world, p, cfg)[:max_ch]])
    else:
        chans = pick_channels(world, p, cfg, max_ch, pd_min)
    if not chans:
        return 0
    ops.probe(world, p, channels=chans)
    return len(chans)


def is_localized(world: World, c: int, cfg: SweepConfig) -> bool:
    tr = world.belief.tracks[c - 1]
    if tr.cleared or tr.est is None or tr.pi < cfg.pi_pursue:
        return False
    if tr.near_pt is not None:
        return True
    return tr.mode == 2 and tr.std_max() <= cfg.clear_std


def _tour_first(world: World, cfg: SweepConfig, cand: List[int]) -> Optional[int]:
    """在候选目标集合上做最近邻 + 2-opt, 返回**路线上第一个**该去的频道。

    为什么需要: 贪心"就近清除"在大圆域上比最优巡回长约 20%~25% (实测 11082 m vs
    TSP 9096 m, 即每局约 400 虚拟秒)。目标只有十几个, 直接做 2-opt 很便宜。
    """
    pts = [(world.x, world.y)]
    order = list(cand)
    for c in order:
        tr = world.belief.tracks[c - 1]
        pts.append(tr.est if tr.est is not None else (world.x, world.y))
    m = len(order)
    if m == 0:
        return None
    if m == 1:
        return order[0]
    idx = list(range(m))
    # 最近邻
    seq = []
    cur = -1
    left = set(idx)
    while left:
        nxt = min(left, key=lambda j: math.hypot(pts[j + 1][0] - pts[cur + 1][0],
                                                 pts[j + 1][1] - pts[cur + 1][1]))
        seq.append(nxt)
        left.discard(nxt)
        cur = nxt

    def path_len(s: List[int]) -> float:
        tot = 0.0
        prev = pts[0]
        for j in s:
            q = pts[j + 1]
            tot += math.hypot(q[0] - prev[0], q[1] - prev[1])
            prev = q
        return tot

    improved = True
    best = path_len(seq)
    while improved:
        improved = False
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                trial = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                L = path_len(trial)
                if L < best - 1e-9:
                    seq, best, improved = trial, L, True
    return order[seq[0]]


def pick_target(world: World, cfg: SweepConfig, blocked: Optional[set] = None) -> Optional[int]:
    cand: List[int] = []
    for i in range(N_CHANNELS):
        tr = world.belief.tracks[i]
        if tr.cleared or tr.est is None or tr.pi < cfg.pi_pursue:
            continue
        if blocked and (i + 1) in blocked:
            continue
        if cfg.require_localized and not is_localized(world, i + 1, cfg):
            continue
        cand.append(i + 1)
    if not cand:
        return None
    return _tour_first(world, cfg, cand)


def info_channels(world: World, p: Vec, cfg: SweepConfig) -> List[Tuple[float, int]]:
    """在 p 处"信息探测"该测哪些频道 (按价值排序)。

    **必须与 ``info_score`` 用同一套判据**: 曾经 ``info_score`` 认为"有 2 个频道值得测",
    而实际执行走的是 ``pick_channels`` (另有一套 pi/pd/std 过滤), 结果一个都不测 ——
    于是原地反复"空测" 5 s 一次, 白白烧掉十几步。选频道的判据只能有一处。
    """
    bel = world.belief
    pds = _pd_all(world, p)
    out: List[Tuple[float, int]] = []
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared or tr.pi < 1e-4 or tr.is_measured(p):
            continue
        pd = float(pds[i])
        if pd < cfg.info_pd_min:
            continue
        if tr.mode == 0:
            w = cfg.info_w_new
        elif tr.mode == 1:
            w = cfg.info_w_ray
        else:
            s0 = tr.std_max()
            if s0 < cfg.clear_std * 1.5:
                continue
            w = cfg.info_w_ray
        out.append((w * float(tr.pi) * pd, i + 1))
    out.sort(key=lambda kv: (-kv[0], kv[1]))
    return out


def info_score(world: World, p: Vec, cfg: SweepConfig) -> Tuple[float, int]:
    """在 p 处测一轮的"信息价值" (折算成期望秒数) 与推荐频道数。"""
    sc = info_channels(world, p, cfg)
    val = sum(v for v, _c in sc)
    return val * cfg.info_value_s, min(len(sc), cfg.info_max_ch)


def best_info_point(world: World, cfg: SweepConfig,
                    cands: Optional[Sequence[Vec]] = None
                    ) -> Optional[Tuple[Vec, float, int, float]]:
    """没有已定位目标时: 去哪测一轮信息量最大。

    返回 ``(点, 信息价值, 频道数, 净收益)``, 其中
    ``净收益 = 信息价值 − 旅行秒 − 检测秒``。**只有净收益为正才值得专程跑一趟** ——
    早先只判断"信息价值 > 0", 于是会出现"为了 0.2 的期望质量跑 300 s"这种亏本探测,
    正式测试里(看不到真值总数, 停不下来)尤其致命。
    """
    pos = (world.x, world.y)
    pool = list(cands) if cands is not None else [tuple(p) for p in INFO_LATTICE]
    best: Optional[Tuple[float, Vec, float, int]] = None
    for p in pool:
        val, n_ch = info_score(world, p, cfg)
        if n_ch <= 0 or val <= 0.0:
            continue
        travel = ops.dist(pos, (float(p[0]), float(p[1]))) / SPEED_MPS
        net = val - travel - n_ch * 6.0
        if best is None or net > best[0]:
            best = (net, (float(p[0]), float(p[1])), val, n_ch)
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def leg_info_stop(world: World, dest: Vec, cfg: SweepConfig,
                  target: Optional[int] = None) -> Optional[Tuple[Vec, int]]:
    """去清除的路上顺路测一轮 (绕路为 0, 只要信息价值 > 检测耗时)。

    ``target``: 当前 leg 追的频道 (仅联合选择用)。``cfg.joint_choice=True`` 且目标是单射线
    (mode==1) 时, 停车点不再从 3 个固定比例里挑, 而是由 :func:`joint_measure_point`
    联合优化 (eps, delta) —— 因为单射线的"第二条示向度"必须**离开第一条射线**才有
    三角基线 (实测发布版停在射线上: 基线 delta 中位 **0 m**, 转换后 std_max 中位
    **897 m**, 即这次检测几乎没带来信息)。
    """
    pos = (world.x, world.y)
    leg = ops.dist(pos, dest)
    if leg < 250.0:
        return None
    if cfg.joint_choice and target is not None and cfg.joint_mode1:
        tr = world.belief.tracks[target - 1]
        if tr.mode == 1:
            q = joint_measure_point(world, target, cfg, tr, leg=True)
            if q is not None:
                val, n_ch = info_score(world, q, cfg)
                if n_ch > 0 and val - n_ch * 6.0 > 0.0:
                    return q, n_ch
    best: Optional[Tuple[float, Vec, int]] = None
    for f in cfg.leg_fracs:
        p = (pos[0] + (dest[0] - pos[0]) * f, pos[1] + (dest[1] - pos[1]) * f)
        val, n_ch = info_score(world, p, cfg)
        if n_ch <= 0:
            continue
        net = val - n_ch * 6.0
        if best is None or net > best[0]:
            best = (net, p, n_ch)
    if best is None or best[0] <= 0.0:
        return None
    return best[1], best[2]


def cover_point(world: World, cfg: SweepConfig, force: bool = False) -> Optional[Tuple[Vec, float]]:
    """没有目标时的专程探测点: 逐频道"未被 1000 m 覆盖"的质量加权覆盖。

    ``cfg.hunt_here`` 额外把**机器狗当前位置**作为候选点。理由: "把这里还没测过的
    频道测掉"是**零移动**的探测, 而发布版只从 800 m 六角格点里挑点 —— 当局部质量已经
    被吃完时, 格点上的赢家就退化成"离我一步远的格点", 即固定 800 m 的往返。
    **数字更正 (T5 复核)**: 早期记录的"5.50 趟/局、4134 m/局, 其中 40.5% 落在 700~900 m,
    220 趟里只有 22 趟测到示向度"是 `hunt_here` 加入**之前**的工况; 当前发布版在
    9500-9539 只有 **1.38 趟/局、1617 m/局**。注意本函数是按**信念残余质量**选点,
    它不能证明"没有漏源": 质量被信念削小不等于圆域真被看过。因此发布版在它之后还接
    一层 :func:`coverage_point` 的 1000 m 覆盖收尾 (题目给定 R>=1000 m, 覆盖即必被发现)。
    """
    bel = world.belief
    cells = bel.pts
    mass = np.zeros(cells.shape[0], dtype=np.float64)
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared or tr.grid is None or tr.pi < 1e-4:
            continue
        unc = np.ones(cells.shape[0], dtype=bool)
        for (px, py) in tr.no_pts:
            unc &= (np.hypot(cells[:, 0] - px, cells[:, 1] - py) > RADIUS_MIN_M)
        if unc.any():
            mass += float(tr.pi) * (tr.grid.astype(np.float64) * unc)
    if mass.sum() <= 1e-9:
        return None
    pos = (world.x, world.y)
    best: Optional[Tuple[float, Vec, float]] = None
    for p in LATTICE:
        m = np.hypot(cells[:, 0] - p[0], cells[:, 1] - p[1]) <= RADIUS_MIN_M
        gain = float(mass[m].sum())
        if gain <= 1e-6:
            continue
        travel = ops.dist(pos, (float(p[0]), float(p[1]))) / SPEED_MPS
        value = gain * cfg.probe_v_source - travel - 40.0
        if best is None or value > best[0]:
            best = (value, (float(p[0]), float(p[1])), gain)
    if cfg.hunt_here:
        # 当前位置: travel = 0。只有"这里还有没测过的频道"时 gain 才非零
        # (``pick_channels`` 会按 pi_min=0 取, 但已经把测过的点排除掉了)。
        m = np.hypot(cells[:, 0] - pos[0], cells[:, 1] - pos[1]) <= RADIUS_MIN_M
        gain = float(mass[m].sum())
        if gain > 1e-6:
            value = gain * cfg.probe_v_source - 40.0
            if best is None or value > best[0]:
                best = (value, (float(pos[0]), float(pos[1])), gain)
    if best is None:
        return None
    if not force and best[0] <= 0.0:
        return None
    return best[1], best[2]


def coverage_point(world: World, cfg: SweepConfig) -> Optional[Tuple[Vec, float]]:
    """收尾补测点: 按**1000 m 覆盖**判据选点, 而不是按信念残余质量。

    题目给定有效接收半径 R >= 1000 m。因此在点 p 测过频道 c 之后, 只要 c 的真实源
    与 p 的距离不超过 1000 m, 就**必然**收到信号。于是

        "每个未清除频道都被一组测过它的点以 1000 m 覆盖"

    是该频道一定被发现过的**充分条件**; 收尾一直做到这条成立, 就得到"确保全部清除"
    的保障。这条判据只用题目给出的下界 R>=1000, 不依赖 R~U(1000,1500) 的分布假设。

    返回 (补测点, 残余未覆盖量); 已无残余时返回 None。每个频道的覆盖掩码按
    "新增测点增量更新"缓存, 因此单次调用只与新增测点有关。
    """
    bel = world.belief
    cells = bel.pts
    n = cells.shape[0]
    unc = np.zeros(n, dtype=np.float64)
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared or tr.pi <= 0.0:
            continue
        mask = getattr(tr, "_cov_mask", None)
        if mask is None:
            mask = np.zeros(n, dtype=bool)
        done = int(getattr(tr, "_cov_done", 0))
        fresh = tr.meas_pts[done:]
        for (px, py) in fresh:
            mask |= (np.hypot(cells[:, 0] - px, cells[:, 1] - py) <= COVER_RADIUS_M)
        if fresh:
            tr._cov_done = len(tr.meas_pts)     # type: ignore[attr-defined]
            tr._cov_mask = mask                 # type: ignore[attr-defined]
        unc += (~mask)
    total = float(unc.sum())
    if total <= 0.0:
        return None
    pos = (world.x, world.y)
    best: Optional[Tuple[float, Vec, float]] = None
    for p in COVER_LATTICE:
        m = np.hypot(cells[:, 0] - p[0], cells[:, 1] - p[1]) <= COVER_RADIUS_M
        g = float(unc[m].sum())
        if g <= 0.0:
            continue
        travel = ops.dist(pos, (float(p[0]), float(p[1]))) / SPEED_MPS
        val = g / max(travel + cfg.ch_cost_s * min(cfg.hunt_max_ch, N_CHANNELS), 1.0)
        if best is None or val > best[0]:
            best = (val, (float(p[0]), float(p[1])), g)
    if best is None:
        return None
    return best[1], total


def _radial_std(tr, ux: float, uy: float) -> float:
    """沿视线方向 (ux,uy) 的位置 1-σ (m)。无协方差时返回 +inf。"""
    if tr.cov is None:
        return float("inf")
    C = tr.cov
    return math.sqrt(max(ux * ux * C[0, 0] + 2.0 * ux * uy * C[0, 1] + uy * uy * C[1, 1], 0.0))




# ======================================================================================
# 单点联合选择: 一条 leg 上"测哪儿"的**单一联合决策** —— 用 (eps, delta) 一个点同时买"推进"和
# "垂直基线"。默认关 (joint_choice=False), 关掉时下面这些函数永不被调用。
# ======================================================================================
JOINT_STATS: Dict[str, float] = {"calls": 0, "leg_calls": 0, "pursue_calls": 0,
                                 "m1": 0, "m2": 0, "delta_sum": 0.0,
                                 "eps_sum": 0.0, "n_zero_delta": 0}


def _joint_pred(tr, q: Vec, cfg: "SweepConfig"):
    """预测"在 q 再测一次"之后的后验 (mode_if_hit, sigma_rad, sigma_max, P(收到))。

    **用信念自己的信息矩阵** (``ChannelTrack._info_matrix``): 把 (q, 指向 est 的合成
    示向度) 加进观测集求逆 —— 与 ``ChannelTrack.refit`` 是同一套公式, 因此"共线测量
    不压缩纵深"这条几何是自动成立的 (不需要手写公式)。

    实测标定 (首批 300 局 / 6007 次目标频道检测):
      * mode 1: 实际/预测 sigma 比 中位 **1.00** (准确);
      * mode 2: 中位 **1.07**、p90 **5.5** (偏乐观, 尾部很重) —— 因此有
        ``joint_sigma_inflate`` 与 ``joint_sigma_floor_frac`` 两个保守化旋钮,
        且代价函数里 sigma 被 ``joint_sig_cap_m`` 截断。
    """
    if tr.est is None or tr.cov is None:
        return None
    ex, ey = tr.est
    b = math.degrees(math.atan2(ey - q[1], ex - q[0])) % 360.0
    obs = list(tr.obs) + [(float(q[0]), float(q[1]), b)]
    try:
        Info = ChannelTrack._info_matrix(obs, ex, ey)
    except Exception:                                    # noqa: BLE001
        return None
    det = Info[0, 0] * Info[1, 1] - Info[0, 1] * Info[1, 0]
    if not (det > 1e-30):
        return None
    C = np.array([[Info[1, 1], -Info[0, 1]], [-Info[1, 0], Info[0, 0]]]) / det
    vx, vy = ex - q[0], ey - q[1]
    n = math.hypot(vx, vy)
    vx, vy = (vx / n, vy / n) if n > 1e-9 else (1.0, 0.0)
    s_rad = math.sqrt(max(vx * vx * C[0, 0] + 2.0 * vx * vy * C[0, 1] + vy * vy * C[1, 1], 0.0))
    s_max = math.sqrt(max(max_eig(C), 0.0))
    fl = cfg.joint_sigma_floor_frac
    s_rad = max(s_rad * cfg.joint_sigma_inflate, _radial_std(tr, vx, vy) * fl, 3.0)
    s_max = max(s_max * cfg.joint_sigma_inflate, tr.std_max() * fl, 3.0)
    pd = 1.0
    try:
        if tr.mode == 1:
            dd = np.hypot(tr.ray_x[:, 0] - q[0], tr.ray_x[:, 1] - q[1])
            k = np.clip((RADIUS_MAX_M - dd) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0)
            pd = float(np.dot(tr.ray_w, k))
        else:
            pd = float(tr.detect_prob(q, None, None))
    except Exception:                                    # noqa: BLE001
        pd = 1.0
    return (2 if tr.mode == 2 else 1), s_rad, s_max, min(max(pd, 0.0), 1.0)


def _joint_rem(mode: int, sig_rad: float, cfg: "SweepConfig") -> float:
    """E[剩余清除代价 | 后验] 里"超出直线逼近"的那部分 (秒), 经验标定,


    mode1: 90 + 0.06·σ_rad (实测 100~113 s, 与 σ 弱相关 —— 单射线 leg 无论 σ 多大都要
           多花约 100 s 才收得掉);
    mode2: 31 + 0.10·σ_rad (实测 35/32/51/58/82/82/92 s 对应 σ 7/22/45/90/185/350/600+)。
    σ 截断在 ``joint_sig_cap_m``: 标定数据只覆盖 σ≤600, 外推不做。
    """
    if mode == 1:
        return cfg.joint_g1_a + cfg.joint_g1_b * min(max(sig_rad, 0.0), cfg.joint_sig_cap_m)
    return cfg.joint_g2_a + cfg.joint_g2_b * min(max(sig_rad, 0.0), cfg.joint_sig_cap_m)


def _joint_baseline_point(world: World, cfg: "SweepConfig", tr, d: float,
                          ux: float, uy: float, px: float, py: float) -> Vec:
    """发布版 ``pursue_fast`` 在同一状态会选的测点 —— 保证联合候选集里永远含它。"""
    if cfg.bow and d <= cfg.bow_d_bow_m and _radial_std(tr, ux, uy) > cfg.bow_rad_th_m:
        half = 0.5 * d
        if cfg.bow_lat_frac > 0.0:
            delta = cfg.bow_lat_frac * d
        else:
            delta = SIGMA_RAD * half * half / max(cfg.bow_sigma_target_m, 0.1)
        delta = min(max(delta, cfg.bow_lat_min_m), cfg.bow_lat_max_m)
        return (world.x + ux * half + px * delta, world.y + uy * half + py * delta)
    if d > cfg.standoff_m:
        step = min(d - cfg.standoff_m, cfg.leg_step_m)
        return (world.x + ux * step, world.y + uy * step)
    return (tr.est[0] + px * cfg.lateral_m, tr.est[1] + py * cfg.lateral_m)


def joint_measure_point(world: World, c: int, cfg: "SweepConfig", tr=None,
                        leg: bool = False) -> Optional[Vec]:
    """联合选择的核心: 在一条 leg 上联合选**一个**测点 q = pos + u·(eps·d) + n·delta。

    目标 (eps, delta 同时优化, 不是两个手挑决定):
        J(q) = |pos→q|/v + |q→est|/v            ← **真实几何绕行** (经过 q 再走向目标)
             + 6 s                              ← 这一次检测
             + E[剩余清除代价 | 后验(q)]         ← 经验标定的 ``_joint_rem``,
                                                   按检测概率对"收到/没收到"加权
    候选集 = (eps 格点 × delta 格点, 两侧) ∪ {发布版动作点} —— 因此 flag 打开**永远不会**
    因为模型偏差而比发布版动作更差(在模型的意义下); ``joint_margin_s>0`` 还能再加一档保守。
    """
    if tr is None:
        tr = world.belief.tracks[c - 1]
    if tr.est is None or tr.cov is None:
        return None
    ex, ey = tr.est
    dx, dy = ex - world.x, ey - world.y
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return None
    ux, uy = dx / d, dy / d
    px, py = -uy, ux
    s_now = _radial_std(tr, ux, uy)
    eps_list = list(cfg.joint_eps) + (list(cfg.leg_fracs) if leg else [])
    base = None if leg else _joint_baseline_point(world, cfg, tr, d, ux, uy, px, py)
    cands: List[Vec] = []
    seen = set()

    def _add(q: Vec) -> None:
        key = (round(q[0] / 0.5), round(q[1] / 0.5))
        if key in seen or tr.is_measured(q):
            return
        seen.add(key)
        cands.append(q)

    for f in eps_list:
        a = f * d
        _add((world.x + ux * a, world.y + uy * a))
        for dl in cfg.joint_dl_fracs:
            delta = min(dl * d, cfg.joint_delta_max_m)
            _add((world.x + ux * a + px * delta, world.y + uy * a + py * delta))
            _add((world.x + ux * a - px * delta, world.y + uy * a - py * delta))
    if base is not None:
        _add(base)
    if not cands:
        return None
    best, best_j = None, float("inf")
    for q in cands:
        dq = ops.dist((world.x, world.y), q)
        Lq = math.hypot(ex - q[0], ey - q[1])
        pred = _joint_pred(tr, q, cfg)
        if pred is None:
            rem = _joint_rem(tr.mode, s_now, cfg)
        else:
            m_hit, s_hit, _sm, pdet = pred
            rem = pdet * _joint_rem(m_hit, s_hit, cfg) + (1.0 - pdet) * _joint_rem(tr.mode, s_now, cfg)
        j = (dq + Lq) / SPEED_MPS + 6.0 + rem
        if j < best_j - 1e-12:
            best, best_j = q, j
    if best is None:
        return None
    if base is not None and cfg.joint_margin_s > 0.0:
        dq = ops.dist((world.x, world.y), base)
        Lq = math.hypot(ex - base[0], ey - base[1])
        pred = _joint_pred(tr, base, cfg)
        if pred is None:
            rem = _joint_rem(tr.mode, s_now, cfg)
        else:
            m_hit, s_hit, _sm, pdet = pred
            rem = pdet * _joint_rem(m_hit, s_hit, cfg) + (1.0 - pdet) * _joint_rem(tr.mode, s_now, cfg)
        if best_j > (dq + Lq) / SPEED_MPS + 6.0 + rem - cfg.joint_margin_s:
            best = base
    # 统计 (纯诊断, 不影响行为)
    JOINT_STATS["calls"] += 1
    JOINT_STATS["leg_calls" if leg else "pursue_calls"] += 1
    JOINT_STATS["m1" if tr.mode == 1 else "m2"] += 1
    _eps = ((best[0] - world.x) * ux + (best[1] - world.y) * uy) / d if d > 1e-9 else 0.0
    _dl = abs((best[0] - world.x) * px + (best[1] - world.y) * py)
    JOINT_STATS["eps_sum"] += _eps
    JOINT_STATS["delta_sum"] += _dl
    if _dl < 1.0:
        JOINT_STATS["n_zero_delta"] += 1
    return best


def pursue_fast(world: World, c: int, cfg: SweepConfig,
                bow_override: Optional[Tuple[float, float]] = None) -> bool:
    """逼近并清除频道 c (不绕远路的版本)。

    ``bow_override=(eps_frac, delta_m)``: 覆盖弦式测点的位置与幅度 (供决策时规划
    枚举候选动作)。默认 ``None`` 时为标准弦式侧移。

    依据 "已知路线 + 真实测量物理" 的下界扫描 (``oracle_route``):
      * 两次**共线**检测 (都在路上) 的深度误差约 330 m, 清除必然失败;
      * 因此必须在某处制造一条**垂直于视线**的基线才能把纵深定下来;
      * 其余检测全部放在"本来就要走"的路上, 因此移动几乎不增加。

    **纵深定向检测的两种做法** (``cfg.bow`` 切换):

    * ``bow=False`` (消融对照): 走到估计点旁固定 ``lateral_m`` 处测一次, 再**折返**
      到估计点。这是一次来回, 每次检测多走约 ``2*lateral_m`` 的纯绕行。
    * ``bow=True`` (默认): 在**剩余路段的中点**、按
      ``delta = sigma_theta * (d/2)^2 / sigma_target`` 定标的侧移处测一次。
      几何: 在离估计点沿视线 eps、侧向 delta 处测一次, 纵深被压到
      ``sigma_depth ~ sigma_theta * (eps^2 + delta^2) / delta``; 取 eps = d/2 且让
      ``delta ∝ d^2`` 就使 **sigma_depth ≈ sigma_target 与 d 无关**, 而绕行只有
      ``2*(sqrt((d/2)^2+delta^2) - d/2) ~ delta^2/d``, 比来回小一个数量级。
      固定 30 m 侧移在估计点本身有 e 的纵深误差时只能给出
      ``sigma_theta*e^2/30`` —— 越远越差, 这正是它要多测几次的原因。

    实测 (首批 300 局 9500-9799, 进程内): 平均定位清除 258.1 -> 250.5 s/源, 清除率
    0.9987 -> 0.9992, 移动 12061 -> 11621 m; 每 pursuit 检测 1.37 -> 1.20 次。
    ``bow=False`` 退化为固定侧移后折返 (仅作消融对照)。

    **纵深精度的正确公式** (用真实 ``refit`` 数值验证):
    ``sigma_depth ~ sigma_theta * D1 * L / delta``, 其中 D1 是旧测点到源的距离,
    L 是新测点到源的距离, delta 是新测点相对旧视线的**垂直**偏移。要点:
      * delta 越大精度越高, 但**饱和**在 ``~sigma_theta*D1`` 附近, 因此
        ``lateral_m`` 从 30 加到 200 几乎没有收益 (实测 +1.4 s) —— 这是个死旋钮;
      * 重要的是 delta 与 L 的比例。固定侧移在 ``eps=0, delta=L=30`` 处正好取到
        饱和点附近, 这解释了为什么它"30 m 就够了", 也解释了它为什么在估计点本身
        有纵深误差 e 时会退化 (此时有效基线变小)。
    """
    standoff = cfg.standoff_m
    lat = cfg.lateral_m
    rad_th = cfg.bow_rad_th_m
    d_bow = cfg.bow_d_bow_m
    lat_frac = cfg.bow_lat_frac
    sig_tgt = cfg.bow_sigma_target_m
    lat_min = cfg.bow_lat_min_m
    lat_max = cfg.bow_lat_max_m
    last_obs = world.belief.tracks[c - 1].n_obs
    stall = 0
    for _ in range(cfg.pursue_max_steps):
        tr = world.belief.tracks[c - 1]
        if tr.cleared:
            return True
        if tr.near_pt is not None:
            world.clear(tr.near_pt[0], tr.near_pt[1], c)
            continue
        # 停滞保护: 连续 3 步没有拿到任何新观测 ⇒ 这条轨迹是假的 (估计点附近没有源),
        # 再测下去只是白烧时间。`ops.pursue` 一直有这个保护, `pursue_fast` 早先漏了,
        # 结果个别案例会无限重试 (实测 seed 9628: 15 330 s / 2818 次检测)。
        if tr.n_obs <= last_obs:
            stall += 1
            if stall >= 3:
                return False
        else:
            stall = 0
        last_obs = tr.n_obs
        if tr.est is None:
            return False
        ex, ey = tr.est
        dx, dy = ex - world.x, ey - world.y
        d = math.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d > 1e-9 else (1.0, 0.0)
        px, py = -uy, ux
        s = tr.std_max()
        if tr.mode == 2 and s <= cfg.clear_std and tr.n_dir >= 2:
            info = world.clear(ex, ey, c)
            if info.result == "success":
                return True
            q = (world.x + px * min(cfg.fix_step_m, max(d, 40.0)),
                 world.y + py * min(cfg.fix_step_m, max(d, 40.0)))
            if tr.is_measured(q):
                q = (world.x + ux * min(cfg.fix_step_m, max(d, 40.0)),
                     world.y + uy * min(cfg.fix_step_m, max(d, 40.0)))
            world.measure(q[0], q[1], c)
            continue
        # --- (eps, delta) 单点联合选择 (替代下面 bow 与固定步长推进两个手挑决定) ---
        # 同一条规则同时回答"要不要侧移"和"推进多远"; 候选集里含发布版动作点,
        # 因此模型认为发布版更好时行为自动退回发布版。
        joint_here = False
        if cfg.joint_choice:
            if tr.mode == 1 and cfg.joint_mode1:
                joint_here = True
            elif tr.mode == 2 and cfg.joint_mode2:
                joint_here = (not cfg.joint_mode2_bow_only) or (
                    cfg.bow and d <= d_bow and _radial_std(tr, ux, uy) > rad_th)
        if joint_here:
            qj = joint_measure_point(world, c, cfg, tr)
            if qj is not None:
                world.measure(qj[0], qj[1], c)
                continue
        # --- 弦式纵深定向检测: 在路径中点侧移, 不折返 ---
        if cfg.bow and d <= d_bow and _radial_std(tr, ux, uy) > rad_th:
            if bow_override is not None:
                eps_frac, delta = bow_override
                eps_frac = min(max(eps_frac, 0.05), 0.95)
                delta = min(max(delta, lat_min), lat_max)
                eps = eps_frac * d
            else:
                # 发布版路径: 逐字保留原来的运算顺序 (``half = 0.5*d`` 再 ``half*half``)。
                # 实测 ``SIGMA_RAD*half*half`` 与 ``SIGMA_RAD*(0.5*d)**2`` 在双精度下
                # 会差 1 ULP, 而测点位置差 1 ULP 就足以让 ``is_measured`` 的去重翻转,
                # 整局轨迹随之分叉 (100 局里 10 局数字变化) —— 发布版必须逐位可复现。
                eps = 0.5 * d
                if lat_frac > 0.0:
                    delta = lat_frac * d          # 线性: 精度/绕行交换比与 d 无关
                else:
                    delta = SIGMA_RAD * eps * eps / max(sig_tgt, 0.1)
                delta = min(max(delta, lat_min), lat_max)
            q = (world.x + ux * eps + px * delta, world.y + uy * eps + py * delta)
            if tr.is_measured(q):
                q = (world.x + ux * eps - px * delta,
                     world.y + uy * eps - py * delta)
            if tr.is_measured(q):
                q = (world.x + ux * eps, world.y + uy * eps)
            world.measure(q[0], q[1], c)
            continue
        if d > standoff:
            step = min(d - standoff, cfg.leg_step_m)
            q = (world.x + ux * step, world.y + uy * step)
            if tr.is_measured(q):
                q = (world.x + ux * min(step * 2.0, d), world.y + uy * min(step * 2.0, d))
            world.measure(q[0], q[1], c)
            continue
        q = (ex + px * lat, ey + py * lat)
        if tr.is_measured(q):
            q = (ex - px * lat, ey - py * lat)
        if tr.is_measured(q):
            q = (ex, ey)
        world.measure(q[0], q[1], c)
    return world.belief.tracks[c - 1].cleared


def run_sweeper(world: World, cfg: Optional[SweepConfig] = None, trace: bool = False,
                real_left: Optional[Any] = None, reserve_s: float = 20.0,
                knows_total: bool = True) -> SweepResult:
    """主循环。

    ``real_left`` / ``reserve_s``: 部署时传入"现实剩余时间"的取数函数, 用于在现实预算
    将尽时收尾 (题目上限 20 分钟现实时间)。进程内评测不传, 只看虚拟时间。

    ``knows_total=False``: **正式测试时机器狗看不到总源数** (真值屏蔽), 因此不能用
    ``all_cleared`` 作为结束条件, 只能靠"没有可清目标 + 探测已无收益 + 残余期望质量低于
    阈值"收手 —— 平时进程内评测用默认 True。
    """
    cfg = cfg or SweepConfig()
    steps = 0
    probes = 0
    sweeps = 0
    pursues = 0
    mv_probe = 0.0
    blocked: set = set()
    no_gain = 0
    cover_steps = 0
    started = False
    while True:
        if knows_total and world.all_cleared:
            break
        if steps >= cfg.max_actions or world.virtual_t > cfg.max_virtual:
            break
        if real_left is not None and real_left() < reserve_s:
            break
        steps += 1
        # 1) 初始盲扫 (一次)
        if not started:
            started = True
            p = (world.x, world.y)
            n = sweep_here(world, cfg, max_ch=cfg.initial_max_ch, pd_min=0.0) \
                if cfg.initial_all else 0
            if trace:
                print("  initial sweep at (%.0f,%.0f): %d ch -> t=%.0f cleared=%d/%d"
                      % (p[0], p[1], n, world.virtual_t, world.cleared_count, world.n_sources),
                      flush=True)
            continue
        # 2) 就近清除**已定位**的源
        c = pick_target(world, cfg, blocked)
        if c is not None:
            pursues += 1
            t0 = world.virtual_t
            before_cleared = world.cleared_count
            tr0 = world.belief.tracks[c - 1]
            n_dir0, n_obs0 = tr0.n_dir, tr0.n_obs
            cfg_p = ops.PursueConfig()
            cfg_p.clear_std_m = cfg.clear_std
            cfg_p.max_steps = cfg.pursue_max_steps
            if cfg.leg_stop:
                tr = world.belief.tracks[c - 1]
                if tr.est is not None and tr.mode != 2:
                    ls = leg_info_stop(world, tr.est, cfg, c)
                    if ls is not None:
                        probe_at(world, ls[0], cfg, max_ch=ls[1], pd_min=cfg.info_pd_min,
                                 info=True)
            if cfg.fast_pursue:
                pursue_fast(world, c, cfg)
            else:
                ops.pursue(world, c, cfg_p, budget_s=700.0)
            if trace:
                tr = world.belief.tracks[c - 1]
                print("  pursue ch%-2d mode=%d est=%s -> %.0fs cleared=%d/%d"
                      % (c, tr.mode, None if tr.est is None else "(%.0f,%.0f)" % tr.est,
                         world.virtual_t - t0, world.cleared_count, world.n_sources), flush=True)
            if world.virtual_t - t0 < 0.5:
                blocked.add(c)
                continue
            if world.cleared_count == before_cleared:
                # 没清掉: 只有"拿到了新观测/新示向度"才允许下次再选它。
                # 否则就是一条假轨迹 (信念给出的估计点附近根本没有源), 会无限重试 ——
                # 实测 seed 9628 因此跑了 15330 s / 2818 次检测 (清除 5/12)。
                tr1 = world.belief.tracks[c - 1]
                if tr1.n_dir <= n_dir0 and tr1.n_obs <= n_obs0:
                    blocked.add(c)
            # 3) 清除成功后原地补测 (无额外移动)
            if cfg.sweep_on_clear and world.cleared_count > before_cleared:
                m0 = world.measures
                sweep_here(world, cfg)
                sweeps += 1
                if trace and world.measures > m0:
                    print("      sweep after clear: %d ch -> t=%.0f"
                          % (world.measures - m0, world.virtual_t), flush=True)
            continue
        # 4) 没有已定位目标: 去信息量最大的点测一轮
        # 覆盖收尾允许突破"专程探测"的次数上限 (它自带 coverage_max_steps 安全网);
        # 其余情况仍按 probe_max 收手。
        if probes >= cfg.probe_max and not cfg.coverage_finish:
            break
        ip = best_info_point(world, cfg)
        if (ip is None or ip[3] <= 0.0) and cfg.hunt:
            # 收尾: "划算"的信息点已经没有了, 但信念里还剩期望质量。
            # 这时按**集合覆盖**把残余质量扫掉 (只要还有可能有源没找到, 就值得走一趟),
            # 否则漏掉的源会直接压低清除比例 (实测 500 局只有 96.3% 清除)。
            er = sum(t.pi for t in world.belief.tracks if not t.cleared)
            need = max(0, cfg.min_sources - world.cleared_count)
            cp = None
            is_cover = False
            if er > cfg.hunt_mass_min or need > 0:
                cp = cover_point(world, cfg, force=True)
            if cp is None and cfg.coverage_finish and cover_steps < cfg.coverage_max_steps:
                # 信念质量已经吃完, 但还有频道没被"1000 m 覆盖" ⇒ 仍可能有源没被发现。
                # 按覆盖判据补一趟: 只要该频道的残余空洞被这次测点以 1000 m 覆盖,
                # 那里若真有源就必然收到信号 (R>=1000 是题目给定的硬下界)。
                cp = coverage_point(world, cfg)
                if cp is not None:
                    cover_steps += 1
                    is_cover = True
            if cp is not None:
                p, gain = cp
                mv0 = world.moved_m
                t0 = world.virtual_t
                obs0 = sum(t.n_dir for t in world.belief.tracks)
                c0 = world.cleared_count
                if is_cover:
                    # 覆盖收尾必须把**每个未清除频道**在本站测掉, 不能像常规探测那样
                    # 按 pi/pd/标准差过滤 (被过滤掉的频道会留下永久空洞)。
                    chans = [c for c in range(1, N_CHANNELS + 1)
                             if not world.belief.tracks[c - 1].cleared
                             and not world.belief.tracks[c - 1].is_measured(p)]
                else:
                    chans = pick_channels(world, p, cfg,
                                          cfg.hunt_max_ch or cfg.probe_max_ch,
                                          0.0, pi_min=0.0)
                if chans:
                    ops.probe(world, p, channels=chans)
                mv_probe += world.moved_m - mv0
                probes += 1
                blocked.clear()
                if (world.cleared_count == c0
                        and sum(t.n_dir for t in world.belief.tracks) == obs0):
                    no_gain += 1
                else:
                    no_gain = 0
                if no_gain >= cfg.hunt_fail_max and need <= 0 and not cfg.coverage_finish:
                    break
                if trace:
                    print("  HUNT (%.0f,%.0f) gain=%.3f er=%.2f -> %.0fs cleared=%d/%d"
                          % (p[0], p[1], gain, er, world.virtual_t - t0,
                             world.cleared_count, world.n_sources), flush=True)
                if world.virtual_t - t0 < 0.5:
                    break
                continue
            if trace:
                print("  stop: no info point", flush=True)
            break
        if ip is None or ip[3] <= 0.0:
            if trace and ip is not None:
                print("  stop: 最好的信息点净收益 %.0f ≤ 0" % ip[3], flush=True)
            break
        p, val, n_ch, _net = ip
        mv0 = world.moved_m
        t0 = world.virtual_t
        obs0 = sum(t.n_dir for t in world.belief.tracks)
        c0 = world.cleared_count
        probe_at(world, p, cfg, max_ch=n_ch, pd_min=cfg.info_pd_min, info=True)
        mv_probe += world.moved_m - mv0
        probes += 1
        # 专程探测"什么都没测到"连续若干次 ⇒ 信念里剩下的质量已经找不到落点, 收手。
        # 这一步对**正式测试**尤其重要: 那里看不到真值总数 (knows_total=False),
        # 没有这个判据就会一直扫到探测预算用尽, 白烧几百秒。
        if (world.cleared_count == c0
                and sum(t.n_dir for t in world.belief.tracks) == obs0):
            no_gain += 1
        else:
            no_gain = 0
        if no_gain >= cfg.hunt_fail_max:
            if trace:
                print("  stop: %d 次专程探测无收获" % no_gain, flush=True)
            break
        if trace:
            print("  info move (%.0f,%.0f) val=%.0f n=%d -> %.0fs moved %.0f cleared=%d/%d"
                  % (p[0], p[1], val, n_ch, world.virtual_t - t0, world.moved_m - mv0,
                     world.cleared_count, world.n_sources), flush=True)
        blocked.clear()
        if world.virtual_t - t0 < 0.5:
            break
    res = SweepResult(world)
    res.steps = steps
    res.probes = probes
    res.sweeps = sweeps
    res.pursues = pursues
    res.mv_probe = mv_probe
    return res


def run_many(seeds: Sequence[int], cfg: Optional[SweepConfig] = None) -> List[Dict[str, float]]:
    out = []
    for sd in seeds:
        w = World(seed=sd)
        r = run_sweeper(w, cfg)
        d = r.as_dict()
        d["seed"] = sd
        out.append(d)
    return out
