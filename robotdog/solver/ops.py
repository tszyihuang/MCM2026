"""问题3 的动作原语: 逼近清除 (pursue)、探测 (probe)、候选点生成、状态特征。

被三处复用, 保证进程内评测与部署程序面对完全相同的动作原语:
  * solver/sweeper.py    —— 问题3 规划器
  * solver/deploy.py     —— 机器狗程序 (HTTP + 同一套原语)
  * solver/e2e_test.py   —— 端到端验收
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .belief import max_eig, perp_dir
from ..consts import (N_CHANNELS,
                     RADIUS_MAX_M,
                     RADIUS_MIN_M,
                     SPEED_MPS,
                     DETECT_DURATION_S,
                     CHANNEL_SWITCH_S)
from .world import World

Vec = Tuple[float, float]
LOG2PI = math.log(2.0 * math.pi)


def dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(a: Vec) -> Vec:
    n = math.hypot(a[0], a[1])
    return (1.0, 0.0) if n < 1e-9 else (a[0] / n, a[1] / n)


def add(a: Vec, b: Vec, s: float = 1.0) -> Vec:
    return (a[0] + b[0] * s, a[1] + b[1] * s)


# --------------------------------------------------------------------------------------
# 候选探测点
# --------------------------------------------------------------------------------------
LATTICE_SPACING = 800.0


def build_lattice(spacing: float = LATTICE_SPACING, max_radius: float = 2450.0) -> np.ndarray:
    """覆盖圆域的六角格点。

    间距 800 m (覆盖半径 462 m) 看起来"过密", 但实测比 1150 m 更快: 候选点越细,
    贪心/策略越能挑到"顺路且信息量大"的点, 省下的移动远多于多出来的候选评估成本。
    """
    pts: List[Vec] = []
    dy = spacing * math.sqrt(3.0) / 2.0
    j = 0
    y = -max_radius
    while y <= max_radius + 1e-9:
        off = (spacing / 2.0) if (j % 2) else 0.0
        x = -max_radius - spacing
        while x <= max_radius + spacing + 1e-9:
            px, py = x + off, y
            if math.hypot(px, py) <= max_radius:
                pts.append((px, py))
            x += spacing
        y += dy
        j += 1
    pts.sort(key=lambda p: (round(math.hypot(p[0], p[1]), 3), round(math.atan2(p[1], p[0]), 6)))
    return np.asarray(pts, dtype=np.float64)


# --------------------------------------------------------------------------------------
# 探测 (probe)
# --------------------------------------------------------------------------------------
def channels_to_probe(world: World, pi_min: float = 0.005) -> List[int]:
    out: List[int] = []
    for i in range(N_CHANNELS):
        tr = world.belief.tracks[i]
        if tr.cleared or tr.pi < pi_min:
            continue
        out.append(i + 1)
    return out


def order_channels(world: World, chans: Sequence[int]) -> List[int]:
    s = sorted(int(c) for c in chans)
    if not s:
        return []
    if world.channel in s:
        k = s.index(world.channel)
        return s[k:] + s[:k]
    return s


def probe(world: World, point: Vec, channels: Optional[Sequence[int]] = None,
          pi_min: float = 0.005, max_channels: int = 20) -> List[Tuple[int, str]]:
    if channels is None:
        channels = channels_to_probe(world, pi_min=pi_min)
    chans = order_channels(world, list(channels)[:max_channels])
    res: List[Tuple[int, str]] = []
    for c in chans:
        if world.belief.tracks[c - 1].cleared:
            continue
        info = world.measure(point[0], point[1], c)
        res.append((c, info.result))
        if info.result == "near":
            world.clear(point[0], point[1], c)
    return res


def probe_utility(world: World, p: Vec, top: bool = False) -> Tuple[float, int, float, List[int]]:
    """在 p 处做一批检测的期望收益 / 时间成本。

    收益 = Σ_c π_c · P(在 p 处能收到 c) · (该次检测对定位的价值)
        * 未探测频道: 价值 1.0 (首次发现, 直接决定还有几个源没找到)
        * 已有示向度的频道: 价值 = 最大不确定度的相对下降 (共线测量自动得 0)
    成本 = 移动耗时 + 频道数 × (5 s 检测 + 1 s 切换)
    """
    bel = world.belief
    pts = bel.pts
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    det_kernel = np.clip((RADIUS_MAX_M - d) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0)
    gain = 0.0
    n_ch = 0
    chans: List[int] = []
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared or tr.pi < 0.005:
            continue
        n_ch += 1
        chans.append(i + 1)
        pd = tr.detect_prob(p, pts, det_kernel if tr.mode == 0 else None)
        if tr.mode == 0:
            gain += float(tr.pi) * pd * 1.0
        else:
            if tr.cov is None:
                continue
            s0 = max_eig(tr.cov) ** 0.5
            s1 = max_eig(tr.cov_with_new(p)) ** 0.5
            red = max(s0 - s1, 0.0)
            w = 1.2 if tr.mode == 1 else 0.8
            gain += float(tr.pi) * pd * min(red / 150.0, 1.2) * w
    if n_ch == 0:
        return 0.0, 0, 0.0, []
    t_cost = dist((world.x, world.y), p) / SPEED_MPS + n_ch * (DETECT_DURATION_S + CHANNEL_SWITCH_S)
    return gain / max(t_cost, 1.0), n_ch, gain, chans


def path_candidates(world: World, top_k: int = 2) -> List[Vec]:
    """在"通向已定位源"的路上取点: 边走边测, 把探测与清除的路程合并。"""
    out: List[Vec] = []
    rows = []
    for i in range(N_CHANNELS):
        tr = world.belief.tracks[i]
        if tr.cleared or tr.est is None or tr.pi < 0.3 or tr.mode != 2:
            continue
        rows.append((dist((world.x, world.y), tr.est), i))
    rows.sort()
    for d, i in rows[:top_k]:
        tr = world.belief.tracks[i]
        ex, ey = tr.est
        for f in (0.35, 0.70):
            out.append((world.x + (ex - world.x) * f, world.y + (ey - world.y) * f))
    return out


def hole_candidates(world: World, lattice: np.ndarray, top_k: int = 6) -> List[Vec]:
    """"覆盖空洞"候选点: 只按"最坏情况下(半径 1000 m)还没被任何检测点覆盖到的
    信念质量"来挑点。

    动机: 自适应探测会在 pi 降到很低时收手, 但 pi 低只代表"按 R~U(1000,1500) 的
    先验不太可能", 而真实 R 可能正好落在下尾 (实测漏掉的源全部是这种:
    最近检测点距真值 1100 m 而 R=1065)。题目要求"确保全部清除", 因此在收尾阶段
    值得为这最后几个百分点付一次覆盖代价。
    """
    bel = world.belief
    cells = bel.pts
    n = cells.shape[0]
    covered = np.zeros(n, dtype=bool)
    for tr in bel.tracks:
        for (px, py) in tr.meas_pts:
            covered |= (np.hypot(cells[:, 0] - px, cells[:, 1] - py) <= RADIUS_MIN_M)
    unc = ~covered
    if not unc.any():
        return []
    mass = np.zeros(n, dtype=np.float64)
    any_mass = False
    for i in range(N_CHANNELS):
        tr = bel.tracks[i]
        if tr.cleared or tr.grid is None or tr.pi < 1e-4:
            continue
        mass += float(tr.pi) * (tr.grid.astype(np.float64) * unc)
        any_mass = True
    if not any_mass or mass.sum() <= 1e-9:
        return []
    scored = []
    for p in lattice:
        m = (np.hypot(cells[:, 0] - p[0], cells[:, 1] - p[1]) <= RADIUS_MIN_M)
        gain = float(mass[m].sum())
        if gain <= 0:
            continue
        cost = dist((world.x, world.y), (float(p[0]), float(p[1]))) / SPEED_MPS + 6.0 * 20.0
        scored.append((gain / max(cost, 1.0), (float(p[0]), float(p[1]))))
    scored.sort(key=lambda kv: -kv[0])
    return [p for _, p in scored[:top_k]]


def best_probe_points(world: World, lattice: np.ndarray, k: int = 6,
                      extra: Optional[Sequence[Vec]] = None) -> List[Vec]:
    """按 (收益/成本) 排序的前 k 个探测点 (格点 + 信念驱动点 + 路径点)。"""
    cands: List[Vec] = [(float(p[0]), float(p[1])) for p in lattice]
    cands.extend(info_candidates(world, top_k=4))
    cands.extend(path_candidates(world, top_k=2))
    if extra:
        cands.extend(extra)
    scored: List[Tuple[float, Vec]] = []
    seen = set()
    for p in cands:
        key = (round(p[0] / 5.0), round(p[1] / 5.0))
        if key in seen:
            continue
        seen.add(key)
        v, n, _, _ = probe_utility(world, p)
        if n > 0 and v > 0:
            scored.append((v, p))
    scored.sort(key=lambda kv: -kv[0])
    out = [p for _, p in scored[:k]]
    # 收尾阶段: 若仍有"存在概率虽低但未被覆盖"的质量, 追加覆盖空洞点
    if len(out) < k:
        for p in hole_candidates(world, lattice, top_k=k - len(out)):
            out.append(p)
    return out


def info_candidates(world: World, top_k: int = 4) -> List[Vec]:
    """信念驱动的高信息量检测点 (用于给单射线频道补第二条示向度)。"""
    out: List[Vec] = []
    bel = world.belief
    order = sorted(range(N_CHANNELS), key=lambda i: -bel.tracks[i].pi)
    used = 0
    for i in order:
        tr = bel.tracks[i]
        if tr.cleared or tr.pi < 0.05 or tr.est is None or tr.n_dir != 1 or tr.cov is None:
            continue
        # 最大不确定度方向 v ⇒ 有效基线方向 w ⟂ v (新检测点的视线需与 v 垂直)
        u = major_axis(tr.cov)
        w = (-u[1], u[0])
        est = tr.est
        for sgn in (1.0, -1.0):
            p = (est[0] + sgn * w[0] * 900.0, est[1] + sgn * w[1] * 900.0)
            out.append(p)
        used += 1
        if used >= top_k:
            break
    return out


# --------------------------------------------------------------------------------------
# 逼近清除 (pursue)
# --------------------------------------------------------------------------------------
class PursueConfig:
    clear_std_m: float = 24.0        # max-eig 标准差低于该值才直接清除
    min_pi: float = 0.20
    max_steps: int = 16
    max_fail: int = 3
    clear_cost_weight: float = 3.0   # 每次候选打分的 "不确定度→秒" 折算


def major_axis(cov: np.ndarray) -> Vec:
    return perp_dir(cov)


def pursue_candidates(world: World, c: int, cfg: PursueConfig) -> List[Vec]:
    """逼近过程中下一检测点的候选集合。

    关键几何: 一次检测只约束"源在某条射线上"(横向已知, 纵向未知)。
    要压缩最大不确定度方向 v (通常是视线方向), 新检测点的**视线必须与 v 垂直**,
    即新检测点落在 est ± w·D (w ⟂ v) 上; 沿视线前后挪动不增加任何信息。
    """
    tr = world.belief.tracks[c - 1]
    est = tr.est
    pos = (world.x, world.y)
    if est is None or tr.cov is None:
        return [pos]
    d = dist(pos, est)
    v = perp_dir(tr.cov)                  # 最不确定方向
    w = (-v[1], v[0])                     # 有效基线方向
    scale = max(d, 3.0 * max(max_eig(tr.cov) ** 0.5, 60.0))
    cands: List[Vec] = []
    # (a) 沿 pos→est 连线的推进点 (不额外绕路)
    for f in (0.30, 0.60, 0.85):
        cands.append((pos[0] + (est[0] - pos[0]) * f, pos[1] + (est[1] - pos[1]) * f))
    # (b) 以 est 为中心的极坐标候选: 垂直基线方向信息量最大
    for D in (0.35 * scale, 0.7 * scale, 1.1 * scale):
        D = max(220.0, min(D, 1800.0))
        for ang in (0.0, 45.0, -45.0, 90.0, -90.0, 180.0):
            a = math.radians(ang)
            ca, sa = math.cos(a), math.sin(a)
            wx, wy = w[0] * ca - w[1] * sa, w[0] * sa + w[1] * ca
            cands.append((est[0] + wx * D, est[1] + wy * D))
    return cands


def choose_measure_point(world: World, c: int, cfg: PursueConfig) -> Vec:
    """一步前瞻: 最小化 "去检测点的移动 + 之后到新估计点的移动 + 检测 + 不确定度折算"。

    不确定度按**检测概率**加权: 在收不到信号的点做检测不产生任何定位信息。
    这个修正消除了 "原地反复检测同一点" 的退化行为。
    """
    tr = world.belief.tracks[c - 1]
    est = tr.est
    pos = (world.x, world.y)
    if est is None or tr.cov is None:
        return pos
    pts = world.belief.pts
    s0 = max_eig(tr.cov) ** 0.5
    best_p = est
    best_s = float("inf")
    for p in pursue_candidates(world, c, cfg):
        if tr.is_measured(p):
            continue                      # 该点已有读数, 再来一次不产生信息
        pd = tr.detect_prob(p, pts, None)
        s1 = max_eig(tr.cov_with_new(p)) ** 0.5
        s_exp = pd * s1 + (1.0 - pd) * s0
        travel = (dist(pos, p) + dist(p, est)) / SPEED_MPS
        s = travel + DETECT_DURATION_S + cfg.clear_cost_weight * s_exp
        if s < best_s:
            best_s, best_p = s, p
    return best_p


def pursue(world: World, channel: int, cfg: Optional[PursueConfig] = None,
           budget_s: float = 420.0) -> Tuple[bool, float]:
    """逼近并清除频道 channel, 返回 (是否清除成功, 消耗虚拟秒)。"""
    cfg = cfg or PursueConfig()
    t0 = world.virtual_t
    fails = 0
    last_obs = world.belief.tracks[channel - 1].n_obs
    stall = 0
    for _ in range(cfg.max_steps):
        if world.virtual_t - t0 > budget_s:
            break
        tr = world.belief.tracks[channel - 1]
        if tr.cleared:
            return True, world.virtual_t - t0
        if tr.pi < cfg.min_pi or tr.est is None:
            break
        if tr.near_pt is not None:
            world.clear(tr.near_pt[0], tr.near_pt[1], channel)
            continue
        if tr.n_obs <= last_obs:
            stall += 1
            if stall >= 3:
                break
        else:
            stall = 0
        last_obs = tr.n_obs
        s = max_eig(tr.cov) ** 0.5 if tr.cov is not None else 1e9
        if tr.n_dir >= 2 and s <= cfg.clear_std_m:
            info = world.clear(tr.est[0], tr.est[1], channel)
            if info.result == "success":
                return True, world.virtual_t - t0
            fails += 1
            if fails >= cfg.max_fail:
                break
            continue
        p = choose_measure_point(world, channel, cfg)
        pd = tr.detect_prob(p, world.belief.pts, None)
        info = world.measure(p[0], p[1], channel)
        if info.result == "near":
            world.clear(p[0], p[1], channel)
            if world.belief.tracks[channel - 1].cleared:
                return True, world.virtual_t - t0
        elif info.result == "no_signal":
            # 只有"本以为能收到却没收到"才算异常, 否则只是还没走到有效半径内
            if pd > 0.5:
                fails += 1
                if fails >= cfg.max_fail:
                    break
    return world.belief.tracks[channel - 1].cleared, world.virtual_t - t0
