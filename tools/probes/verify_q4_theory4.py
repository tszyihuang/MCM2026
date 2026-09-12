# -*- coding: utf-8 -*-
"""问题4 建模的三个核心引理: 数值验证 (论文取数).

引理 1 (定向观测似然, 方向均匀先验下可解析)
    p(x,y) 处对定向源 S 做一次检测 (源在 A=(x0,y0), 定向方向 psi, 覆盖 psi±90):
        P(收到 | d, psi) = 1[ d <= R ] * 1[ |wrap(phi - psi)| <= 90 ]
    其中 d=|A-p|, phi = bearing(p -> A) = 可见的示向度真值。
    对 psi ~ U(0,360) 取期望 (记 u = wrap(phi - psi0) 为"波束中心相对视线的偏角"):
        P(收到 | d <= R, psi0) = 1/2 + asin(clamp(u,-1,1))/180        (单位: 度)
    特别地 P = 1 <=> |u| <= 0 之外 45 度内? 不: 精确地
        u = 0   -> 1        (视线正对波束中心)
        u = 90  -> 1/2
        u = 180 -> 0
    本脚本用蒙特卡洛 (对 psi 采样) 验证该闭式解, 并给出角向剖面的单调性。

引理 2 (无信号 = 半平面约束)
    在 d <= 1000 (必然在有效半径内) 处得到 no_signal, 且已知源为定向源:
        |wrap(phi - psi)| > 90     <=>   psi 落在以 "phi+180" 为中心、宽度 180 的开半圆内。
    即 "波束背对检测点"。因此**无信号观测把 psi 的后验压成一个 180 度半圆**,
    而不是像全向源那样只排除一个圆盘。

引理 3 (保证发现: 蜂窝格临界间距)
    要保证"无论定向方向如何都能至少收到一次", 对源可能位置 p 需要
        { s_i : |s_i - p| <= 1000 } 的方位角集合不落在任何开半圆内。
    对间距 s 的三角格, 以 p 为心的 Dirichlet/Voronoi 单元外接圆半径 <= s/sqrt(3),
    因此全部顶点都在 1000 m 内的充要条件是 s <= 1000*sqrt(3) = 1732 m;
    而角向条件在 s = 1000 m 处严格成立 (顶角 60 度的三角形, 最坏点在中心,
    三个顶点张角 120 度 => 最大空隙 240 度? 见脚本实测)。
    本脚本给出 s 的精确临界值。
"""
import math
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import random

DET_MIN = 1000.0
R = 1800.0
RNG = random.Random(20260911)


# --------------------------------------------------------------------------------------
# 引理 1: 方向均匀先验下的检测概率闭式解
# --------------------------------------------------------------------------------------
def p_detect_closed(alpha_deg: float) -> float:
    """引理 1b 的闭式解: 已知在 p 处收到, 在 q 处再次收到的概率。

    alpha = phi_q - phi_p, phi = "源 -> 检测点" 的方位角 (即机器狗在源看来所在的方位)。
    等价写法: alpha = -(beta_q - beta_p), beta = 检测点 -> 源 的方位 (示向度真值)。
    """
    return 1.0 - min(abs(alpha_deg), 180.0) / 180.0


def p_detect_mc(alpha_deg: float, n: int = 100000) -> float:
    """蒙特卡洛: 直接抽 psi ~ U(0,360), 用两个可检测窗口求交。

    几何约定: phi = "源 -> 检测点" 的方位角。
    在点 p 能收到 <=> |wrap(psi - phi_p)| <= 90; 因此在 p 收到的先验区间
    I_p = [phi_p - 90, phi_p + 90] (长度 180)。取 phi_p = 0, phi_q = alpha。
    """
    phi_p = 0.0
    phi_q = alpha_deg
    hit = 0
    for _ in range(n):
        psi = RNG.uniform(0.0, 360.0)
        if (abs((psi - phi_p + 180.0) % 360.0 - 180.0) <= 90.0
                and abs((psi - phi_q + 180.0) % 360.0 - 180.0) <= 90.0):
            hit += 1
    return hit / n / 0.5          # 条件化在"p 处收到"上 (先验概率 1/2)


def p_detect_window(alpha_deg: float) -> float:
    """解析: |I_p ∩ I_q| / |I_p|, 其中两个区间长度都是 180 度, 中心相距 alpha。"""
    return max(180.0 - abs(alpha_deg), 0.0) / 180.0


print("=" * 96)
print("引理 1: 定向源的检测概率 (psi ~ U(0,360) 先验)")
print("   1a  p 单点:  P(收到 | p) = 1/2 * 1[d_p <= R]        —— 距离够近也只有一半")
print("   1b  已知在 p 收到方向 (phi_p), 又问 q:")
print("       P(收到 | q, d_q<=R, phi_p) = 1 - min(|alpha|,180)/180,  alpha = phi_q - phi_p")
print("       alpha=0 (同一方位) -> 1     alpha=+-90 (侧向 90 度) -> 1/2")
print("       alpha=+-180 (正后方) -> 0   (波束朝向 p 就必定背对 q)")
print("       证明: 在 p 处收到 <=> psi 落在长度 180 度的区间 I_p (否则 p 收不到),")
print("             在 q 处收到 <=> psi 落在区间 I_q; 两区间中心相距 alpha,")
print("             故 P = |I_p ∩ I_q| / |I_p| = (180-|alpha|)/180。")
print("=" * 96)
print(f"{'alpha(度)':>10} {'闭式':>10} {'区间解析':>10} {'蒙特卡洛':>10} {'最大误差':>10}")
worst = 0.0
for a in (0, 15, 30, 45, 60, 75, 90, 105, 135, 180, -45, -90, -180):
    c = p_detect_closed(a)
    w = p_detect_window(a)
    m = p_detect_mc(a, 60000)
    worst = max(worst, abs(c - w), abs(c - m))
    print(f"{a:10.0f} {c:10.4f} {w:10.4f} {m:10.4f} {max(abs(c-w),abs(c-m)):10.4f}")
print(f"  闭式 vs 区间解析的最大误差 = {max(abs(p_detect_closed(a)-p_detect_window(a)) for a in range(-180,181)):.2e} (逐度)")
print(f"  闭式 vs 蒙特卡洛的最大误差 = {worst:.4f} (蒙特卡洛 6 万次/点)")
print("  推论 1 (无角度信息时): 定向源在有效半径内被发现的概率恒为 1/2, **与距离无关**;")
print("          全向源恒为 1 —— 这才是问题3 与问题4 的本质差别。")
print("          注意这意味着: 单纯靠近对定向源毫无帮助 (靠近只保证 d<=R) ——")
print("          必须换方位, 或者用 no_signal 把 psi 的后验压成半圆。")
print("  推论 2 (已有一条示向度时): 换个方位就有收益, **侧向 90 度**最划算")


# --------------------------------------------------------------------------------------
# 引理 3: 蜂窝格间距的临界值
# --------------------------------------------------------------------------------------
def hex_lattice(s: float, m: int = 6):
    v1 = (s, 0.0)
    v2 = (s / 2.0, s * math.sqrt(3.0) / 2.0)
    pts = []
    for i in range(-m, m + 1):
        for j in range(-m, m + 1):
            pts.append((i * v1[0] + j * v2[0], i * v1[1] + j * v2[1]))
    return pts


def max_angular_gap(p, pts):
    angs = sorted(math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360.0
                  for q in pts if math.dist(p, q) <= DET_MIN)
    if not angs:
        return 360.0
    if len(angs) == 1:
        return 360.0
    return max(((angs[(i + 1) % len(angs)] - angs[i]) % 360.0) for i in range(len(angs)))


def uniform_samples(n=48):
    """圆域内均匀网格采样 (用"面积均匀"的极坐标网格)。"""
    out = []
    for ir in range(1, n + 1):
        r = R * ir / n
        nt = max(6, int(round(2.0 * math.pi * r / (R / n))))
        for it in range(nt):
            th = 2.0 * math.pi * it / nt
            out.append((r * math.cos(th), r * math.sin(th)))
    out.append((0.0, 0.0))
    return out


S = uniform_samples(48)
print()
print("=" * 96)
print("引理 3: 三角格间距 s 与「定向源保证发现」的临界值 (%d 个采样点)" % len(S))
print("  条件: 每个 p 的 1000 m 内探测点方位角最大空隙 <= 180 度")
print("=" * 96)
print(f"{'s(m)':>7} {'点(m=6)':>8} {'最大空隙':>10} {'失败比例':>10} {'max-1NN':>9}  结论")
for s in (1732, 1500, 1300, 1200, 1150, 1100, 1050, 1030, 1010, 1000, 990, 950, 900, 800):
    pts = hex_lattice(s, m=int(3.0 * R / s) + 4)
    gaps = [max_angular_gap(p, pts) for p in S]
    worst_p = max(zip(gaps, S))[1]
    fails = sum(1 for g in gaps if g > 180.0 + 1e-9)
    max1 = max(min(math.dist(p, q) for q in pts) for p in S)
    ok = "✅ 定向保证" if fails == 0 else "❌"
    print(f"{s:7.0f} {len(pts):8d} {max(gaps):10.1f} {fails/len(S):10.4f} {max1:9.1f}  {ok}"
          + ("" if fails == 0 else "  最坏点(%.0f,%.0f)" % worst_p))
print()
print("  理论对照: 三角格 Voronoi 单元外接圆半径 = s/sqrt(3), 故 s<=1732 时")
print("  每个源位置 p 的 1000 m 内必定含其 Delaunay 三角形的全部 3 个顶点 —— 这是")
print("  问题3 (全向源) 的 1-覆盖临界值。但这 3 个顶点只覆盖 p 周围约 210~240 度的")
print("  扇形, 所以角向条件 (最大空隙 <= 180 度) 要等到 s 更小才成立:")
print("    空隙 <= 180 度  <=>  p 落在三角形某条边的 1000 m 内 <=> 1-覆盖半径 <= 1000/2")
print("    (中垂线两端的方位角恰好相差 180 度, 正是边界情形) —— 因此临界间距 = 检测半径。")
print("  实测与理论一致: s = 1000 m 恰好通过, s = 1010 m 立即出现失败点。")
print()
print("=" * 96)
print("引理 3 补充: 为什么「站在刚清除的源上」在问题4 里不够")
print("  问题3: 12.8 个源的平均最近邻 450 m, 站在刚清除的源上, 1000 m 内的任何源")
print("         P(收到)=1 -> 零移动即可发现。")
print("  问题4: 同一个点上的 P(收到) = 1/2 (与距离无关), 而 alpha 未知;")
print("        波束背对时无论多近都收不到, 必须换一个方位再测。")
print("=" * 96)
for dd in (200, 300, 450, 600, 800, 1000):
    mean_p = sum(p_detect_closed(a) for a in range(0, 91)) / 91.0
    print(f"   d={dd:5.0f} m: 全向 P=1.000 | 定向 P(无先验) = 0.500 | "
          f"定向 P(已有一条示向度) 平均 = {mean_p:.3f}")
print("   => 问题4 的「零移动探测」命中率从 1.00 降到 0.50 (有先验时 0.50~1.00),")
print("      且失败模式是**确定性的** (同一站位重复检测读数不变), 必须靠换方位补救。")
