"""几何与拟合工具 (纯标准库).

方位角、坐标偏移、两直线交会、加权角度残差最小二乘 —— 机器狗程序与工具链共用。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def clamp_coord(value: float) -> float:
    limit = 2_000_000.0
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, float(value)))


# --------------------------------------------------------------------------------------
# 几何工具
# --------------------------------------------------------------------------------------


def bearing_to(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 360.0


def offset(p: Tuple[float, float], bearing_deg: float, distance: float) -> Tuple[float, float]:
    rad = math.radians(bearing_deg)
    return (p[0] + distance * math.cos(rad), p[1] + distance * math.sin(rad))


def line_intersection(
    p1: Tuple[float, float], b1_deg: float, p2: Tuple[float, float], b2_deg: float
) -> Optional[Tuple[float, float]]:
    a1, a2 = math.radians(b1_deg), math.radians(b2_deg)
    d1 = (math.cos(a1), math.sin(a1))
    d2 = (math.cos(a2), math.sin(a2))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / den
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def angle_diff(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def weighted_residual_fast(point: Tuple[float, float], obs) -> float:
    """加权残差的热路径实现。

    与原实现数值完全一致 (同样的运算次序、同样的 ``bearing_to``/``angle_diff``
    语义), 只做两件"不改数学"的事:

    * ``obs`` 用 ``((x, y, b), ...)`` 扁平元组, 避免每轮嵌套解包;
    * ``length``/``atan2``/``degrees``/``fmod``/``fabs`` 绑成局部名, 省掉
      全局查找 —— 这个函数单局会被调用几十万次, 是整个链路最大的 CPU 热点。

    注: 质心拟合必须保持"角度残差 + 距离加权"的口径, 不能换成向量残差,
    否则估计点会变, 属于改变行为而非优化。
    """
    atan2 = math.atan2
    degrees = math.degrees
    hypot = math.hypot
    fabs = math.fabs
    px, py = point
    total = 0.0
    for x, y, b in obs:
        dx = x - px
        dy = y - py
        # bearing_to((x, y), point) = degrees(atan2(py - y, px - x)) % 360
        # 注意必须用 Python 的 % (负数结果非负), 不能用 math.fmod —— 两者在
        # 负数上语义不同, 会让角度残差错位。
        deg = degrees(atan2(-dy, -dx)) % 360.0
        # angle_diff(a, b) = |((a - b + 180) % 360) - 180|
        delta = (deg - b + 180.0) % 360.0 - 180.0
        # dist((x, y), point) = hypot(px - x, py - y), 下限 1.0
        d = hypot(dx, dy)
        if d < 1.0:
            d = 1.0
        q = fabs(delta) / d
        # 必须写成 q ** 2 而不是 q * q: 两者数学等价但舍入不同, 用 q * q 会在
        # 极少数点上产生 1 ulp 偏差, 从而可能改变拟合的取舍。
        total += q ** 2
    return total
