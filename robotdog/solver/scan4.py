# -*- coding: utf-8 -*-
"""问题4 的补扫点集几何 (纯函数, 不依赖任何信念/策略)。

补扫点集要回答的唯一问题:
**要在多少米的路程内, 让至少 x% 的源被看见一次?**

一个源 S=(P, R, 型, ψ) 在点 Q 可见 ⟺ ``|Q−P| ≤ R`` 且 (全向 或 Q 在 ψ±90° 扇区内)。
扇区是半平面, 所以**定向源只有一半的观察位置能看见它** —— 这正是问题4 比问题3
难的全部原因, 也是本模块所有点集的构造依据。
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec = Tuple[float, float]

ARENA_R = 1800.0


# ------------------------------------------------------------------ 点集构造

def hex_points(spacing: float, rmax: float = ARENA_R, phase: float = 0.0) -> List[Vec]:
    """六边形 (三角) 网格, 间距 ``spacing``, 只保留 ``|P| ≤ rmax`` 的点。

    六边形网格是"给定间距下覆盖平面最省点"的布点方式 (每个点负责 √3/2·s² 面积)。
    """
    pts: List[Vec] = []
    dy = spacing * math.sqrt(3.0) / 2.0
    j = 0
    y = -rmax
    while y <= rmax + 1e-9:
        offs = (spacing / 2.0 if j % 2 else 0.0) + phase
        x = -rmax + offs
        while x <= rmax + 1e-9:
            if math.hypot(x, y) <= rmax:
                pts.append((x, y))
            x += spacing
        y += dy
        j += 1
    pts.sort(key=lambda p: (round(math.hypot(p[0], p[1]), 6), math.atan2(p[1], p[0])))
    return pts


def square_points(spacing: float, rmax: float = ARENA_R) -> List[Vec]:
    pts: List[Vec] = []
    n = int(rmax // spacing) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = i * spacing, j * spacing
            if math.hypot(x, y) <= rmax:
                pts.append((x, y))
    pts.sort(key=lambda p: (round(math.hypot(p[0], p[1]), 6), math.atan2(p[1], p[0])))
    return pts


def ring_points(radius: float, n: int, phase: float = 0.0,
                rmax: float = 0.0) -> List[Vec]:
    """半径 ``radius`` 上均匀的 ``n`` 个点 (每 360/n 度一个)。

    ``rmax > 0`` 时只保留 ``|P| ≤ rmax`` 的点。**默认不过滤** —— 补扫点允许落在
    竞技场外 (机器狗可以在区域外行走), 这一点对问题4 很重要: 贴边定向源的可检测
    弧带常常压在边界上, 场内补扫点可能一个都落不进去 (见 :func:`audit`)。
    """
    n = int(n)
    if n <= 0:
        return []
    pts = [(radius * math.cos(phase + 2.0 * math.pi * i / n),
            radius * math.sin(phase + 2.0 * math.pi * i / n)) for i in range(n)]
    if rmax > 0:
        pts = [p for p in pts if math.hypot(p[0], p[1]) <= rmax]
    return pts


def boundary_ring(radius: float, n: int, phase: float = 0.0) -> List[Vec]:
    """贴边环: 半径接近竞技场边界的补扫点 (专治"贴边定向源的可检测弧带")。"""
    return ring_points(radius, n, phase)


def hex_ring(r_in: float, n_rings: int, spacing: float,
             phase: float = 0.0) -> List[Vec]:
    """同心环布点: 中心 1 点 + 第 k 环半径 ``k·r_in``、点数 ``6k`` (间距≈``r_in``)。

    比矩形/六边形网格更"像巡游", 但覆盖效率略低; 保留作对照。
    """
    pts: List[Vec] = [(0.0, 0.0)]
    for k in range(1, int(n_rings) + 1):
        r = k * r_in
        if r > ARENA_R:
            break
        pts += ring_points(r, 6 * k, phase)
    return pts


# ------------------------------------------------------------------ 巡游长度

def tour_length(points: Sequence[Vec], start: Vec = (0.0, 0.0)) -> float:
    """开放巡游 (从 ``start`` 出发, 不返回) 的近似最优长度。

    点数 ≤ 13 时用 Held-Karp 精确解; 更多时用最近邻 + 2-opt + Or-opt。
    本函数只用于**离线比较点集**, 在线路线由规划器自己决定。
    """
    n = len(points)
    if n == 0:
        return 0.0
    if n == 1:
        return math.dist(start, points[0])
    if n <= 13:
        return _held_karp(points, start)
    return _heuristic(points, start)


def _held_karp(points: Sequence[Vec], start: Vec) -> float:
    n = len(points)
    dist0 = [math.dist(start, points[i]) for i in range(n)]
    D = [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]
    INF = float("inf")
    full = 1 << n
    dp = [[INF] * n for _ in range(full)]
    for i in range(n):
        dp[1 << i][i] = dist0[i]
    for mask in range(full):
        row = dp[mask]
        for i in range(n):
            cur = row[i]
            if cur == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                nm = mask | (1 << j)
                v = cur + D[i][j]
                if v < dp[nm][j]:
                    dp[nm][j] = v
    return min(dp[full - 1])


def _heuristic(points: Sequence[Vec], start: Vec) -> float:
    n = len(points)
    unvisited = set(range(n))
    cur = start
    order: List[int] = []
    while unvisited:
        j = min(unvisited, key=lambda k: math.dist(cur, points[k]))
        unvisited.discard(j)
        order.append(j)
        cur = points[j]
    improved = True
    while improved:
        improved = False
        for a in range(len(order) - 1):
            for b in range(a + 1, len(order)):
                i, j = order[a], order[b]
                pa = points[order[a - 1]] if a else start
                pb = points[order[b + 1]] if b + 1 < len(order) else None
                old = math.dist(pa, points[i]) + (math.dist(points[j], pb) if pb else 0.0)
                new = math.dist(pa, points[j]) + (math.dist(points[i], pb) if pb else 0.0)
                if new < old - 1e-9:
                    order[a:b + 1] = order[a:b + 1][::-1]
                    improved = True
    # Or-opt: 单点插入
    improved = True
    while improved:
        improved = False
        for a in range(len(order)):
            rest = order[:a] + order[a + 1:]
            for b in range(len(rest) + 1):
                cand = rest[:b] + [order[a]] + rest[b:]
                if _path_len(cand, points, start) < _path_len(order, points, start) - 1e-9:
                    order = cand
                    improved = True
                    break
            if improved:
                break
    return _path_len(order, points, start)


def _path_len(order: Sequence[int], points: Sequence[Vec], start: Vec) -> float:
    d = 0.0
    cur = start
    for i in order:
        d += math.dist(cur, points[i])
        cur = points[i]
    return d


# ------------------------------------------------------------------ 覆盖审计

def visible(q: Vec, sx: float, sy: float, R: float,
            is_dir: bool, psi: float) -> bool:
    """覆盖判据 (与 ``simulator.core.InterferenceSource`` 一致)。"""
    dx, dy = sx - q[0], sy - q[1]
    if dx * dx + dy * dy > R * R:
        return False
    if not is_dir:
        return True
    bearing = math.degrees(math.atan2(q[1] - sy, q[0] - sx)) % 360.0
    return abs(((bearing - psi + 180.0) % 360.0) - 180.0) <= 90.0 + 1e-9


def audit(points: Sequence[Vec], sources, chunk: int = 4000) -> float:
    """``sources`` 为 ``(x, y, R, is_dir, psi)`` 序列时, 返回被看见的比例。"""
    hit = 0
    total = len(sources)
    for s in sources:
        for q in points:
            if visible(q, *s):
                hit += 1
                break
    return hit / max(total, 1)
