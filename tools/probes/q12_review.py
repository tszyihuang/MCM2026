"""独立复核 docs/问题一与问题二论文.md 的几何模型。

不依赖该文档引用的 q12_optimal.py（该文件不在本仓库中），
从题目附录2 的交会定位定义重新实现，用于交叉验证论文中的数字。

用法:
    python tools/probes/q12_review.py
"""
from __future__ import annotations

import math

import numpy as np

EPS_DEG = 1.0          # 示向度误差 ±1°（题目附录2(1)）
R_MIN_DEFAULT = 1500.0  # 源距上界（有效接收半径 ≤1500）
R_EFF_CONSERVATIVE = 1000.0  # 保守有效接收半径


def _u(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def wedge_halfplanes(s: np.ndarray, theta_deg: float, eps: float = EPS_DEG):
    """楔形 {P: ∠(P-S, θ) ≤ eps} 的两条半平面，约定 cross(n, P-p) ≥ 0 为可行侧。"""
    return [(s, _u(theta_deg - eps)), (s, -_u(theta_deg + eps))]


def clip(poly: list[np.ndarray], p: np.ndarray, n: np.ndarray) -> list[np.ndarray]:
    """Sutherland-Hodgman: 保留 cross(n, P-p) ≥ 0 的一侧。"""
    if not poly:
        return []
    out: list[np.ndarray] = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        da = float(np.cross(n, a - p))
        db = float(np.cross(n, b - p))
        if da >= 0.0:
            out.append(a)
        if (da > 0.0 > db) or (da < 0.0 < db):
            out.append(a + (da / (da - db)) * (b - a))
    return out


BOX = 4.0e5


def region(points, thetas, box: float = BOX):
    """交会定位区域 = 各检测点楔形之交（凸多边形顶点表）。"""
    poly = [np.array([-box, -box]), np.array([box, -box]),
            np.array([box, box]), np.array([-box, box])]
    for s, th in zip(points, thetas):
        for p, n in wedge_halfplanes(np.asarray(s, float), th):
            poly = clip(poly, p, n)
        if not poly:
            return []
    return poly


def diameter(poly) -> float:
    m = len(poly)
    return max(math.dist(poly[i], poly[j]) for i in range(m) for j in range(i + 1, m))


def diameter_pair(poly):
    m = len(poly)
    best = (-1.0, 0, 1)
    for i in range(m):
        for j in range(i + 1, m):
            d = math.dist(poly[i], poly[j])
            if d > best[0]:
                best = (d, i, j)
    return best


def min_enclosing_circle(poly):
    """暴力最小包围圆（顶点数很少，够用）。"""
    pts = [np.asarray(p, float) for p in poly]
    best = None
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            c = (a + b) / 2.0
            r = float(np.linalg.norm(a - c))
            if all(np.linalg.norm(p - c) <= r + 1e-9 for p in pts):
                if best is None or r < best[1]:
                    best = (c, r)
    for i, a in enumerate(pts):
        for j, b in enumerate(pts[i + 1:], i + 1):
            for c in pts[j + 1:]:
                ax, ay = a
                bx, by = b
                cx, cy = c
                d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
                      + (cx**2 + cy**2) * (ay - by)) / d
                uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
                      + (cx**2 + cy**2) * (bx - ax)) / d
                ctr = np.array([ux, uy])
                r = float(np.linalg.norm(a - ctr))
                if all(np.linalg.norm(p - ctr) <= r + 1e-9 for p in pts):
                    if best is None or r < best[1]:
                        best = (ctr, r)
    return best


def covered_by_diameter_circle(poly) -> tuple[bool, float, float, float]:
    """返回 (是否被直径圆覆盖, d, r*, eta=2r*/d)。"""
    d, i, j = diameter_pair(poly)
    m = (poly[i] + poly[j]) / 2.0
    thales = max(float(np.dot(np.asarray(p) - poly[i], np.asarray(p) - poly[j])) for p in poly)
    c, r = min_enclosing_circle(poly)
    eta = 2.0 * r / d
    return (thales <= 1e-6, d, r, eta)


def q2_geometry(L: float, beta_deg: float, r: float, eps: float = EPS_DEG):
    """S1=(0,0)、θ1=0；S2=L·u(β)；源在 (r,0)（落在 θ1 测量线上）。"""
    s1 = np.array([0.0, 0.0])
    s2 = L * _u(beta_deg)
    g = np.array([r, 0.0])
    theta2 = math.degrees(math.atan2(g[1] - s2[1], g[0] - s2[0]))
    poly = region([s1, s2], [0.0, theta2])
    return poly, s2, theta2


def q2_alpha(L: float, beta_deg: float, r: float) -> float:
    b = math.radians(beta_deg)
    rho2 = math.hypot(r - L * math.cos(b), -L * math.sin(b))
    if rho2 < 1e-12:
        return 0.0
    return math.degrees(math.asin(min(1.0, L * abs(math.sin(b)) / rho2)))


# --------------------------------------------------------------------------
# 实验一：问题二模型里 D(r) 的形状（最坏源距在哪）
# --------------------------------------------------------------------------
def exp1_dr_curve(L=1004.10, beta=34.78, r_lo=5.0, r_max=1500.0):
    print(f"[E1] D(r) 曲线  L={L}  β={beta}°  r∈[{r_lo},{r_max}]")
    print(f"{'r':>8} {'D(r)':>10} {'α(°)':>8} {'ρ2':>9}")
    best = (0.0, 0.0)
    for r in [r_lo, 100, 300, 500, 700, 825, 900, 1000, 1100, 1200, 1350, 1500]:
        poly, _, _ = q2_geometry(L, beta, r)
        d = diameter(poly) if poly else float("nan")
        rho2 = math.hypot(r - L * math.cos(math.radians(beta)), -L * math.sin(math.radians(beta)))
        print(f"{r:8.1f} {d:10.3f} {q2_alpha(L, beta, r):8.3f} {rho2:9.1f}")
    # 细扫找最坏点
    for k in range(3001):
        r = r_lo + (r_max - r_lo) * k / 3000.0
        poly, _, _ = q2_geometry(L, beta, r)
        d = diameter(poly) if poly else float("nan")
        if d > best[0]:
            best = (d, r)
    print(f"     细扫最坏: D={best[0]:.4f} @ r={best[1]:.1f}   "
          f"(u=L·cosβ={L * math.cos(math.radians(beta)):.1f}, "
          f"D(r_lo)={diameter(q2_geometry(L, beta, r_lo)[0]):.4f}, "
          f"D(r_max)={diameter(q2_geometry(L, beta, r_max)[0]):.4f})")
    return best


def _feasible(L, beta, r_lo, r_max=1500.0, r_eff=R_EFF_CONSERVATIVE,
              alpha_lo=25.0, alpha_hi=155.0):
    """按题目口径的可行域：源必须落在两台测向机的有效接收半径内（保守 R_eff=1000），
    且交会角 α 落在 [25°,155°] 内。返回 (是否可行, 违背原因)。"""
    for r in (r_lo, r_max):
        rho2 = math.hypot(r - L * math.cos(math.radians(beta)), -L * math.sin(math.radians(beta)))
        if rho2 > r_eff + 1e-9:
            return False, f"ρ2({r:.0f})={rho2:.0f}>{r_eff:.0f}"
    for r in (r_lo, r_max, L * math.cos(math.radians(beta))):
        a = q2_alpha(L, beta, r)
        if not (alpha_lo - 1e-9 <= a <= alpha_hi + 1e-9):
            return False, f"α({r:.0f})={a:.1f}°"
    return True, ""


def exp2_minimax(r_lo, r_max=1500.0, L_grid=(800.0, 1120.0, 5.0), beta_grid=(5.0, 85.0, 1.0),
                 n_r=61):
    """网格极小极大：min_{L,β} max_{r} D(r;L,β)。"""
    rs = np.linspace(r_lo, r_max, n_r)
    best = (float("inf"), None)
    L0, L1, dL = L_grid
    b0, b1, db = beta_grid
    Ls = np.arange(L0, L1 + 1e-9, dL)
    bs = np.arange(b0, b1 + 1e-9, db)
    for L in Ls:
        for beta in bs:
            ok, _ = _feasible(float(L), float(beta), r_lo, r_max)
            if not ok:
                continue
            worst = 0.0
            for r in rs:
                poly, _, _ = q2_geometry(float(L), float(beta), float(r))
                if not poly:
                    worst = float("inf")
                    break
                worst = max(worst, diameter(poly))
            if worst < best[0]:
                best = (worst, (float(L), float(beta)))
    return best


# --------------------------------------------------------------------------
# 实验二：问题一覆盖率的定性结论
# --------------------------------------------------------------------------
def _coverage_by_errors(stations, src, errs):
    """stations: 检测点坐标表；errs: 各点示向度误差(度)，加在真实方位角上。"""
    thetas = []
    for s, e in zip(stations, errs):
        true = math.degrees(math.atan2(src[1] - s[1], src[0] - s[0]))
        thetas.append(true + e)
    poly = region(stations, thetas)
    if not poly:
        return None
    ok, d, r, eta = covered_by_diameter_circle(poly)
    return ok, d, r, eta


def exp3_n2_coverage(trials=4000, seed=0):
    """n=2：两站点放在源附近的不同位形下，随机 ±1° 误差的覆盖率。"""
    rng = np.random.default_rng(seed)
    src = np.array([0.0, 0.0])
    hit = 0
    etas = []
    tot = 0
    for _ in range(trials):
        rho1 = rng.uniform(300, 1500)
        rho2 = rng.uniform(300, 1500)
        az1 = rng.uniform(0, 360)
        # 两站点张角随机
        span = rng.uniform(20, 180)
        az2 = az1 + span * rng.choice([-1, 1])
        s1 = rho1 * _u(az1)
        s2 = rho2 * _u(az2)
        errs = rng.uniform(-1, 1, size=2)
        res = _coverage_by_errors([s1, s2], src, errs)
        if res is None:
            continue
        tot += 1
        hit += int(res[0])
        etas.append(res[3])
    return hit / max(1, tot), tot, float(np.mean(etas)), float(np.max(etas))


# --------------------------------------------------------------------------
# 实验三：真源位置 → (误差 e1,e2) 与源距 r 的完整参数化
# --------------------------------------------------------------------------
def q2_region_general(L: float, beta_deg: float, r: float, e1: float, e2: float):
    """S1 在原点、θ1 实测 = 0（旋转规范）；S2 = L·u(β)。
    真源 G：从 S1 看真实方位 θ1_true = -e1，源距 r；
    从 S2 看真实方位为 ψ - e2，其中 ψ 是 S2 处实测示向度。"""
    s1 = np.array([0.0, 0.0])
    s2 = L * _u(beta_deg)
    g = s1 + r * _u(-e1)
    theta2_true = math.degrees(math.atan2(g[1] - s2[1], g[0] - s2[0]))
    psi = theta2_true + e2
    poly = region([s1, s2], [0.0, psi])
    return poly, s2, psi, theta2_true


def exp4_error_sensitivity(L=1004.10, beta=34.78, r_lo=5.0, r_max=1500.0, step=0.25, n_r=121):
    """在论文最优 S2 上，把 ±1° 测量误差也当作不确定量：max_{r,e1,e2} D。"""
    rs = np.linspace(r_lo, r_max, n_r)
    es = np.arange(-1.0, 1.0 + 1e-9, step)
    worst = (0.0, None)
    worst_zero = (0.0, None)
    vals = []
    for e1 in es:
        for e2 in es:
            w = 0.0
            arg = None
            for r in rs:
                poly, _, _, _ = q2_region_general(L, beta, float(r), float(e1), float(e2))
                if not poly:
                    continue
                d = diameter(poly)
                if d > w:
                    w, arg = d, float(r)
            vals.append(w)
            if w > worst[0]:
                worst = (w, (float(e1), float(e2), arg))
            if e1 == 0.0 and e2 == 0.0:
                worst_zero = (w, (float(e1), float(e2), arg))
    return worst, worst_zero, float(np.mean(vals)), float(np.min(vals))


def boundary_L(beta_deg: float, r_lo: float, r_eff: float = R_EFF_CONSERVATIVE) -> float:
    """约束 ρ2(r_lo) ≤ R_eff 取等时的 L（该约束把 L 从上方卡住）。"""
    b = math.radians(beta_deg)
    return r_lo * math.cos(b) + math.sqrt(max(0.0, r_eff**2 - (r_lo * math.sin(b)) ** 2))


def exp5_beta_profile(r_lo, r_max=1500.0, errs=((0.0, 0.0),), n_r=61, b_lo=5.0, b_hi=85.0, db=1.0):
    """沿 ρ2(r_lo)=1000 边界，做 β 一维极小极大。返回 [(β, L, D_worst, 最坏 r, 最坏 err), ...]"""
    out = []
    rs = np.linspace(r_lo, r_max, n_r)
    for beta in np.arange(b_lo, b_hi + 1e-9, db):
        L = boundary_L(float(beta), r_lo)
        best = (0.0, None)
        for e1, e2 in errs:
            for r in rs:
                poly, _, _, _ = q2_region_general(L, float(beta), float(r), e1, e2)
                if not poly:
                    continue
                d = diameter(poly)
                if d > best[0]:
                    best = (d, (float(r), e1, e2))
        out.append((float(beta), L, best[0], best[1]))
    return out


def polygon_area(poly) -> float:
    a = 0.0
    m = len(poly)
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def exp6_worst_metrics(L, beta, r_lo, r_max=1500.0, n_r=1501):
    """给定 S2，返回 max_r D(r) 与 max_r Area(r) 及其最坏 r。"""
    wd = (0.0, None)
    wa = (0.0, None)
    for k in range(n_r):
        r = r_lo + (r_max - r_lo) * k / (n_r - 1)
        poly, _, _ = q2_geometry(L, beta, r)
        if not poly:
            continue
        d = diameter(poly)
        a = polygon_area(poly)
        if d > wd[0]:
            wd = (d, r)
        if a > wa[0]:
            wa = (a, r)
    return wd, wa


def exp7_L_monotone(beta, Ls, r_lo=5.0, r_max=1500.0, n_r=61):
    out = []
    for L in Ls:
        ok, why = _feasible(L, beta, r_lo, r_max)
        wd = None
        if ok:
            wd = 0.0
            for k in range(n_r):
                r = r_lo + (r_max - r_lo) * k / (n_r - 1)
                poly, _, _ = q2_geometry(L, beta, r)
                wd = max(wd, diameter(poly))
        out.append((L, ok, why, wd))
    return out


def exp8_q1_coverage(n_stations=(3, 6), trials=4000, seed=1, layout="uniform",
                     span_range=(100.0, 360.0), rho_range=(300.0, 1500.0)):
    """问题一：随机布站 + ±1° 误差下的直径圆覆盖率。
    误差加在真实方位角上，故定位区域必非空（真源在其中）。"""
    rng = np.random.default_rng(seed)
    src = np.array([0.0, 0.0])
    hit = tot = unbounded = 0
    etas = []
    for _ in range(trials):
        n = int(rng.integers(n_stations[0], n_stations[1] + 1))
        if layout == "arc":
            span = rng.uniform(*span_range)
            start = rng.uniform(0, 360)
            az = start + np.sort(rng.uniform(0, span, size=n)) * rng.choice([-1, 1])
        else:
            az = rng.uniform(0, 360, size=n)
        rho = rng.uniform(*rho_range, size=n)
        pts = [rho[i] * _u(float(az[i])) for i in range(n)]
        errs = rng.uniform(-1, 1, size=n)
        thetas = []
        for s, e in zip(pts, errs):
            true = math.degrees(math.atan2(-s[1], -s[0]))
            thetas.append(true + e)
        poly = region(pts, thetas)
        if not poly:
            continue
        d, i, j = diameter_pair(poly)
        if d > 1e4:                      # 触到裁剪框 = 区域无界
            unbounded += 1
            continue
        tot += 1
        ok, _, _, eta = covered_by_diameter_circle(poly)
        hit += int(ok)
        etas.append(eta)
    return hit / max(1, tot), tot, unbounded, float(np.mean(etas)), float(np.max(etas))


def exp9_jung(trials=20000, seed=7):
    """随机凸多边形的 2r*/d 是否落在 [1, 2/√3]，等边三角形是否取到上界。"""
    rng = np.random.default_rng(seed)
    lo, hi = float("inf"), 0.0
    for _ in range(trials):
        m = int(rng.integers(3, 9))
        pts = rng.normal(size=(m, 2))
        # 取凸包顶点（Monotone chain）
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        p = pts[order]
        def half(ps):
            st = []
            for q in ps:
                while len(st) >= 2 and np.cross(st[-1] - st[-2], q - st[-2]) <= 0:
                    st.pop()
                st.append(q)
            return st
        hull = np.array(half(p)[:-1] + half(p[::-1])[:-1])
        if len(hull) < 3:
            continue
        poly = [np.asarray(v) for v in hull]
        eta = covered_by_diameter_circle(poly)[3]
        lo, hi = min(lo, eta), max(hi, eta)
    tri = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3) / 2])]
    _, dt, rt, etat = covered_by_diameter_circle(tri)
    return lo, hi, 2.0 / math.sqrt(3.0), etat


# --------------------------------------------------------------------------
# 论文 §1.3 覆盖率表的规范采样器（文档里必须写明的那几条口径）
#   源：位于目标圆域中心（Q1 是"给定检测点+示向度"，这里按统计口径抽样）
#   站点：n ~ U{3,...,6}；方向与距离按下面两种布站方案
#   示向度：= 真实方位角 + e, e ~ U[-1°,1°]（故定位区域必含真源、非空）
#   无界（直径无穷）与空区域单独计数，不计入覆盖率分母
# --------------------------------------------------------------------------
def exp13_coverage_table(trials=2000, seed=17, n_lo=3, n_hi=6):
    rng = np.random.default_rng(seed)
    layouts = {
        "uniform": "全场均匀：以源为心、半径 1500 m 圆域内均匀撒点",
        "arc100": "绕源等半径圆弧：R=1000 m，等角间隔，张角 Θ~U[100°,360°]，弧朝向随机",
        "arc90": "绕源等半径圆弧：R=1000 m，等角间隔，张角 Θ~U[90°,360°]，弧朝向随机",
    }
    out = {}
    for key in layouts:
        hit = valid = unbounded = empty = 0
        etas = []
        for _ in range(trials):
            n = int(rng.integers(n_lo, n_hi + 1))
            if key == "uniform":
                rho = 1500.0 * np.sqrt(rng.uniform(0, 1, size=n))
                az = rng.uniform(0, 360, size=n)
            else:
                span = float(rng.uniform(100.0 if key == "arc100" else 90.0, 360.0))
                start = float(rng.uniform(0, 360))
                az = start + np.linspace(0.0, span, n)
                rho = np.full(n, 1000.0)
            pts = [rho[i] * _u(float(az[i])) for i in range(n)]
            errs = rng.uniform(-1, 1, size=n)
            thetas = []
            for s, e in zip(pts, errs):
                true = math.degrees(math.atan2(-s[1], -s[0]))
                thetas.append(true + e)
            poly = region(pts, thetas)
            if not poly:
                empty += 1
                continue
            d, i, j = diameter_pair(poly)
            if d > 1e4:
                unbounded += 1
                continue
            valid += 1
            hit += int(covered_by_diameter_circle(poly)[0])
            etas.append(covered_by_diameter_circle(poly)[3])
        out[key] = (hit / max(1, valid), valid, unbounded, empty,
                    float(np.mean(etas)) if etas else float("nan"),
                    float(np.max(etas)) if etas else float("nan"))
    return out


def solve_area_opt_with_alpha(r_lo, r_max=1500.0, tol=1e-3):
    """面积口径在 α∈[25°,155°] 约束下的最优解（α 在 r∈[r_lo,r_max] 上检验，
    最坏值在 r_max）。α 关于 β 单调，故最优点取 α(r_lo)=25° 的边界。"""
    lo, hi = 5.0, 45.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        L = boundary_L(mid, r_lo)
        if _feasible(L, mid, r_lo, r_max)[0]:
            hi = mid
        else:
            lo = mid
    beta, L = hi, boundary_L(hi, r_lo)
    wd, wa = exp6_worst_metrics(L, beta, r_lo, r_max, n_r=1501)
    return beta, L, wd[0], wa[0], q2_alpha(L, beta, r_lo)


def exp14_sampler_sensitivity(trials=1200, seed=5, span_lo=100.0):
    """圆弧布站口径的敏感性分解：等角间隔 vs 张角内均匀随机 × 等半径 vs 随机半径。"""
    rng = np.random.default_rng(seed)
    out = {}
    for spacing in ("equal", "uniform"):
        for rho_mode in ("fixed", "random"):
            valid = hit = unbounded = empty = 0
            for _ in range(trials):
                n = int(rng.integers(3, 7))
                span = float(rng.uniform(span_lo, 360.0))
                start = float(rng.uniform(0, 360))
                rel = (np.linspace(0.0, span, n) if spacing == "equal"
                       else np.sort(rng.uniform(0.0, span, n)))
                az = start + rel
                rho = (np.full(n, 1000.0) if rho_mode == "fixed"
                       else rng.uniform(300.0, 1500.0, n))
                pts = [rho[i] * _u(float(az[i])) for i in range(n)]
                errs = rng.uniform(-1, 1, size=n)
                thetas = [math.degrees(math.atan2(-p[1], -p[0])) + e
                          for p, e in zip(pts, errs)]
                poly = region(pts, thetas)
                if not poly:
                    empty += 1
                    continue
                d, _, _ = diameter_pair(poly)
                if d > 1e4:
                    unbounded += 1
                    continue
                valid += 1
                hit += int(covered_by_diameter_circle(poly)[0])
            out[(spacing, rho_mode)] = (hit / max(1, valid), valid, unbounded, empty)
    return out


def convex_hull(pts) -> list[np.ndarray]:
    """Monotone chain，返回逆时针凸包顶点（不含重复首点）。"""
    p = np.asarray(pts, dtype=float)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def half(ps):
        st: list[np.ndarray] = []
        for q in ps:
            while len(st) >= 2 and float(np.cross(st[-1] - st[-2], q - st[-2])) <= 0:
                st.pop()
            st.append(q)
        return st
    hull = half(p)[:-1] + half(p[::-1])[:-1]
    return [np.asarray(v) for v in hull]


def rotating_calipers_diameter(hull) -> float:
    """凸多边形直径（旋转卡壳 O(m)）。"""
    h = np.asarray(hull, dtype=float)
    n = len(h)
    if n == 1:
        return 0.0
    if n == 2:
        return float(np.linalg.norm(h[0] - h[1]))
    k = 1
    best = 0.0
    for i in range(n):
        ni = (i + 1) % n
        edge = h[ni] - h[i]
        while True:
            nk = (k + 1) % n
            cur = abs(float(np.cross(edge, h[k] - h[i])))
            nxt = abs(float(np.cross(edge, h[nk] - h[i])))
            if nxt > cur:
                k = nk
            else:
                break
        best = max(best, float(np.linalg.norm(h[i] - h[k])),
                   float(np.linalg.norm(h[ni] - h[k])))
    return best


def exp15_selfcheck(polys=200, sym=300, seed=23):
    """自校验：旋转卡壳 vs 暴力直径（随机凸多边形）；中心对称族是否 100% 被覆盖。"""
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(polys):
        m = int(rng.integers(3, 11))
        hull = convex_hull(rng.normal(size=(m, 2)) * 10.0)
        if len(hull) < 3:
            continue
        max_err = max(max_err, abs(rotating_calipers_diameter(hull) - diameter(hull)))
    hit = 0
    for _ in range(sym):
        pts = rng.normal(size=(int(rng.integers(2, 7)), 2))
        pts = np.vstack([pts, -pts])                 # 中心对称点集
        hull = convex_hull(pts)
        if len(hull) < 3:
            continue
        hit += int(covered_by_diameter_circle(hull)[0])
    return max_err, hit, sym


def main() -> None:
    r_max = 1500.0
    print("=" * 78)
    print("[E1] 论文最优 S2 的 D(r)（L=1004.10, β=34.78°）")
    best = exp1_dr_curve(1004.10, 34.78, 5.0, r_max)
    print()
    print("[E2] 两口径互查")
    for r_lo, tag, L, b in ((5.0, "面积最优(论文原报)", 1004.62, 22.41),
                            (5.0, "直径最优", 1004.10, 34.78),
                            (100.0, "面积最优(论文原报)", 1095.02, 17.32),
                            (100.0, "直径最优", 1086.23, 29.06)):
        wd, wa = exp6_worst_metrics(L, b, r_lo)
        print("   r_lo=%-4.0f %-18s L=%8.2f β=%6.2f°  maxD=%8.3f  maxA=%9.2f  α(r_lo)=%.2f°%s"
              % (r_lo, tag, L, b, wd[0], wa[0], q2_alpha(L, b, r_lo),
                 "" if q2_alpha(L, b, r_lo) >= 25 - 1e-9 else "  <-- 违反 α≥25°"))
    print()
    print("[E3] 面积口径在 α 约束下的改正最优解")
    for r_lo in (5.0, 100.0):
        b, L, d, a, al = solve_area_opt_with_alpha(r_lo, r_max)
        print("   r_lo=%-4.0f β*=%.2f°  L*=%.2f m  maxA=%.2f m²  maxD=%.3f m  α(r_lo)=%.2f°"
              % (r_lo, b, L, a, d, al))
    print()
    print("[E4] β 一维极小极大（沿 ρ2(r_lo)=1000 边界）")
    for r_lo in (5.0, 100.0):
        prof = exp5_beta_profile(r_lo, r_max, errs=((0.0, 0.0),))
        b, L, d, arg = min(prof, key=lambda t: t[2])
        prof_e = exp5_beta_profile(r_lo, r_max,
                                   errs=((0.0, 0.0), (-1.0, 1.0), (1.0, -1.0),
                                         (-1.0, -1.0), (1.0, 1.0)))
        be, Le, de, arge = min(prof_e, key=lambda t: t[2])
        print("   r_lo=%-4.0f  仅对 r: β*=%.2f° L*=%.2f maxD=%.3f @r=%.0f | "
              "含 ±1° 误差: β*=%.2f° maxD=%.3f @r=%.0f e=(%.0f,%.0f)"
              % (r_lo, b, L, d, arg[0], be, de, arge[0], arge[1], arge[2]))
    print("   增益 = %.2f%%" % ((134.058 / 121.674 - 1) * 100))
    print()
    print("[E5] 论文最优 S2 上把 ±1° 误差计入最坏情形的细网格结果")
    w, wz, mean, mn = exp4_error_sensitivity(1004.10, 34.78, 5.0, r_max)
    print("   max(r,e1,e2) D=%.3f @ e=(%.2f,%.2f) r=%.0f ; 仅 e=0 时 %.3f"
          % (w[0], w[1][0], w[1][1], w[1][2], wz[0]))
    print()
    print("[E6] Jung 界")
    lo, hi, ub, et = exp9_jung(trials=8000)
    print("   随机凸多边形 η∈[%.6f,%.6f]，理论上界 %.10f，等边三角形 %.10f" % (lo, hi, ub, et))
    print()
    print("[E7] 问题一覆盖率表（规范采样器，2000 例/行）")
    for key, val in exp13_coverage_table().items():
        print("   %-8s 覆盖率=%.3f  有效=%4d  无界=%3d  空=%3d  平均η=%.4f 最大η=%.4f"
              % (key, val[0], val[1], val[2], val[3], val[4], val[5]))
    print()
    print("[E8] 圆弧布站口径的敏感性分解（1200 例/格）")
    for (spacing, rho_mode), val in exp14_sampler_sensitivity().items():
        print("   等角间隔=%-7s 半径=%-6s 覆盖率=%.3f (有效%4d 无界%3d 空%3d)"
              % (spacing, rho_mode, val[0], val[1], val[2], val[3]))
    print()
    print("[E9] 自校验")
    err, hit, sym = exp15_selfcheck()
    print("   旋转卡壳 vs 暴力直径 200 例最大误差 = %.3g m；中心对称族覆盖 %d/%d"
          % (err, hit, sym))


if __name__ == "__main__":
    main()
