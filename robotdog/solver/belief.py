"""问题3 的贝叶斯信念: "该频道是否存在源" × "源在哪".

三种表示 (按信息量递进), 每种都只使用机器狗可见的信息:
  mode 0  PRIOR  未收到任何示向度: 100 m 分辨率占用网格 (≈1000 格), 描述
                 "源还可能在哪"。no_signal 观测乘核 P(R<d)=clamp((d-1000)/500)。
  mode 1  RAY    只有一条示向度: 源被压在一条射线上, 纵向距离未知。用 128 个
                 **沿射线的一维样本**表示 (径向密度 ∝ r·P(R≥r)); no_signal 观测
                 按 P(R<d) 重加权。一维重采样不会退化 (粒子滤波退化的根源是
                 "窄楔形 × 二维均匀先验", 一维表示直接绕开了它)。
  mode 2  POINT  ≥2 条示向度: 角度残差 Gauss-Newton 最小二乘 + 解析协方差
                 Σ = (JᵀJ/σ² + 先验信息)⁻¹, σ=0.5774° (对应误差 ~ U(-1°,1°))。

关键: "可能在哪里" 与 "检测概率" 必须用同一套表示算, 否则会出现
"反复去同一个点探测" (同点同频道读数不变, 不带来任何新信息)。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from ..consts import ARENA_RADIUS_M, CLEAR_RADIUS_M, N_CHANNELS, RADIUS_MAX_M, RADIUS_MIN_M

SIGMA_DEG = 2.0 / math.sqrt(12.0)
SIGMA_RAD = math.radians(SIGMA_DEG)
GRID_STEP = 100.0
# 源必在半径 1800 m 的目标区域内 (题目给定), 因此占用网格的先验支撑就是该圆域:
# 既符合模型, 又让收尾的"1000 m 覆盖"判据只在真正可能有源的范围内要求覆盖
# (原先铺到 1900 m 会把圆域外一圈也算进覆盖要求, 白白多跑路)。
GRID_RADIUS = 1800.0
N_RAY = 128


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def build_grid(step: float = GRID_STEP, radius: float = GRID_RADIUS) -> np.ndarray:
    n = int(radius / step)
    xs = (np.arange(-n, n + 1) * step).astype(np.float32)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    keep = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= radius * radius
    return np.ascontiguousarray(pts[keep])


def perp_dir(cov: np.ndarray) -> Tuple[float, float]:
    """协方差最大特征值方向 (位置最不确定方向)。"""
    tr_ = float(cov[0, 0] + cov[1, 1])
    det = float(cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0])
    disc = max(tr_ * tr_ / 4.0 - det, 0.0) ** 0.5
    l1 = tr_ / 2.0 + disc
    if abs(cov[0, 1]) > 1e-12:
        v = (l1 - cov[1, 1], cov[0, 1])
    else:
        v = (1.0, 0.0) if cov[0, 0] >= cov[1, 1] else (0.0, 1.0)
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-12 else (1.0, 0.0)


def max_eig(cov: np.ndarray) -> float:
    tr = cov[0, 0] + cov[1, 1]
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    return float(max(tr / 2.0 + disc, 0.0))


class ChannelTrack:
    """单频道信念。"""

    def __init__(self, idx: int, grid: np.ndarray,
                 rng: Optional[np.random.Generator] = None) -> None:
        self.idx = idx
        # 射线重采样需要随机数, 但**必须可复现**: 策略/规划器的评测要能"一跑一个数"。
        # 因此随机数发生器由 Belief 统一创建并注入, 播种固定 (见 Belief.__init__)。
        self.rng = rng if rng is not None else np.random.default_rng(20260911)
        self.pi = 13.0 / 20.0
        self.mode = 0                       # 0=PRIOR 1=RAY 2=POINT
        self.grid = grid.astype(np.float32).copy()   # mode 0: 归一化位置分布
        self.grid_pts = None
        self.ray_x: Optional[np.ndarray] = None      # mode 1: (N,2) 沿射线样本
        self.ray_w: Optional[np.ndarray] = None
        self.ray_src: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # (mx,my,bearing)
        self.cleared = False
        self.obs: List[Tuple[float, float, float]] = []
        self.n_dir = 0
        self.n_obs = 0
        self.n_no = 0
        self.near_pt: Optional[Tuple[float, float]] = None
        self.est: Optional[Tuple[float, float]] = None
        self.cov: Optional[np.ndarray] = None
        self.r_lo = RADIUS_MIN_M
        self.r_hi = RADIUS_MAX_M
        self.no_pts: List[Tuple[float, float]] = []
        self.last_t = -1e9
        self.meas_pts: List[Tuple[float, float]] = []
        # --- 有效接收半径 R 的**联合**似然所需的基准量 ---
        # R 是该源固定不变的隐变量, 多次 no_signal 观测给出的是 R < min_i d_i,
        # 因此位置权重必须由"未观测先验 × F(min 距离)"重算, 不能把 F(d_i) 连乘
        # (连乘等于假设每次观测都重新抽一次 R, 会过度削权)。详见 _grid_weights。
        self.grid0 = self.grid.astype(np.float64)                 # 未观测先验 (重算基准)
        self._dmin0 = np.full(self.grid.shape, np.inf)            # 格点到 no_signal 点的最小距离
        self._keep0 = np.ones(self.grid.shape, dtype=np.float64)  # 清除失败给出的硬约束
        self.ray_r: Optional[np.ndarray] = None                   # 射线样本到测点的距离
        self._dmin1: Optional[np.ndarray] = None
        self._keep1: Optional[np.ndarray] = None
        self._Z = 1.0                                             # 当前位置表示的归一化常数
        # True (问题3): no_signal / 清除失败按 R 的联合似然**重算**位置后验;
        # False (问题4 沿用原口径): 仍按传入似然做乘性更新, 保持问题4 已标定结果不变。
        self.r_joint = True

    # ---------------- 统计量 ----------------
    def std_max(self) -> float:
        if self.cov is None:
            return 1e6
        return float(math.sqrt(max(max_eig(self.cov), 0.0)))

    def r_bounds(self) -> Tuple[float, float]:
        """有效接收半径 R 的后验区间 [lo, hi]。

        **关键**: 与位置估计一致地使用全部观测 ——
          * 在距离 d 处**收到过**信号 ⇒ R >= d  (下界, 原来漏掉了这一半!)
          * 在距离 d 处**没收到**信号 ⇒ R <  d  (上界)
        只写上界会让"离得远一点的 no_signal"把已经确认存在的源的存在概率 pi 一路压低,
        实测导致约 2% 的已定位源被无谓放弃 (pi 掉到 min_pi 以下, 再没人去清)。
        """
        lo, hi = RADIUS_MIN_M, RADIUS_MAX_M
        est = self.est
        if est is not None:
            for (mx, my, _b) in self.obs:
                lo = max(lo, math.hypot(mx - est[0], my - est[1]))
            for (mx, my) in self.no_pts:
                hi = min(hi, math.hypot(mx - est[0], my - est[1]))
        return lo, max(lo, hi)

    def r_est(self) -> float:
        """有效接收半径 R 的点估计 (后验区间中点); 供收益折算使用。"""
        lo, hi = self.r_bounds()
        return 0.5 * (lo + hi)

    def active(self) -> bool:
        return (not self.cleared) and self.pi > 1e-6

    def is_measured(self, p: Tuple[float, float], tol: float = 0.5) -> bool:
        """该点是否已对本站做过该频道检测。

        模拟器对同一 (位置, 频道) 的读数固定 (附件2 §2.3), 因此重复检测不带来
        任何新信息。策略必须显式排除这种点, 否则会退化成"原地反复检测"。
        """
        for (qx, qy) in self.meas_pts:
            if abs(qx - p[0]) < tol and abs(qy - p[1]) < tol:
                return True
        return False

    # ---------------- 检测概率 (用于探测点收益) ----------------
    def detect_prob(self, q: Tuple[float, float], grid_pts: np.ndarray,
                    kernel: Optional[np.ndarray] = None) -> float:
        """P(在 q 处能收到该频道信号 | 源存在且未清除)。"""
        if self.mode == 0:
            if self.grid is None:
                return 0.0
            if kernel is None:
                d = np.hypot(grid_pts[:, 0] - q[0], grid_pts[:, 1] - q[1])
                kernel = np.clip((RADIUS_MAX_M - d) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0)
            return float(np.dot(self.grid, kernel))
        if self.mode == 1:
            d = np.hypot(self.ray_x[:, 0] - q[0], self.ray_x[:, 1] - q[1])
            k = np.clip((RADIUS_MAX_M - d) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0)
            return float(np.dot(self.ray_w, k))
        # mode 2: 高斯近似
        return float(self._gauss_detect_prob(q))

    def _gauss_detect_prob(self, q: Tuple[float, float]) -> float:
        if self.est is None or self.cov is None:
            return 0.0
        s = max(self.std_max(), 1.0)
        d = math.hypot(self.est[0] - q[0], self.est[1] - q[1])
        # P(d' <= R), d' ~ N(d, s²), R ~ U(lo, hi) —— 区间由观测给出
        lo, hi = self.r_bounds()
        if d <= lo - 3.0 * s:
            return 1.0
        if d >= hi + 3.0 * s:
            return 0.0
        # 数值积分: P(d' <= R) = E_{d'}[ P(R >= d') ] = 1 - E_{d'}[F_R(d')]
        # 注意方向: 早先写成了 E[F_R(d')] (自己的补集), 于是在 d 远离 R 区间时
        # 把"几乎必然的 no_signal"算成"几乎必然能收到", 反过来把已确认存在的源
        # 的存在概率 pi 打到接近 0 —— 这是漏清的直接原因。
        grid = np.linspace(max(lo - 3 * s, d - 3 * s), min(hi + 3 * s, d + 3 * s), 9)
        w = np.exp(-0.5 * ((grid - d) / s) ** 2)
        # 网格点全部下溢 (距离远超 R 区间) 时权重和为 0, 不做归一化会得到 NaN。
        # 这里做成显式契约而不是靠调用方各自兜底。
        if w.sum() <= 1e-12:
            return 0.0
        w /= w.sum()
        cdf = np.clip((grid - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        return float(np.dot(w, 1.0 - cdf))

    # ---------------- 观测更新 ----------------
    def add_direction(self, mx: float, my: float, svd: float, t: float) -> None:
        self.meas_pts.append((mx, my))
        for (ox, oy, ob) in self.obs:
            if abs(ox - mx) < 0.5 and abs(oy - my) < 0.5 and abs(ob - svd) < 1e-9:
                return
        self.obs.append((mx, my, svd))
        self.n_dir += 1
        self.n_obs += 1
        self.last_t = t
        self.pi = 1.0
        if self.n_dir == 1:
            self.mode = 1
            self.ray_src = (mx, my, svd)
            self._build_ray()
            self.est = self.ray_mean()
            self.cov = self._ray_cov()
        else:
            self.mode = 2
            self.grid = None                       # type: ignore
            self.ray_x = None                      # type: ignore
            self.ray_w = None                      # type: ignore
            self.refit()

    def add_near(self, mx: float, my: float, t: float) -> None:
        self.meas_pts.append((mx, my))
        self.near_pt = (mx, my)
        self.pi = 1.0
        self.n_obs += 1
        self.last_t = t
        self.mode = 2
        self.est = (mx, my)
        self.cov = np.zeros((2, 2))
        self.grid = None            # type: ignore
        self.ray_x = None           # type: ignore

    def apply_no_signal(self, q: Tuple[float, float], L: float,
                        grid_pts: np.ndarray, grid_kernel: Optional[np.ndarray]) -> None:
        self.meas_pts.append(q)
        self.no_pts.append(q)
        self.n_obs += 1
        self.n_no += 1
        if self.mode == 0 and self.grid0 is not None and grid_pts is not None:
            d = np.hypot(grid_pts[:, 0] - q[0], grid_pts[:, 1] - q[1])
            self._dmin0 = np.minimum(self._dmin0, d)
            if self.r_joint:
                L = self._recompute_grid()
            elif self.grid is not None and grid_kernel is not None:
                # 兼容口径 (问题4): 乘性更新
                self.grid = (self.grid * grid_kernel).astype(np.float32)
                s = float(self.grid.sum())
                if s > 1e-12:
                    self.grid /= s
                else:
                    self.grid = None      # type: ignore
        elif self.mode == 1 and self.ray_x is not None:
            d = np.hypot(self.ray_x[:, 0] - q[0], self.ray_x[:, 1] - q[1])
            self._dmin1 = np.minimum(self._dmin1, d)
            if self.r_joint:
                L = self._recompute_ray()
            else:
                k = self._F(d)
                self.ray_w = self.ray_w * k
                s = float(self.ray_w.sum())
                if s > 1e-12:
                    self.ray_w /= s
                    self._resample_ray()
                else:
                    self.ray_w[:] = 1.0 / len(self.ray_w)
                self.est = self.ray_mean()
                self.cov = self._ray_cov()
        elif self.mode == 2 and self.est is not None:
            d = math.hypot(self.est[0] - q[0], self.est[1] - q[1])
            # 收不到 ⇒ 该点超出有效接收半径, 故 R < d
            self.r_hi = min(self.r_hi, d)
        p = self.pi
        post = p * L / (p * L + (1.0 - p)) if (p * L + 1.0 - p) > 0 else 0.0
        self.pi = float(min(max(post, 1e-12), 1.0 - 1e-12))

    def apply_clear_miss(self, q: Tuple[float, float]) -> None:
        """清除失败。题目规定清除成功与否只取决于与源的距离, 因此这是一条硬观测:
        "源不在以 q 为心、20 m 为半径的圆内"。mode 0/1 直接把它作为约束重算位置
        似然 (L 取真实的似然比); mode 2 的位置是高斯近似, 仍用原来的保守启发式。"""
        if self.mode == 1 and self.ray_x is not None and self.r_joint and self._keep1 is not None:
            d = np.hypot(self.ray_x[:, 0] - q[0], self.ray_x[:, 1] - q[1])
            self._keep1 = self._keep1 * (d > CLEAR_RADIUS_M)
            L = self._recompute_ray()
            p = self.pi
            post = p * L / (p * L + (1.0 - p)) if (p * L + 1.0 - p) > 0 else 0.0
            self.pi = float(min(max(post, 1e-12), 1.0 - 1e-12))
            return
        if self.mode == 1 and self.ray_x is not None and not self.r_joint:
            # 兼容口径 (问题4): 乘性更新
            d = np.hypot(self.ray_x[:, 0] - q[0], self.ray_x[:, 1] - q[1])
            keep = (d > CLEAR_RADIUS_M).astype(np.float64)
            L = float(np.dot(self.ray_w, keep))
            if L > 1e-12:
                self.ray_w = self.ray_w * keep / L
                self._resample_ray()
                self.est = self.ray_mean()
                self.cov = self._ray_cov()
            p = self.pi
            post = p * L / (p * L + (1.0 - p))
            self.pi = float(min(max(post, 1e-12), 1.0 - 1e-12))
            return
        if (self.mode == 0 and self.grid0 is not None and self.r_joint
                and self.grid_pts is not None):
            d = np.hypot(self.grid_pts[:, 0] - q[0], self.grid_pts[:, 1] - q[1])
            self._keep0 = self._keep0 * (d > CLEAR_RADIUS_M)
            L = self._recompute_grid()
            p = self.pi
            post = p * L / (p * L + (1.0 - p)) if (p * L + 1.0 - p) > 0 else 0.0
            self.pi = float(min(max(post, 1e-12), 1.0 - 1e-12))
            return
        if (self.mode == 0 and not self.r_joint and self.grid is not None
                and self.grid_pts is not None):
            # 兼容口径 (问题4): 乘性更新
            d = np.hypot(self.grid_pts[:, 0] - q[0], self.grid_pts[:, 1] - q[1])
            keep = (d > CLEAR_RADIUS_M).astype(np.float32)
            L = float(np.dot(self.grid, keep))
            if L > 1e-12:
                self.grid = self.grid * keep / L
        if self.mode == 2 and self.est is not None and self.cov is not None:
            d = math.hypot(self.est[0] - q[0], self.est[1] - q[1])
            if d < 80.0:
                self.cov = self.cov * 2.5
        p = self.pi
        L = 0.97
        if self.est is not None:
            d = math.hypot(self.est[0] - q[0], self.est[1] - q[1])
            if d <= CLEAR_RADIUS_M:
                L = 0.02
        post = p * L / (p * L + (1.0 - p))
        self.pi = float(min(max(post, 1e-12), 1.0 - 1e-12))

    # ---------------- 有效接收半径 R 的联合似然 (重算型更新) ----------------
    @staticmethod
    def _F(d: np.ndarray) -> np.ndarray:
        """P(R < d), R ~ U(RADIUS_MIN_M, RADIUS_MAX_M)。"""
        return np.clip((d - RADIUS_MIN_M) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0)

    def _recompute_grid(self) -> float:
        """由未观测先验重算 mode 0 的权重, 返回本次观测的似然比 L = Z_new/Z_old。"""
        w = self.grid0 * self._F(self._dmin0) * self._keep0
        Z = float(w.sum())
        L = (Z / self._Z) if self._Z > 1e-300 else 0.0
        if Z > 1e-12:
            self.grid = (w / Z).astype(np.float32)
        else:
            self.grid = None      # type: ignore
        self._Z = Z
        return L

    def _recompute_ray(self) -> float:
        """由未观测先验重算 mode 1 的权重, 返回似然比 L = Z_new/Z_old。

        每个射线样本的先验质量相同 (采样半径 r = r_max*sqrt(u) 已含面积权重 ∝ r),
        样本权重 ∝ P(r ≤ R < min_i d_i) = max(0, F(min_i d_i) - F(r))。
        """
        if self._dmin1 is None or self._keep1 is None or self.ray_r is None:
            return 1.0
        w = np.maximum(self._F(self._dmin1) - self._F(self.ray_r), 0.0) * self._keep1
        Z = float(w.mean())
        L = (Z / self._Z) if self._Z > 1e-300 else 0.0
        if Z > 1e-15:
            self.ray_w = w / float(w.sum())
            self.est = self.ray_mean()
            self.cov = self._ray_cov()
        self._Z = Z
        return L

    # ---------------- 射线表示 ----------------
    def _build_ray(self) -> None:
        mx, my, b = self.ray_src
        a = math.radians(b)
        ux, uy = math.cos(a), math.sin(a)
        # 沿射线的可达半径: 与圆域边界的交点 (上限 1500)
        # |m + r u| <= R_arena
        bq = 2.0 * (mx * ux + my * uy)
        cq = mx * mx + my * my - ARENA_RADIUS_M ** 2
        disc = bq * bq - 4.0 * cq
        r_arena = (-bq + math.sqrt(disc)) / 2.0 if disc > 0 else 0.0
        r_max = max(60.0, min(RADIUS_MAX_M, r_arena))
        # 径向先验 ∝ r·P(R>=r) ∝ r·clamp((1500-r)/500)
        u = np.linspace(0.0, 1.0, N_RAY)
        r = r_max * np.sqrt(u)                      # 面积权重 ∝ r
        self.ray_x = np.stack([mx + ux * r, my + uy * r], axis=-1).astype(np.float64)
        self.ray_r = r
        # 把"该频道此前所有 no_signal 观测"一并作为约束: R < min_i d_i,
        # 与"收到信号 ⇒ R >= r"合并成样本权重 ∝ max(0, F(min_i d_i) - F(r))。
        self._dmin1 = np.full(len(r), np.inf)
        for (qx, qy) in self.no_pts:
            self._dmin1 = np.minimum(
                self._dmin1, np.hypot(self.ray_x[:, 0] - qx, self.ray_x[:, 1] - qy))
        self._keep1 = np.ones(len(r))
        w = np.maximum(self._F(self._dmin1) - self._F(r), 1e-12)
        self._Z = float(w.mean())
        self.ray_w = w / float(w.sum())

    def _resample_ray(self) -> None:
        cum = np.cumsum(self.ray_w)
        cum[-1] = 1.0
        n = len(cum)
        # **系统重采样** (systematic resampling), 且固定相位 (u = 0.5)。
        # 原来的多项式重采样每次抽样不同, 同一案例同一策略会跑出 13% 的差异
        # (实测 260 ~ 296 s/源), 评测与回归都没法做。系统重采样是粒子滤波里的标准做法:
        # 期望不变、方差更小; 取固定相位后完全确定, 而偏差可以忽略。
        pos = (np.arange(n) + 0.5) / n
        idx = np.clip(np.searchsorted(cum, pos), 0, n - 1)
        self.ray_x = self.ray_x[idx]
        self.ray_w = np.full(len(idx), 1.0 / len(idx))

    def ray_mean(self) -> Optional[Tuple[float, float]]:
        if self.ray_x is None:
            return None
        return (float(np.dot(self.ray_x[:, 0], self.ray_w)),
                float(np.dot(self.ray_x[:, 1], self.ray_w)))

    def _ray_cov(self) -> Optional[np.ndarray]:
        mu = self.ray_mean()
        if mu is None:
            return None
        d = self.ray_x - np.array(mu)
        return (d * self.ray_w[:, None]).T @ d

    # ---------------- 点估计 (Gauss-Newton) ----------------
    def refit(self) -> None:
        obs = self.obs
        n = len(obs)
        if n < 2:
            return
        g = self._seed()
        if g is None:
            return
        gx, gy = g
        for _ in range(15):
            JtJ = np.zeros((2, 2))
            Jtr = np.zeros(2)
            for (mx, my, b) in obs:
                dx, dy = gx - mx, gy - my
                d2 = dx * dx + dy * dy
                if d2 < 1.0:
                    d2 = 1.0
                th = math.atan2(dy, dx)
                r = _wrap(th - math.radians(b))
                jx, jy = -dy / d2, dx / d2
                JtJ[0, 0] += jx * jx; JtJ[0, 1] += jx * jy
                JtJ[1, 0] += jx * jy; JtJ[1, 1] += jy * jy
                Jtr[0] += jx * r; Jtr[1] += jy * r
            tr = JtJ[0, 0] + JtJ[1, 1]
            JtJ[0, 0] += 1e-13 + 1e-9 * tr
            JtJ[1, 1] += 1e-13 + 1e-9 * tr
            det = JtJ[0, 0] * JtJ[1, 1] - JtJ[0, 1] * JtJ[1, 0]
            if abs(det) < 1e-26:
                break
            sx = (-Jtr[0] * JtJ[1, 1] + Jtr[1] * JtJ[0, 1]) / det
            sy = (-Jtr[1] * JtJ[0, 0] + Jtr[0] * JtJ[1, 0]) / det
            step = math.hypot(sx, sy)
            if step > 600.0:
                sx *= 600.0 / step; sy *= 600.0 / step
            gx += sx; gy += sy
            if math.hypot(sx, sy) < 0.05:
                break
        if math.hypot(gx, gy) > ARENA_RADIUS_M * 1.5:
            return
        self.est = (gx, gy)
        Info = self._info_matrix(self.obs, gx, gy)
        det = Info[0, 0] * Info[1, 1] - Info[0, 1] * Info[1, 0]
        self.cov = np.array([[Info[1, 1], -Info[0, 1]], [-Info[1, 0], Info[0, 0]]]) / det

    @staticmethod
    def _info_matrix(obs, gx: float, gy: float) -> np.ndarray:
        JtJ = np.zeros((2, 2))
        for (mx, my, b) in obs:
            dx, dy = gx - mx, gy - my
            d2 = max(dx * dx + dy * dy, 1.0)
            jx, jy = -dy / d2, dx / d2
            JtJ[0, 0] += jx * jx; JtJ[0, 1] += jx * jy
            JtJ[1, 0] += jx * jy; JtJ[1, 1] += jy * jy
        Info = JtJ / (SIGMA_RAD ** 2)
        Info[0, 0] += 1.0 / (900.0 ** 2) + 1e-14
        Info[1, 1] += 1.0 / (900.0 ** 2) + 1e-14
        return Info

    def cov_with_new(self, p: Tuple[float, float]) -> np.ndarray:
        """若在 p 再测一次, 位置协方差会变成多少 (mode 2)。"""
        if self.est is None or self.cov is None:
            return np.eye(2) * 1e10
        dx, dy = self.est[0] - p[0], self.est[1] - p[1]
        d2 = max(dx * dx + dy * dy, 1.0)
        d = math.sqrt(d2)
        tx, ty = -dy / d, dx / d
        Ip = np.array([[tx * tx, tx * ty], [tx * ty, ty * ty]]) / ((SIGMA_RAD * d) ** 2)
        try:
            S = np.linalg.inv(self.cov + np.eye(2) * 1e-9)
        except np.linalg.LinAlgError:
            S = np.eye(2) * 1e-9
        try:
            return np.linalg.inv(S + Ip)
        except np.linalg.LinAlgError:
            return self.cov

    def _seed(self) -> Optional[Tuple[float, float]]:
        obs = self.obs
        cands: List[Tuple[float, float]] = []
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                p = _line_intersection(obs[i], obs[j])
                if p is not None and math.hypot(p[0], p[1]) < 4200.0:
                    cands.append(p)
        if cands:
            xs = sorted(c[0] for c in cands)
            ys = sorted(c[1] for c in cands)
            m = len(cands) // 2
            return (xs[m], ys[m])
        mx, my, b = obs[0]
        a = math.radians(b)
        return (mx + 1150.0 * math.cos(a), my + 1150.0 * math.sin(a))


def _line_intersection(o1, o2) -> Optional[Tuple[float, float]]:
    (x1, y1, b1), (x2, y2, b2) = o1, o2
    a1, a2 = math.radians(b1), math.radians(b2)
    d1 = (math.cos(a1), math.sin(a1))
    d2 = (math.cos(a2), math.sin(a2))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-3:
        return None
    dx, dy = x2 - x1, y2 - y1
    t = (dx * d2[1] - dy * d2[0]) / den
    return (x1 + t * d1[0], y1 + t * d1[1])


class Belief:
    """20 个频道的联合信念。"""

    def __init__(self, prior_pi: float = 13.0 / 20.0, seed: int = 20260911) -> None:
        self.pts = build_grid()
        w = np.full(self.pts.shape[0], 1.0, dtype=np.float32)
        w /= w.sum()
        # 每个 Belief 一个独立的、播种固定的随机数发生器 —— 射线重采样用它。
        # 这样"同一案例 + 同一策略"每次跑出来逐位一致, 评测/回归才有意义。
        rng = np.random.default_rng(int(seed))
        self.tracks: List[ChannelTrack] = [ChannelTrack(i, w, rng) for i in range(N_CHANNELS)]
        for t in self.tracks:
            t.pi = prior_pi
            t.grid_pts = self.pts

    # ---------- 观测更新 ----------
    def on_measure(self, mx: float, my: float, channel: int, kind: str,
                   svd: Optional[float], t: float) -> None:
        tr = self.tracks[channel - 1]
        if kind == "direction":
            tr.add_direction(mx, my, float(svd), t)
            return
        if kind == "near":
            tr.add_near(mx, my, t)
            return
        # no_signal: 该频道在该点收不到信号 ⇒ R < d(x)。
        # mode 0/1 由 ChannelTrack 用"未观测先验 × F(min 距离)"重算位置似然,
        # 且似然比 L = Z_new/Z_old 由它内部给出; mode 2 的位置是高斯近似, 这里
        # 仍给预测概率 P(no_signal) = 1 - P(检测)。
        L = 1.0 - tr._gauss_detect_prob((mx, my)) if tr.mode == 2 else 1.0
        tr.apply_no_signal((mx, my), L, self.pts, None)

    def on_clear(self, mx: float, my: float, channel: int, hit: bool) -> None:
        tr = self.tracks[channel - 1]
        if hit:
            tr.cleared = True
            tr.pi = 0.0
            return
        tr.apply_clear_miss((mx, my))
