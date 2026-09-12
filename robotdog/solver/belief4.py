"""问题4 的贝叶斯信念: "该频道是否存在源" × "源在哪" × "是不是定向源" × "定向朝哪".

与问题3 (``belief.py``) 的唯一物理差别是**观测模型多了一个角度因子**:

    收到信号  <=>  d <= R  且  |wrap(psi - beta)| <= 90        (附录1 第3条)
    其中 d = 检测点到源的距离, R = 有效接收半径, beta = 检测点->源 的真实方位,
    psi = 定向方向 (全向源视为恒满足)。

由此得到三类信息, 本模块都显式建模 (推导与数值验证见 ``tools/probes/verify_q4_theory4.py``):

  * **收到方向** ⇒ psi ∈ [beta-90, beta+90] (半圆), 且源的存在概率 π 归 1;
  * **未收到** ⇒ 若 d 已落在 R 的可能范围内, 则 psi ∉ [beta-90, beta+90] (背对),
    即 psi 落在**另一半圆**里 —— 这是问题4 相对问题3 新增的、最强的一类信息;
  * **两次检测的联合关系**: 已知在 p 收到, 则在 q 再次收到的概率
    ``1 - |wrap(phi_q - phi_p)|/180`` (phi = 源->检测点 的方位 = beta+180)。
    它在 "q 与 p 对源同侧同向" 时为 1, "源在 p、q 之间" 时为 0 —— **这一条决定了
    逼近时该往哪一侧绕**, 见 ``sweeper4.beam_swing_point``。

表示 (与问题3 一致的三层结构, 只做最小扩展):
  * 位置: mode 0 占用网格 / mode 1 沿射线样本 / mode 2 Gauss-Newton 点估计 —— 全部复用
    ``belief.ChannelTrack``;
  * **源型**: 用两个似然累加器 ``logL_omni`` / ``logL_dir`` 表示 (比值即 λ 的后验几率);
  * **定向方向**: 36 个 10 度分箱上的后验权重, 先验均匀。

因为 psi 只以 "源->检测点方位" 的形式进入似然, 这里采用平均场式分解
q(pos, psi) = q(pos)·q(psi)。两个直接后果:
  * 检测概率 ``P(收到|q) = λ·Σ_b w_b·Γ(psi_b - beta)·K(d) + (1-λ)·K(d)``,
    其中 ``Γ = 1/2 + min(|delta|,90)/360`` (引理 1); λ=0 时逐位退化为问题3;
  * psi 的单向后验会**自动抑制镜像解**: 真解与镜像解要求 psi 相差 180 度, 而 w 只在
    真解一侧有质量。实现方式是给每个观测点维护一个位置乘子 ``cov_scale``
    (mode 2) 或直接压低网格 (mode 0/1) —— 见 ``_obs_consistency``。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .belief import Belief, ChannelTrack, _line_intersection
from ..consts import N_CHANNELS, RADIUS_MAX_M, RADIUS_MIN_M

N_BINS = 36
BINS_RAD = (np.arange(N_BINS) + 0.5) * (2.0 * math.pi / N_BINS)
BINS_DEG = np.degrees(BINS_RAD)
# 方向指示函数查表: LUT[b, a] = 1[ |wrap(psi_b - (a 度))| <= 90 ]。
# 热路径 (格点 × 分箱) 上原本要做 arctan2/sin/cos, 换成整数取整查表后只做一次索引,
# 实测把 ``detect_prob_grid`` 的开销压掉一个数量级 (原本占单局 54%).
_DIR_LUT = ((np.abs(((BINS_DEG[:, None] - np.arange(360)[None, :] + 180.0) % 360.0)
                    - 180.0)) <= 90.0 + 1e-9).astype(np.float64)
# 软化用的小常数: 一次观测把 "与观测矛盾" 的分箱按 EPS 的幂次降权, 而不是置零 ——
# 硬截断在观测互相矛盾时会把整个后验清空 (实测必然退化), 幂次软化永远可逆。
# 取 1e-4: 5 次一致观测后矛盾分箱的权重比降到 1e-20 量级 (等价于硬排除),
# 而单次边界观测只降 4 个数量级 (留有余量, 不会因一次误判锁死)。
MATH_EPS = 1e-4


def wrap_deg(a: float) -> float:
    """把角度差折算到 (-180, 180]。"""
    return (a + 180.0) % 360.0 - 180.0


def dir_factor(delta_rad: np.ndarray) -> np.ndarray:
    """psi 已知时的方向因子: Γ(psi-phi) = 1[ |wrap(psi-phi)| <= 90 ]。

    这是**确定性**的: 波束中心 psi 给定、源在方位 phi, 要么在覆盖角内 (收到), 要么
    不在 (收不到)。可靠性来自 sigma 与源位置无关, 所以式子里不出现探测概率。
    """
    return (np.abs(delta_rad) <= 0.5 * math.pi + 1e-12).astype(np.float64)


def detect_prob_dir(d2: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                    weights: np.ndarray) -> np.ndarray:
    """一次算完所有格点的 "定向假设下的检测概率" Γ̄·K(d)。

    d2 = 格点到检测点的距离平方; (ux,uy) = **格点(源) -> 检测点** 的单位向量。
    """
    if weights.size == 0 or d2.size == 0:
        return np.zeros(d2.shape, dtype=np.float64)
    phi = np.arctan2(uy, ux)[:, None]                     # (N,1) 源->检测点 方位
    delta = np.abs(np.arctan2(np.sin(BINS_RAD[None, :] - phi),
                              np.cos(BINS_RAD[None, :] - phi)))
    gain = dir_factor(delta) @ weights
    d = np.sqrt(np.maximum(d2, 0.0))
    kern = np.clip((RADIUS_MAX_M - d) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0, 1.0)
    return gain * kern


def dir_gap(p: Tuple[float, float], meas_pts, r_min: float = RADIUS_MIN_M
            ) -> Tuple[float, float]:
    """点 p 的**角向覆盖缺口** (问题4 的 "保证发现" 判据)。

    对 "源在 p" 这一假设, 检查所有 1000 m 内检测点相对**源**的方位
    ``phi_i = atan2(m_i - p)``: 只有 "波束朝 phi_i 一侧" (|wrap(psi-phi_i)|<=90)
    才收得到, 因此可发现 ⟺ 这些方位不落在任何开半圆内 (最大空隙 <= 180 度)。
    返回 ``(最大空隙/度, 1000 m 内检测点数)``。
    """
    angs = []
    for (mx, my) in meas_pts:
        if math.hypot(mx - p[0], my - p[1]) <= r_min:
            angs.append(math.degrees(math.atan2(my - p[1], mx - p[0])) % 360.0)
    if len(angs) < 2:
        return 360.0, len(angs)
    angs.sort()
    gap = 0.0
    for i in range(len(angs)):
        gap = max(gap, (angs[(i + 1) % len(angs)] - angs[i]) % 360.0)
    return gap, len(angs)


class ChannelTrack4(ChannelTrack):
    """单频道信念 (问题4): 问题3 的三种位置表示 + 源型 + 定向方向。"""

    def __init__(self, idx: int, grid: np.ndarray, rng=None) -> None:
        super().__init__(idx, grid, rng)
        self.dir_w = np.full(N_BINS, 1.0 / N_BINS, dtype=np.float64)
        self.logL_omni = 0.0                  # Σ log P(obs | 全向)
        self.logL_dir = 0.0                   # Σ log P(obs | 定向, psi 已边缘化)
        self.n_shadow = 0                     # "在范围内却收不到" 次数 = 定向的硬证据
        self.n_inrange = 0
        self.n_det = 0
        self.cov_scale = 1.0                  # mode 2: 与观测不一致的乘子 (抑制镜像解)

    # ---------------- 源型 ----------------
    @property
    def lam(self) -> float:
        """P(源为定向源 | 观测) —— 由两个似然累加器直接给出。"""
        m = max(self.logL_omni, self.logL_dir)
        a = math.exp(self.logL_omni - m)
        b = math.exp(self.logL_dir - m)
        return float(b / (a + b)) if (a + b) > 0 else 0.5

    def add_loglik(self, lo: float, ld: float) -> None:
        self.logL_omni += lo
        self.logL_dir += ld

    def add_evidence(self, p_dist: float, p_dir: float) -> None:
        """一次 no_signal 对两个源型假设的似然 (在**距离**边缘化之后)。

        全向假设: P(没收到) = 1 − K(d)      (距离上就不该收到 ⇒ 与源型无关)
        定向假设: P(没收到) = 1 − Γ̄·K(d)    (还可能是被波束挡住)
        两者之差正是 "有没有可能是被挡住", 因此比值就是源型的证据。

        ``p_dist`` / ``p_dir`` 必须是**距离边缘化**后的检测概率 (格点点积或高斯积分),
        不能用单点核函数近似 —— 后者在源已定位时会给出 O(1) 的错误证据。
        """
        p_dist = min(max(float(p_dist), 1e-9), 1.0 - 1e-9)
        p_dir = min(max(float(p_dir), 1e-9), 1.0 - 1e-9)
        self.add_loglik(math.log(1.0 - p_dist), math.log(1.0 - p_dir))

    # ---------------- psi 的后验 ----------------
    def dir_interval(self, mass: float = 0.9) -> Tuple[float, float, float]:
        """psi 后验的 ``(中心角, 半宽, 覆盖质量)`` (度) —— 用循环均值避免跨越 0 度出错。"""
        w = self.dir_w
        if w.sum() <= 1e-12:
            return 0.0, 180.0, 0.0
        w = w / w.sum()
        z = complex(float(np.dot(w, np.cos(BINS_RAD))), float(np.dot(w, np.sin(BINS_RAD))))
        if abs(z) < 1e-9:
            return 0.0, 180.0, 0.0
        center = math.degrees(math.atan2(z.imag, z.real))
        # 在中心附近累积到 mass
        order = sorted(range(N_BINS), key=lambda i: abs(wrap_deg(BINS_DEG[i] - center)))
        acc = 0.0
        half = 180.0
        for i in order:
            acc += w[i]
            half = abs(wrap_deg(BINS_DEG[i] - center))
            if acc >= mass:
                break
        return center, half, acc

    def dir_mode(self) -> float:
        return float(BINS_DEG[int(np.argmax(self.dir_w))])

    def _dir_gain(self, phi_deg: float) -> float:
        """``Γ̄(phi) = Σ_ψ w_ψ · 1[|wrap(ψ-phi)|<=90]`` 的单点求值。

        用整数取整 + 查表实现, 避免在格点级别的热路径上做三角函数。
        """
        return float(self.dir_w @ _DIR_LUT[:, int(round(phi_deg)) % 360])

    def detect_prob_grid(self, d: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                         kernel: np.ndarray) -> np.ndarray:
        """mode 0 的**向量化**检测概率 (给 sweeper 的 "所有频道所有点一次算完" 用)。

        d = 检测点到各格点的距离, (ux,uy) = 格点(源)->检测点 的单位向量,
        kernel = P(R >= d) = clamp((1500-d)/500)。返回逐格点的加权检测概率
        ``(1-λ)·K + λ·Γ̄·K``。
        """
        lam = self.lam
        if lam <= 1e-6 or self.dir_w.size == 0:
            return kernel.astype(np.float64)
        # ``Γ̄`` 的向量化批量求值 Γ̄(phi) = Σ_ψ w_ψ·1[|wrap(ψ-phi)|<=90]:
        # 把方位取整成度数再查 ``_DIR_LUT``, 避免在格点热路径上做三角函数。
        idx = np.mod(np.rint(np.degrees(np.arctan2(uy, ux))).astype(np.int64), 360)
        gain = self.dir_w @ _DIR_LUT[:, idx]
        return ((1.0 - lam) + lam * gain) * kernel

    def _renorm_dir(self) -> None:
        s = float(self.dir_w.sum())
        if s > 1e-12:
            self.dir_w = self.dir_w / s
        else:
            self.dir_w = np.full(N_BINS, 1.0 / N_BINS)

    def constrain_dir(self, phi_deg: float, half_deg: float = 90.0,
                      strength: float = 1.0) -> None:
        """收到信号: psi 必须落在 [phi-half, phi+half] 内, 用**软**加权实现。

        ``phi`` 是**源 -> 检测点** 的方位角 (模拟器 ``in_coverage`` 用的那一支),
        等价于 "检测点 -> 源" 的示向度读数加 180 度。

        为什么是软的: 位置本身有不确定度, 硬截断在观测互相矛盾时会把后验直接清空
        (实测 3 条 no_signal 后硬截断必然触发退化分支)。这里用 "psi 落在这个半圆内"
        与 "落在半圆外" 的似然比做幂次加权, 信息量等价而数值上永远可逆。
        """
        d = np.abs(((BINS_DEG - phi_deg + 180.0) % 360.0) - 180.0)
        ratio = np.where(d <= half_deg + 1e-9, 1.0, MATH_EPS)
        self._scale_dir(ratio, strength)

    def exclude_dir(self, phi_deg: float, half_deg: float = 90.0,
                    strength: float = 1.0) -> None:
        """未收到 (且在有效半径内): psi **不该**落在 [phi-half, phi+half] 内。

        这就是 "被波束挡住" 的硬信息 —— 问题4 相对问题3 唯一新增的信息通道。
        """
        d = np.abs(((BINS_DEG - phi_deg + 180.0) % 360.0) - 180.0)
        ratio = np.where(d > half_deg, 1.0, MATH_EPS)
        self._scale_dir(ratio, strength)

    def _scale_dir(self, ratio: np.ndarray, strength: float) -> None:
        self.dir_w = self.dir_w * np.power(ratio, float(strength))
        if float(self.dir_w.sum()) <= 1e-300:
            self.dir_w = np.full(N_BINS, 1.0 / N_BINS)
            return
        self.dir_w = np.maximum(self.dir_w, 0.0)
        self._renorm_dir()

    # ---------------- 检测概率 ----------------
    def detect_prob_omni(self, q: Tuple[float, float], grid_pts: np.ndarray,
                         kernel: Optional[np.ndarray] = None) -> float:
        """只算**全向假设**下的检测概率 = 问题3 的口径 (父类实现)。"""
        return ChannelTrack.detect_prob(self, q, grid_pts, kernel)

    def detect_prob_dir_only(self, q: Tuple[float, float],
                             grid_pts: np.ndarray) -> float:
        """只算**定向假设**下的检测概率 (psi 已边缘化)。"""
        if self.mode == 2 and self.est is not None and self.cov is not None:
            return self._gauss_detect_prob(q, omni=False)
        if self.mode == 1 and self.ray_x is not None:
            dx = q[0] - self.ray_x[:, 0]                 # 源 -> 检测点
            dy = q[1] - self.ray_x[:, 1]
            d2 = dx * dx + dy * dy
            nz = np.maximum(np.sqrt(d2), 1e-9)
            return float(np.dot(self.ray_w, detect_prob_dir(d2, dx / nz, dy / nz, self.dir_w)))
        if self.mode == 0 and self.grid is not None:
            dx = q[0] - grid_pts[:, 0]
            dy = q[1] - grid_pts[:, 1]
            d2 = dx * dx + dy * dy
            nz = np.maximum(np.sqrt(d2), 1e-9)
            return float(np.dot(self.grid, detect_prob_dir(d2, dx / nz, dy / nz, self.dir_w)))
        return 0.0

    def detect_prob(self, q: Tuple[float, float], grid_pts: np.ndarray,
                    kernel: Optional[np.ndarray] = None) -> float:
        """P(在 q 处能收到该频道信号 | 源存在且未清除) —— 按源型混合。"""
        lam = self.lam
        if lam <= 1e-6:
            return self.detect_prob_omni(q, grid_pts, kernel)     # 逐位退化为问题3
        po = self.detect_prob_omni(q, grid_pts, kernel)
        if lam >= 1.0 - 1e-9:
            return self.detect_prob_dir_only(q, grid_pts)
        return float((1.0 - lam) * po + lam * self.detect_prob_dir_only(q, grid_pts))

    def _dir_gauss(self, q: Tuple[float, float], n_grid: int = 9) -> float:
        """mode 2 的定向分量 P(收到) = Σ_ψ w_ψ · E_pos[ Γ(ψ-φ(pos)) · K(|pos-q|) ]。

        位置用 ``N((est), cov)`` 的三点式高斯求积近似 (与父类 ``_gauss_detect_prob``
        同一套做法), 但**保留方向依赖**: 源在地图上哪个方位、以及那里的距离核,
        都要按位置协方差平均。原先用 "距离只看到 est" 的近似会在源已定位时
        给出与几何完全不符的概率 (实测背面仍报 0.75, 而真值恒为 0)。
        """
        if self.est is None or self.cov is None:
            return 0.0
        cov = np.asarray(self.cov, dtype=np.float64)
        if not np.all(np.isfinite(cov)):
            return 0.0
        w, V = np.linalg.eigh(cov)
        w = np.maximum(w, 0.0)
        sig = np.sqrt(w)
        pts = [(0.0, 0.0)]
        for k in range(2):
            if sig[k] > 1e-6:
                pts.append((V[0, k] * sig[k], V[1, k] * sig[k]))
        wt = [1.0 / len(pts)] * len(pts)
        acc = 0.0
        for (dx, dy), wq in zip(pts, wt):
            sx, sy = self.est[0] + dx, self.est[1] + dy
            ddx, ddy = q[0] - sx, q[1] - sy
            d = math.hypot(ddx, ddy)
            kern = min(max((RADIUS_MAX_M - d) / (RADIUS_MAX_M - RADIUS_MIN_M), 0.0), 1.0)
            if kern <= 0.0:
                continue
            phi = math.degrees(math.atan2(ddy, ddx))
            gain = self._dir_gain(phi)
            acc += wq * gain * kern
        return acc

    def _gauss_detect_prob(self, q: Tuple[float, float], omni: bool = True) -> float:  # type: ignore[override]
        """mode 2 的检测概率。``omni=False`` 时只算定向分量 (保留方向依赖)。

        全向分量与父类 (:meth:`ChannelTrack._gauss_detect_prob`) **逐行等价**,
        直接复用 —— 这里只多一个 "要不要换成定向分量" 的分派。
        """
        if self.est is None or self.cov is None:
            return 0.0
        if not omni:
            return self._dir_gauss(q)
        return ChannelTrack._gauss_detect_prob(self, q)

    def _obs_consistency(self, est: Optional[Tuple[float, float]] = None) -> float:
        """位置与 "已收到的方向读数" 的一致性权重 Γ̄ (用于识别镜像解)。

        真解与镜像解对同一读数给出的 "源->检测点" 方位相差 180 度; 当前 psi 后验只在
        真解一侧有质量, 于是镜像解处的 Γ̄ 很小。取所有方向观测点上 Γ̄ 的最小值。
        """
        est = self.est if est is None else est
        if not self.obs or est is None or self.mode == 0:
            return 1.0
        phis = np.array([math.degrees(math.atan2(my - est[1], mx - est[0]))
                         for (mx, my, _s) in self.obs])
        idx = np.mod(np.rint(phis).astype(np.int64), 360)
        return float(np.min(self.dir_w @ _DIR_LUT[:, idx]))

    def refit(self) -> None:  # type: ignore[override]
        """Gauss-Newton 定位, 但**用 psi 后验挑初值**来避开镜像解。

        一条 ±1 度的示向度只给出 "源在某条射线上", 两条射线就有两个解:
        真解与跨基线反射的镜像解。问题3 里无所谓 (第二次检测会把错的那个压掉);
        问题4 里 psi 的后验恰好带角度信息 —— 真解要求 psi ≈ phi_true, 镜像解要求
        psi ≈ phi_true ± 180, 于是只要在候选交点里挑 Γ̄ 最大的那个做初值即可。
        实测把 (2+ 条示向度) 的定位误差 p90 从 ~145 m 压到 ~15 m。
        """
        obs = self.obs
        if len(obs) >= 2 and self.mode != 0:
            cands: List[Tuple[float, float]] = []
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    p = _line_intersection(obs[i], obs[j])
                    if p is not None and math.hypot(p[0], p[1]) < 3600.0:
                        cands.append(p)
            if cands:
                best = max(cands, key=lambda p: self._obs_consistency(p))
                self.est = best
                # 用选中的初值重建协方差 (与父类同样的解析式)
                Info = self._info_matrix(obs, best[0], best[1])
                det = Info[0, 0] * Info[1, 1] - Info[0, 1] * Info[1, 0]
                if abs(det) > 1e-26:
                    cov = np.array([[Info[1, 1], -Info[0, 1]],
                                    [-Info[1, 0], Info[0, 0]]]) / det
                    # 一致性越差, 协方差越大 (策略据此判断 "还没定好位"); 双向夹紧
                    k = max(self._obs_consistency(best), 0.2)
                    cov = cov * float(min(1.0 / k, 5.0))
                    # 近共线基线会让解析协方差发散; 一旦出 NaN/Inf, 后续所有比较都
                    # 失去意义 (实测 numpy overflow 之后整局估计全废) —— 直接弃用。
                    if np.all(np.isfinite(cov)):
                        self.cov = cov
                    else:
                        self.cov = np.eye(2) * (900.0 ** 2)
                # 再做几轮 GN 精修 (从该初值出发, 与父类同一目标函数)
                super().refit()
                if self.cov is None or not np.all(np.isfinite(self.cov)):
                    self.cov = np.eye(2) * (900.0 ** 2)
                elif max(abs(float(self.cov[0, 0])), abs(float(self.cov[1, 1]))) > 1e9:
                    self.cov = np.eye(2) * (900.0 ** 2)

    # ---------------- 观测更新 ----------------
    def add_direction(self, mx: float, my: float, svd: float, t: float) -> None:
        super().add_direction(mx, my, svd, t)
        # 收到方向 => 源->检测点 的方位 = 读数 + 180 (见 constrain_dir 的说明)
        self.constrain_dir((float(svd) + 180.0) % 360.0)
        self.n_inrange += 1
        self.n_det += 1
        self.cov_scale = 1.0

    def add_near(self, mx: float, my: float, t: float) -> None:
        super().add_near(mx, my, t)
        self.n_inrange += 1
        self.n_det += 1
        self.cov_scale = 1.0

    def apply_no_signal_typed(self, q: Tuple[float, float], p_dist: float,
                              p_dir: float, grid_pts, kernel) -> None:
        """no_signal 的完整更新。

        ``p_dist`` 是**只含距离**的检测概率 ``K(d)`` (与源型无关, 与问题3 同一个量);
        ``p_dir = Γ̄·K(d)`` 是定向假设下的检测概率。两者在似然里分工明确:

          * **位置后验**只能用 ``p_dist``: 没收到只能说明 "距离上本就不该收到"
            或者 "被波束挡住", 后者对位置没有直接约束 (约束的是 psi)。
            原先用混合似然同时更新位置, 会把 "被挡住" 误读成 "源不在这儿",
            把 π_c 一起打下去 —— 实测直接导致约 2% 的源被永久放弃。
          * **psi 后验**用方向本身: "本该收到却没收到" ⇒ 波束背对 (``exclude_dir``)。
          * **源型 λ** 用两者之比: 比值显著小于 1 才说明 "只能是被挡住"。
        """
        if p_dist >= 0.35:
            self.n_shadow += 1
            mu = (self.est if (self.est is not None and self.mode != 0)
                  else self._grid_mean(grid_pts))
            if mu is not None:
                phi = math.degrees(math.atan2(q[1] - mu[1], q[0] - mu[0]))
                # 强度随 "本该收到的把握" 增加: 几乎必然该收到却没收到, 才是有力的
                # "波束背对" 证据; 在有效半径边缘的 no_signal 几乎不提供角度信息。
                self.exclude_dir(phi, strength=min(max(p_dist, 0.0), 1.0))
        self.add_evidence(p_dist, p_dir)

    def _grid_mean(self, grid_pts) -> Optional[Tuple[float, float]]:
        if self.grid is None or self.mode != 0:
            if self.est is not None:
                return self.est
            return None
        s = float(self.grid.sum())
        if s <= 1e-12:
            return None
        return (float(np.dot(self.grid, grid_pts[:, 0]) / s),
                float(np.dot(self.grid, grid_pts[:, 1]) / s))


class Belief4(Belief):
    """20 个频道的联合信念 (问题4)。"""

    def __init__(self, prior_pi: float = 13.0 / 20.0, prior_lam: float = 0.5,
                 seed: int = 20260911) -> None:
        super().__init__(prior_pi=prior_pi, seed=seed)
        base = self.tracks
        pl = max(min(float(prior_lam), 1.0 - 1e-9), 1e-9)
        self.tracks: List[ChannelTrack4] = [  # type: ignore[assignment]
            ChannelTrack4(i, base[i].grid, base[i].rng) for i in range(N_CHANNELS)
        ]
        for t in self.tracks:
            t.pi = prior_pi
            t.grid_pts = self.pts
            # 先验几率 = λ/(1-λ) ⇒ 用两个似然累加器的初值表达
            t.logL_omni = math.log(1.0 - pl)
            t.logL_dir = math.log(pl)
        self.prior_lam = pl

    # ---------------- 观测更新 ----------------
    def on_measure(self, mx: float, my: float, channel: int, kind: str,
                   svd: Optional[float], t: float) -> None:
        tr = self.tracks[channel - 1]
        if kind == "direction":
            tr.add_direction(mx, my, float(svd), t)
            return
        if kind == "near":
            tr.add_near(mx, my, t)
            return
        # ---- no_signal ----
        dg = np.hypot(self.pts[:, 0] - mx, self.pts[:, 1] - my)
        kernel = np.clip((dg - RADIUS_MIN_M) / (RADIUS_MAX_M - RADIUS_MIN_M),
                         0.0, 1.0).astype(np.float32)
        if tr.mode == 0 and tr.grid is not None:
            p_dist = float(np.dot(tr.grid, kernel))
            d = np.maximum(dg, 1e-9)
            p_dir = float(np.dot(tr.grid, detect_prob_dir(
                dg * dg, (mx - self.pts[:, 0]) / d, (my - self.pts[:, 1]) / d, tr.dir_w)))
        else:
            p_dist = tr.detect_prob_omni((mx, my), self.pts, None)
            p_dir = tr.detect_prob_dir_only((mx, my), self.pts)
        tr.apply_no_signal_typed((mx, my), p_dist, p_dir, self.pts, kernel)
        # 位置后验吃**按源型混合**后的似然
        #   P(没收到 | 源在格点 i) = 1 - [(1-λ)·K(d_i) + λ·Γ̄_i·K(d_i)]
        # λ 小 (可能是全向) 时 ≈ 1-K, 与问题3 逐位一致; λ 大 (已确认定向) 时
        # 只惩罚 "正对波束却收不到" 的格点, 不会把整片区域一起抹掉。
        lam = tr.lam
        if lam <= 1e-6:
            L = max(1.0 - p_dist, 1e-9)
        else:
            L = max(1.0 - ((1.0 - lam) * p_dist + lam * p_dir), 1e-9)
        ChannelTrack.apply_no_signal(tr, (mx, my), L, self.pts, kernel)
        if tr.mode == 2 and tr.cov is not None:
            k = tr._obs_consistency()
            # k 很小 (典型: GN 收敛到镜像解, 该处要求的 psi 与后验差 180 度)
            # ⇒ 位置与 psi 后验不一致 ⇒ 抬高协方差, 避免策略把 "定位已完成" 当既成事实。
            # 必须双向夹紧: k 触到 1e-3 时 1/k 会把协方差放大 1000 倍, 接着 GN 的
            # 信息矩阵运算就溢出 (实测 numpy overflow), 反向污染后续所有估计。
            k = max(k, 0.2)
            tr.cov = tr.cov * float(min(1.0 / k, 5.0))
