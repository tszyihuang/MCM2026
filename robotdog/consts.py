"""题目物理常量 (严格取自附件1 与附件2), 供机器狗程序与 solver 求解器共用.

单一来源: 机器狗程序 (``client.py`` / ``solver/deploy.py``)、求解器 (``solver/``)
与本目录下的进程内环境 (``robotdog/solver/world.py``) 都从这里取值, 不再各自维护副本。
模拟器 (``simulator/core.py``) 刻意**不**共享本模块 —— 它是对题目的独立复现,
两份常量各自独立才能让逐位等价性测试 (``tests/test_solver_equiv.py``) 有意义。
"""

# --- 区域与运动 ---
ARENA_RADIUS_M = 1800.0          # 目标区域半径 (附件1 1.1)
SPEED_MPS = 5.0                  # 机器狗移动速度 (附件2 4.2)

# --- 检测与清除 ---
NEAR_RADIUS_M = 5.0              # 近距阈值: <= 5 m 返回 near (附件1 2.4)
CLEAR_RADIUS_M = 20.0            # 清除半径 (附件1 2.4)
DETECT_DURATION_S = 5.0          # 每次检测 5 s (附件2 4.3)
CHANNEL_SWITCH_S = 1.0           # 每次切换频道 1 s, 仅 /measure 计 (附件2 4.3)
CLEAR_LOCATE_S = 3.0             # /clear 未发现 3 s (附件2 4.4)
CLEAR_FIRE_S = 2.0               # /clear 命中 5 s = 定位 3 s + 清除 2 s (附件2 4.4)

# --- 观测模型 ---
SVD_ERROR_DEG = 1.0              # 示向度误差上界 ±1 deg (附件1 2.3)
RADIUS_MIN_M = 1000.0            # 有效接收半径下界 (附件1 2.1)
RADIUS_MAX_M = 1500.0            # 有效接收半径上界 (附件1 2.1)

# --- 频道与源 ---
CHANNELS = tuple(range(1, 21))   # 频道 1..20 (附件1 1.3)
N_CHANNELS = 20
N_SOURCES_MIN, N_SOURCES_MAX = 10, 16

# --- 时间上限 ---
MAX_VIRTUAL_S = 360_000.0        # 虚拟世界活动时长上限 100 h (仅防死循环)
MAX_REAL_S = 1200.0              # 现实程序运行时间上限 20 min (真正的约束)
