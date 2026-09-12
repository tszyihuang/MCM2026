"""机器狗程序: 问题3 的自动搜索定位与清除.

模块划分:
  * consts.py     —— 题目物理常量 (与附件1 / 附件2 逐条对应)
  * geometry.py   —— 方位角 / 交会 / 加权残差拟合等几何工具
  * client.py     —— RobotClient: HTTP+JSON 客户端 (重试 / 幂等 / 现实预算 / 行为日志)
  * solver/       —— 求解器 ``sweeper``, 另见其 README

只依赖 Python 标准库 (Python 3.9+); ``solver`` 子包另需 numpy。
"""

from .client import RobotClient

__all__ = ["RobotClient"]
