"""问题3 求解器: 贝叶斯信念 + 确定性清除规划器。

``sweeper``: 清除即探测 —— 就近清除已知源, 每次清除后原地补测其它频道,
无已知目标时按信息净收益专程探测, 最后按集合覆盖收尾。

设计动机与全部实验数据见同目录下的 ``README.md``。

包结构按职责分层, 依赖方向单向 (箭头指向被依赖方)::

    deploy.py / eval.py / e2e_test.py / oracle_route.py   入口 (命令行)
                |
                v
    sweeper.py (规划器)
                |
                v
    ops.py (动作原语)  belief.py (信念)
                |
                v
    world.py (进程内环境)  ../consts.py (物理常量)

脚本一律以 ``python -m robotdog.solver.<模块>`` 从仓库根目录运行, 因此不需要各自
推算仓库路径或改写 ``sys.path``。
"""

from __future__ import annotations

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(PACKAGE_DIR))

__all__ = ["PACKAGE_DIR", "PROJECT_ROOT"]
