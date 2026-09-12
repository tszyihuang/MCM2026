"""无线电干扰源环境模拟器 (2026 高教社杯全国大学生数学建模竞赛 B 题自建复现版).

模块划分:
  * core.py    —— 物理规则、虚拟计时、案例生成、报告与行为日志
  * server.py  —— HTTP+JSON 通信层 (/enter /measure /clear /exit) 与状态码语义
  * session.py —— 登录/时间校验、四个测试模块、25 分钟窗口、日志队列与导出
  * gui.py     —— 模拟器界面 (测试控制台 + 日志列表)
"""

__all__ = ["core", "server", "session", "gui"]
__version__ = "1.0.0"
