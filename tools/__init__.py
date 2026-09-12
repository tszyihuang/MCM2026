"""工程与验收工具 (不是模拟器的一部分, 也不被策略导入).

自检与审计
  * ``smoke_test.py``        一键冒烟 (协议 / 计时 / 并发 / 保密 / 真实进程端到端)
  * ``spec_audit.py``        独立赛题符合性审计 (不依赖 tests/ 既有用例)
  * ``quick_check.py``       手工冒烟: 对运行中的模拟器跑一遍附件示例指令序列

评测与验收 (问题3)
  * ``candidates.py``        策略评测 (进程内口径 ``knows_total=True``)
  * ``eval_deploy.py``       策略评测 (**部署口径** ``knows_total=False``, 正式测试口径)
  * ``verify_candidate.py``  交付验收: 确定性 + 双口径成绩
  * ``profile_run.py``       阶段剖析: 每个阶段的动作数/秒/移动 (``--deploy`` 切口径)
  * ``move_bound.py``        移动下界: 真值精确 TSP vs 策略实际移动
  * ``oracle.py``            清除个数上界估计 (虚拟时间口径, 见 docs/验收报告.md 4.3)

批次与性能
  * ``run_batch.py``         演练 / 正式测试批处理 (导出日志与统计表, 支持多核并行)
  * ``bench_cases.py``       小批量多核对局基准 (``--serve`` 常驻模式, 反复回归用)
  * ``perf_bench.py``        性能基准 (微基准 + 多核对局 + 行为指纹对比)
  * ``decision_fingerprint.py`` 逐动作决策指纹 (重构安全网)

手工调试
  * ``run_sim.py``           启动一个会话并保持运行
"""
