"""工程与验收工具 (不是模拟器的一部分, 也不被策略导入).

自检与审计
  * ``smoke_test.py``        一键冒烟 (协议 / 计时 / 并发 / 保密 / 真实进程端到端)
  * ``spec_audit.py``        独立赛题符合性审计 (不依赖 ``tests/`` 子包的既有用例)
  * ``quick_check.py``       手工冒烟: 对运行中的模拟器跑一遍附件示例指令序列

评测与验收 (问题3)
  * ``candidates.py``        策略评测 (进程内口径 ``knows_total=True``)
  * ``eval_deploy.py``       策略评测 (**部署口径** ``knows_total=False``, 正式测试口径)
  * ``verify_candidate.py``  交付验收: 确定性 + 双口径成绩
  * ``profile_run.py``       阶段剖析: 每个阶段的动作数/秒/移动 (``--deploy`` 切口径)
  * ``move_bound.py``        移动下界: 真值精确 TSP vs 策略实际移动
  * ``oracle.py``            清除个数上界估计 (虚拟时间口径, 见 docs/验收报告.md 4.3)

评测与调参 (问题4: 含定向源)
  * ``eval_q4.py``           问题4 策略评测 (进程内 / ``--deploy`` 部署口径)
  * ``tune_q4.py``           问题4 参数扫描 (多旋钮网格 + 排序, 直接给 s/源)
  * ``probes/q4_cover.py``   覆盖几何审计: 任意点集的覆盖率与巡游长度
  * ``probes/q4_missed.py``  失败归因: 漏掉的源的真值与策略所见 (旧版规划器)
  * ``probes/q4_profile.py`` 阶段剖析: 巡游/交会/残余 三段的秒数与移动 (旧版规划器)
  * ``probes/q4_opt1_parity.py`` 移植保真度: 当前策略 vs Desktop/opt1 沙盒单文件策略
  * ``probes/q4_make_figures.py`` 问题4 论文插图

批次与性能
  * ``run_batch.py``         演练 / 正式测试批处理 (导出日志与统计表, 支持多核并行)
  * ``bench_cases.py``       小批量多核对局基准 (``--serve`` 常驻模式, 反复回归用)
  * ``perf_bench.py``        性能基准 (微基准 + 多核对局 + 行为指纹对比)
  * ``decision_fingerprint.py`` 逐动作决策指纹 (重构安全网)

手工调试
  * ``run_sim.py``           启动一个会话并保持运行

测试套件 (``tests/`` 子包, 只用标准库 unittest)
  * ``tests/harness.py``           测试夹具 (可注入假时钟的模拟器实例 + HTTP 客户端)
  * ``tests/test_protocol.py``     通信协议一致性 (状态码/字段集/幂等/并发 409/时间预算)
  * ``tests/test_physics.py``      物理规则与虚拟计时 (附件1 表2、附件2 第10节)
  * ``tests/test_scenario.py``     四个测试模块 / 倒计时 / 日志导出与加密 / 真值屏蔽
  * ``tests/test_acceptance.py``   真实进程端到端验收 (问题3 演练 + 正式测试)
  * ``tests/test_solver_equiv.py`` 求解器与 simulator.core 的逐动作物理等价性
  * ``tests/test_q4.py``           问题4: 定向源规则 / 覆盖几何 / 端到端指标
"""
