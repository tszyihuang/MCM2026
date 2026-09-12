# 2026 高教社杯 B 题 · 无线电干扰源环境模拟器与机器狗程序

本仓库是 **B 题「无线电干扰源的快速自动定位与清除」** 的完整自研工具链，分四层：

| 层 | 目录 | 说明 |
| --- | --- | --- |
| **环境** | `simulator/` | 严格按题目正文、附件1《模拟器使用说明》、附件2《模拟器通信接口说明及编程指南》实现的自建模拟器，可本地复现四个测试模块与全部协议细节 |
| **策略** | `robotdog/solver/` | 问题3 求解器 **`sweeper`**；问题4 求解器 **`sweeper4`** |
| **通信** | `robotdog/` | `RobotClient`：HTTP+JSON 客户端（重试 / 幂等 / 现实预算 / 行为日志） |
| **验证** | `tests/`、`tools/`、`docs/` | 自动化用例、冒烟测试、独立赛题符合性审计、验收与实测报告 |

> **每题只有一条策略**：问题3 = `sweeper`，问题4 = `sweeper4`。
> 历史基线与已淘汰的路线已从工程中移除，代码与文档不再区分"版本"或"路线"。

---

> [!WARNING]
> ### 计时口径：虚拟 100 小时，现实至多 20 分钟（最容易搞错的一点，先读这里）
>
> **虚拟环境里有 100 小时的虚拟时间，而策略程序自己的运行时间至多 20 分钟。**
> 这是两个**相互独立**的上限，不是同一个数字的两种说法：
>
> | | 上限 | `/enter` 返回的字段 | 由什么决定 |
> | --- | --- | --- | --- |
> | **虚拟世界活动时长** | **100 小时 = 360000 s** | `max_virtual_duration_s` | 机器狗**移动 / 检测 / 清除**累计消耗的虚拟秒数 |
> | **策略程序运行时间** | **20 分钟 = 1200 s** | `remaining_real_duration_s` | **现实墙钟**，从 `/enter` 成功那一刻开始计 |
>
> 关键含义：**现实 1 秒可以推进任意多虚拟秒**。附件2 §1.4 明确「每次检测的 5 秒只
> 增加虚拟时间，**不要求在现实中等待 5 秒**」，所以 20 分钟现实预算内可以累计出
> 数万虚拟秒。
>
> 三条推论，写程序前务必确认：
>
> 1. **先触发的几乎总是现实的 20 分钟**。100 小时虚拟上限只是防止程序死循环的兜底，
>    正常运行撞不到 —— 本项目每局只用掉 360000 s 的约 1%。
> 2. **不要写 `if 已用虚拟时间 > 1200: 退出`**。那是把虚拟时钟当成现实预算，
>    历史上正因此让程序只跑了 0.4 s 现实时间就自行收尾（见 `docs/验收报告.md` §6.1）。
> 3. 还存在第三个上限：**25 分钟测试窗口**（`window_left_s`，从倒计时结束算起）。
>    它与上述两者相互独立，三者取**最早到达者**结束测试。
>
> 预算判断一律用 `/enter` 返回的 `remaining_real_duration_s`（**现实**口径）。

## 目录结构

```
simulator/                模拟器本体 (纯标准库)
  core.py                 物理规则、虚拟计时、案例生成、报告与加密行为日志
  server.py               HTTP+JSON 通信层（状态码、幂等、并发串行闸门、严格校验）
  session.py              登录/时间校验、四个测试模块、25 分钟窗口、日志队列与导出
  gui.py                  模拟器界面（测试控制台 + 指令与反馈 + 日志列表）
  __main__.py             命令行入口: python -m simulator

robotdog/                 机器狗程序
  consts.py               题目物理常量（唯一来源）
  geometry.py             方位角 / 交会 / 加权残差拟合
  client.py               RobotClient: HTTP 客户端（重试 / 幂等 / 现实预算 / 行为日志）
  solver/                 问题3 / 问题4 求解器 (见 solver/README.md)  [需 numpy]
    world.py              进程内精确复刻环境 (真值与可观测信息严格分离; problem=3/4)
    belief.py             问题3 信念（网格 / 射线样本 / Gauss-Newton 三表示）
    belief4.py            问题4 信念（位置三表示 + 源型 λ + 定向方向 36 分箱）
    ops.py                动作原语: pursue / probe / 候选点 / 收益评估
    sweeper.py            【问题3】"清除即探测"规划器
    sweeper4.py           【问题4】集合覆盖补扫 + 锚点射线盲清 + 贴边补扫环
    deploy.py             机器狗程序部署入口: python -m robotdog.solver.deploy --problem 3|4
    eval.py               问题3 进程内评测入口: python -m robotdog.solver.eval
    e2e_test.py           真实进程 + HTTP 端到端验收 --problem 3|4
    oracle_route.py       已知路线下界探针（论文取数）

tests/                    自动化测试套件
  harness.py              测试夹具（可注入假时钟的模拟器实例 + HTTP 客户端）
  test_protocol.py        协议层：状态码/方法/请求体校验/幂等/并发 409/时间预算
  test_physics.py         物理规则与虚拟计时（附件1 表2、附件2 第 10 节的完整时序示例）
  test_scenario.py        四个测试模块、倒计时、日志导出与加密、正式测试真值屏蔽
  test_acceptance.py      真实进程端到端：问题3/问题4 演练 + 正式测试真值屏蔽与加密日志
  test_solver_equiv.py    求解器与 simulator.core 的逐动作物理等价性
  test_q4.py              问题4：案例生成逐位一致、动作等价、求解器不变量回归

tools/                    工程与验收工具（不被策略导入）
  smoke_test.py           一键冒烟测试（协议/计时/并发/保密/真实进程端到端）
  spec_audit.py           独立赛题符合性审计（不依赖 tests/ 既有用例；失败退出码非 0）
  candidates.py           批量评测（进程内口径 knows_total=True）
  eval_deploy.py          批量评测（**部署口径** knows_total=False，正式测试口径）
  eval_q4.py              问题4 统一评测 / 参数扫描（双口径 + 泛化集）
  profile_run.py          问题3 阶段剖析：每个阶段的动作数/秒/移动（--deploy 切口径）
  move_bound.py           移动下界：真值精确 TSP vs 策略实际移动
  verify_candidate.py     交付验收：确定性 + 双口径成绩
  pareto_front.py         帕累托前沿：清除比例 vs 平均定位清除时间 (问题3/4)
  make_paper_figures.py   论文插图生成 (中文, 全部可复现)
  run_batch.py            演练/正式测试批处理（导出日志与统计表，支持 --jobs 多核并行）
  bench_cases.py          小批量多核对局基准（--serve 常驻模式，反复回归用）
  perf_bench.py           模拟器性能基准（微基准 + 多核对局 + 行为指纹对比）
  decision_fingerprint.py 逐动作决策指纹（重构安全网）
  oracle.py               清除个数上界估计（虚拟时间口径，见 docs/验收报告.md §4.3）
  probes/                 论文取数用的一次性数值探针（不参与回归）
    verify_q4_theory4.py  问题4 理论引理的数值验证
  quick_check.py          手工冒烟: 对运行中的模拟器跑一遍附件示例指令序列
  run_sim.py              手工调试: 启动一个会话并保持运行

docs/                     文档 (见 docs/README.md 的完整索引)
  README.md               docs 索引
  **问题3解题思路.md**     问题3 的解题思路（简要）
  **问题4解法.md**         问题4 的机制、参数、实测与复现
  论文_问题4_集合覆盖路线.md  问题4 的论文成稿
  模拟器设计说明.md        模拟器逐条对照题目与附件的实现口径
  模拟器使用说明.md        命令行参数、与附件2 的对应关系、测试与验收命令
  验收报告.md             自动化测试结果、计时口径勘误、已修复问题清单
  figures/                论文插图 (11 张, 由 tools/make_paper_figures.py 生成)
  experiments/            实验存档索引（原始报告已丢失，见该目录 README.md）

data/                     运行时产物与对照证据（见 data/README.md）
  logs/                   行为日志留档：正式测试 3 组（加密 .log + 可读导出 .json 同名成对）
  robot_logs/             与上面对应的机器狗侧行为日志（JSONL）
  reports/                评测结果 JSON
附件/                     题目附件1、附件2 原文
```

## 快速开始

模拟器与部署程序只依赖 **Python 标准库（3.9+）**；
求解器的进程内评测需要 `numpy`（`pip install -r requirements.txt`）。

```powershell
# 1) 启动模拟器（机器狗接口 :2026，操作界面 :2027）
python -m simulator --team-id <参赛队号>

# 2) 浏览器打开界面，选择测试模块并确认开始；倒计时结束后运行机器狗程序
python -m robotdog.solver.deploy --team-id <参赛队号> --problem 3   # 问题3
python -m robotdog.solver.deploy --team-id <参赛队号> --problem 4   # 问题4
```

一键自检：

```powershell
python tools\smoke_test.py --full      # 冒烟 + 完整测试套件
python tools\spec_audit.py             # 独立赛题符合性审计 (115 项断言, 失败时退出码非 0)

# 问题3: 两种口径 (不可混比)
python tools\candidates.py  --modules robotdog.solver.sweeper --seeds 9500-9799 --jobs 6
python tools\eval_deploy.py --module  robotdog.solver.sweeper --seeds 9500-9799 --jobs 6

# 问题4: 两种口径
python tools\eval_q4.py --planners sweeper4 --seeds 9500-9799 --jobs 8
python tools\eval_q4.py --planners sweeper4 --seeds 9500-9799 --jobs 8 --deploy

# 剖析与下界
python tools\profile_run.py --module robotdog.solver.sweeper --seeds 9500-9529 --deploy
python tools\move_bound.py --seeds 9500-9539 --jobs 3

# 端到端 (真实进程 + HTTP), 问题3 / 问题4
python -m robotdog.solver.e2e_test --runs 3
python -m robotdog.solver.e2e_test --runs 3 --problem 4

# 论文插图 -> docs/figures/
python tools\make_paper_figures.py

# 正式测试 (真值屏蔽 + 加密日志)
python tools\run_batch.py --problem 3 --module formal --runs 3 --jobs 3 --countdown 0
```

## 策略成绩

留出案例集 **seed 9500-9799（300 局）**，策略确定性执行。

> ### ⚠ 先看口径，再看数字：进程内与部署**不可直接比较**
>
> `robotdog/solver/eval.py` 走**进程内**（`knows_total=True`：能看到真值总数，清完最后一个源
> 立刻停）；`robotdog/solver/deploy.py` 走**部署**（`knows_total=False`：**正式测试隐藏真值总数**，
> 只能靠信念判据收手，因此还要多花时间确认"真的清完了"）。
> **被打分的是部署口径**，它天生比进程内慢。

**问题3（`sweeper`）**

| 口径 | 清除比例 | 全清率 | 平均定位清除 | 移动 |
| --- | --- | --- | --- | --- |
| **部署（正式测试口径，被打分）** | **0.9982** | 293/300 | **259.7 s/源** | 12067 m |
| 进程内（知道总数，清完即停） | 0.9982 | 293/300 | 241.0 s/源 | 11153 m |

泛化（从未参与标定）：seed 8000-8099 部署口径 **0.9953 / 261.7 s/源**。
端到端（真实模拟器进程 + HTTP + 真值屏蔽）单局清除 **93%~100%**，
现实程序运行时间约 **0.1 s**（上限 1200 s）。**瓶颈不是时间，而是「还能不能找到剩下的源」**。

**问题4（`sweeper4`）**

| 档位 | 清除比例 | 全清率 | 平均定位清除 |
| --- | --- | --- | --- |
| **全清档（默认）** | **1.0000** | 300/300 | 1819.8 s/源 |
| 极速档（`boundary_ring=False`） | 0.9997 | 299/300 | 1340.2 s/源 |

泛化与全新种子合计 **2500 局、31 000+ 个源、零遗漏**。端到端单局 100%，现实程序运行约 0.25 s。

## 关键约定

* **计时口径**：见文首标注。虚拟上限 360000 s 仅防死循环，现实 1200 s 才是约束；
  预算判断一律用 `/enter` 返回的 `remaining_real_duration_s`。
* **并发**：附件2 表2 规定「并发发送了不同动作 → 409」。模拟器内置串行闸门，
  引擎内同时只执行一个动作；网络重试须复用原 `request_id` 与请求内容。
* **保密**：正式测试的案例真值在任何界面、完成提示与控制接口（`/api/state`）中都不显示，
  只写入加密行为日志（< 2 MB）。
* **依赖边界**：`simulator/`、`robotdog/`（通信层）、`tests/`、`tools/` 只依赖标准库；
  只有 `robotdog/solver/` 的进程内评测需要 numpy，**部署程序只用标准库**。
* **常量单一来源**：题目物理常量只在 `robotdog/consts.py` 定义一份。
  `simulator/core.py` 刻意**不**共享它——模拟器是对题目的独立复现，
  两份常量各自独立，逐位等价性测试（`tests/test_solver_equiv.py`）才有意义。
* **可复现**：策略与评测全部确定性（射线重采样用固定相位而非无播种随机数），
  同一命令跑两次逐位一致。

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/README.md` | **docs 索引**：目录里每份文档的现状与用途 |
| **`docs/问题3解题思路.md`** | **问题3 的解题思路（简要）**：建模、核心洞察、算法主循环、结果 |
| **`docs/问题4解法.md`** | **问题4 解法的唯一入口**：机制、参数档位、2500 局零遗漏实测、复现命令 |
| `docs/论文_问题4_集合覆盖路线.md` | 问题4 的论文成稿 |
| `robotdog/solver/README.md` | 求解器总说明：§1-§8 问题3，§9 问题4 |
| `docs/模拟器设计说明.md` | 逐条对照题目正文与附件的实现口径、需自行假设的部分 |
| `docs/模拟器使用说明.md` | 命令行参数、与附件2 的对应关系、测试与验收命令 |
| `docs/验收报告.md` | 自动化测试结果、计时口径勘误、已修复问题清单 |
| `docs/experiments/README.md` | 实验存档索引：原始报告已丢失，这里列出每条结论的代码/数字还活在哪里 |
| `data/README.md` | 运行时产物与证据文件清单 |
