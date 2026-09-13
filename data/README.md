# `data/` — 运行时产物与对照证据

本目录分两类内容：

* **运行时产物** —— 程序每次运行都会重新生成，**随时可以删除**（目录本身靠本文件保留）。
* **`reports/`** —— **保留**：对照证据，是 `README.md` 与 `docs/` 中所有数字的来源。

## 运行时产物

| 路径 | 由谁生成 | 说明 |
| --- | --- | --- |
| `logs/` | `simulator` 导出 | 行为日志。演练为可读的 `.json`，正式测试为加密的 `.log`（< 2 MB，`ENC1` 格式）。当前留档 **正式测试 8 组**（问题3 3 局 + 问题4 5 局）+ **演练 5 组** |
| `robot_logs/` | `robotdog.solver.deploy --log <路径>` 或 `tools/run_batch.py` | 机器狗程序**自己**记录的行为日志（JSONL），模拟器不提供该功能。当前 16 个 |
| `results.jsonl` | `tools/run_batch.py` | 批处理统计汇总，每次运行**追加**一行 |
| `login_state.json` | `simulator.session` | 登录状态与服务器时间校验结果 |
| `perf_*.json` | `tools/perf_bench.py --save` | 性能快照，用于对照「优化不改行为」 |

> `logs/` 与 `robot_logs/` 按**案例 ID 配对**，一组留档是三件套：
> `.json`（可读导出）+ `.log`（`ENC1` 加密原件）+ 机器狗侧 `.jsonl`。
>
> 当前口径：正式测试留 **问题3 `C20260913-P3-*` 三局**与**问题4 `C20260912-P4-*` 五局**
> （即 `README.md` / `docs/问题4论文.md` 里 63/64 的那五局，**对应已被替换的旧版规划器**）；
> 演练只留 `docs/问题4论文.md` §5 引用的三局（另有两局是**当前 opt1 策略**的端到端实测：
> seed 9500 全清 14/14、seed 9501 全清 10/10，现实耗时 0.11~0.13 s）。`robot_logs/` 里另有
> 6 个属于**已被替换的旧问题三正式档**（`C20260911-P3-*`、`C20260912-P3-*`），保留作对照。
>
> 取舍判据只有一条：**该数字是否被 `README.md` / `docs/` 引用**。
> 不被引用的演练档、被取代的旧正式档、临时评测输出都已清理（见文末清理记录）。

> [!TIP]
> **写日志的目录由 `--data-dir` 决定**（`tools/run_batch.py` 默认 `data/`，
> 会话会把日志放进 `<data-dir>/logs/`）。跑演练、批量评测和对照实验时，
> **请一律把它指到工作区外**，不要在仓库里留档：
>
> ```powershell
> python tools\run_batch.py --problem 4 --module practice --runs 5 --data-dir $env:TEMP\mcm
> ```
>
> `.gitignore` 只忽略 `data/logs/practice_*.json`，**`robot_logs/` 没有忽略规则**
> —— 不指到工作区外，演练产生的机器狗日志会被 `git add .` 一起收进版本库。

## 对照证据（`reports/`）

| 路径 | 由谁生成 | 内容 |
| --- | --- | --- |
| `q3_insample.json` / `q3_deploy.json` | `tools/candidates.py` / `tools/eval_deploy.py` | 问题3 标定集（seed 9500-11499，2000 局）进程内 / 部署口径的逐局明细与汇总 |
| `q3_deploy_prev396.json` / `q3_deploy_cover377.json` / `q3_deploy_speed250.json` | `tools/eval_deploy.py` | 问题3 收尾口径的三个留档点：旧参数 396.4 s/源 → 保障模式 **377.5 s/源 / 1.0000 / 失败率 0%** → 速度优先 λ=0.30 **250.4 s/源 / 0.9907 / 29%**；**当前发布 = 保障模式（`q3_deploy_cover377.json`）**，速度优先 λ=0.05（281.6 s/源 / 0.9988 / 4.5%）是可选提速档 |
| `q3_deploy_gen1000.json` / `q3_deploy_indep1000.json` | `tools/eval_deploy.py` | 问题3 独立集复核：seed 8000-8999 / 20000-20999 各 1000 局（1.0000 / 382.9、375.4 s/源） |
| `p3_tune_oat500.json` / `p3_tune_combo500.json` / `p3_tune_combo2000.json` / `p3_tune_combo_indep1000.json` / `p3_tune_combo_fresh1000.json` | `tools/tune_p3.py` | 问题3 参数扫描（单旋钮 OAT + 组合候选），部署口径；当前发布参数的选取依据 |
| `pareto_p3*.json` | `tools/pareto_front.py` | 收尾口径与门限 λ 的帕累托前沿（**已按当前参数重算**，供论文表 5/表 6 与图 3/图 4） |
| `pareto_p3_ci.json` | `tools/pareto_ci.py` | 前沿各点相对采纳点 λ=0.20 的配对 95% 区间（按案例 bootstrap 4000 次；论文 §5.2 表 2） |
| `q4_calib500.json` | `tools/eval_q4.py` | 问题4 标定集（seed 9500-9999，500 局）逐局明细与汇总（**历史版本**，727 s/源） |
| `q4_holdout1000.json` | `tools/eval_q4.py` | 问题4 独立测试集（seed 20000-20999，1000 局，从未参与标定）（**历史版本**） |
| `q4_calib_v4.json` | `tools/eval_q4.py` | 上一版问题4（`sweeper4_legacy`）标定集 500 局：0.9960 / 545.3 s/源 |
| `q4_holdout_v4.json` | `tools/eval_q4.py` | 上一版问题4 独立集 1000 局：0.9929 / 549.8 s/源 |
| `q4_deploy_v3.json` | `tools/eval_q4.py --deploy` | 上一版问题4 部署口径 300 局：0.9931 / 551.9 s/源 |
| `q4_opt1_calib500.json` | `tools/eval_q4.py --deploy` | **当前版本**问题4（`sweeper4` = opt1 在线路线重优化 + 18 站发布布局）标定集 500 局：**0.9995 / 463.7 s/源** |
| `q4_opt1_holdout1000_deploy.json` | `tools/eval_q4.py --deploy` | **当前版本**问题4 独立集 1000 局（从未参与标定）：**0.9994 / 461.9 s/源** |
| `q4_opt1_deploy300.json` | `tools/eval_q4.py --deploy` | **当前版本**问题4 部署口径 300 局：**0.9990 / 461.4 s/源** |
| `q4_opt1_ladder500.json` | `tools/probes/q4_opt1_ladder.py` | **当前版本**问题4 的 A~E 调度器阶梯与单旋钮消融（标定 500 局，逐档汇总） |
| `q4_ship18_calib4000.json` | `tools/eval_q4.py --deploy` | **当前版本**问题4 标定集 4000 局（9500-13499）：0.9995 / 464.1 s/源 |
| `q4_relax_R*.json` | `tools/eval_q4.py --deploy` | 覆盖—速度前沿各档（24/23/21/18/17/16/15 站）的 4000 局记录 |
| `q4_relax_frontier.md` | 汇总 | **覆盖—速度前沿**：档位表、风险口径、被排除的旋钮清单、切换方法 |
| `q4_opt1_route_audit.txt` | `tools/probes/q4_opt1_route.py` / `q4_opt1_layout.py` | 走线审计（`L_actual` / 骨架 / 全知上界 / 全知对照，两版各 200 局）与站点布局覆盖审计（角隙判据 + 抽样覆盖重数） |
| `fingerprint_*.json` | `tools/decision_fingerprint.py` | 逐动作决策指纹（重构前后对照）。**按需生成，当前仓库内没有** |
| `verify_*.json` | `tools/verify_candidate.py` | 交付验收明细（确定性 + 双口径成绩）。**按需生成，当前仓库内没有** |

问题3 的标定集成绩重新生成：

```powershell
python tools\candidates.py  --modules robotdog.solver.sweeper --seeds 9500-11499 --jobs 6 `
    --out data/reports/q3_insample.json
python tools\eval_deploy.py --module  robotdog.solver.sweeper --seeds 9500-11499 --jobs 6 `
    --out data/reports/q3_deploy.json
```

评测是**确定性**的（射线重采样用固定相位而非无播种随机数），因此重跑结果逐位一致。

## 清理记录

**2026-09-13**　按「不被 `README.md` / `docs/` 引用即删」清理，回收约 **24 MB**
（项目 33 MB → 9 MB）：

| 已删 | 说明 |
| --- | --- |
| `data/logs/superseded_p3/`（6 局） | 已被 `C20260913-P3-*` 三局取代的旧问题三正式档 |
| 旧问题四正式档 6 局 | 论文只引用 63/64 那 5 局，其余 6 局连同机器狗侧日志一并删除 |
| `data/reports/opt1_port_*` | opt1 对照实验的 5 个 E2E 目录 + 3 份 JSON，未进任何文档 |
| 未被引用的演练档 19 个 | 演练档随时可重新生成，只留被文档引用与最近一次实测的 |
| `paper/preview/*.png`（18 张） | 页面渲染预览，无文档引用，可用 `paper/main.pdf` 重新渲染 |
| `__pycache__/`、`paper/main.aux` `.log` `.out` | 可再生的缓存与 LaTeX 编译中间产物 |

> [!CAUTION]
> 以上内容**都不在版本库中**（`data/` 多为运行时产物），删除后无法用 git 找回；
> 正式测试留档一旦删除即永久丢失。后续新增证据请同时更新本文件与 `README.md` 的引用。

---

## 2026-09-13 问题4 策略替换后新增的报告

问题4 发布策略换成 **22 站 + 中继测量版**（`robotdog/solver/sweeper4.py`，决策层来自
单文件版 `Desktop/solver_q4.py`）后新增的三份证据，全部是 `tools/eval_q4.py --deploy`
的**部署口径**（`knows_total=False`，正式测试看不到真值总数）：

| 文件 | 口径 | 成绩 |
| --- | --- | --- |
| `q4_ship_relay_calib500.json` | 标定集 seed 9500-9999，500 局 | 清除比例 **1.0000**，500/500 全清，**451.2 s/源** |
| `q4_ship_relay_holdout1000.json` | 独立集 seed 20000-20999，1000 局（未参与标定） | **1.0000**，1000/1000 全清，**450.6 s/源** |
| `q4_ship_relay_fresh2000.json` | 全新集 seed 30000-30999 + 40000-40999，2000 局 | **1.0000**，2000/2000 全清，**452.6 s/源** |

上面表格里的 `q4_opt1_*.json`（0.9995 / 463.7 s/源）与 `q4_ship18_calib4000.json`
是**上一版 18 站档**的记录，现在对应留档模块 `robotdog/solver/sweeper4_ring18.py`。
两档的对照、切换方法与"为什么 22 站能零漏源"的几何判据见 `q4_relax_frontier.md`
与 `tools/probes/q4_opt1_layout.py` 的输出。

### 2026-09-13 二次调整：16 站速度档（已退回，仅作对照）

"降低清除率要求能不能换速度"的站数轴实测（3500 局 = 标定 500 + 独立 1000 + 全新 2000，
全部 `tools/eval_q4.py --deploy`）。结论：**放宽清除率几乎换不到速度**
（从 100% 一路放到 91% 只快 13%），可用的一档是 16 站。

| 文件 | 口径 | 成绩 |
| --- | --- | --- |
| `q4_ship16_calib500.json` | 标定集 seed 9500-9999，500 局 | 清除比例 **0.9992**，495/500 全清，**424.9 s/源** |
| `q4_ship16_holdout1000.json` | 独立集 seed 20000-20999，1000 局（未参与标定） | **0.9992**，990/1000 全清，**422.3 s/源** |
| `q4_ship16_fresh2000.json` | 全新集 seed 30000-30999 + 40000-40999，2000 局 | **0.9992**，1978/2000 全清，**421.6 s/源** |
| `q4_ship16_http40.jsonl` | 真实进程 + HTTP（`tools/run_batch.py --problem 4 --module formal`），seed 20000-20039，40 局 | **1.0000**，40/40 全清，**413.1 s/源**，现实运行 0.126 s |

`q4_ship_relay_*.json` 是 **22 站档**的同批记录（3500 局 **1.0000 / 450.6 s/源 / 零漏源**），
**它是当前发布的默认档**；16 站档留作"更快但会漏源"的对照。两档的几何差异
（最坏角隙 178.99° vs 360°）、3 局小样本风险（0% vs 3.1%）与完整前沿见
`q4_relax_frontier.md` §5 与 `../README.md`「精度 ⇄ 速度」。

### 2026-09-13 三次调整：两题都切到"100% 清除"档

按"确保全部干扰源被清除"的要求，发布配置改回/改为：

| 文件 | 口径 | 成绩 |
| --- | --- | --- |
| `q4_ship22_http20.jsonl` | 真实进程 + HTTP（`tools/run_batch.py --problem 4 --module formal`），seed 20000-20019，20 局 | **1.0000**，20/20 全清，现实运行 0.2 s |
| `q3_cover_http20.jsonl` | 真实进程 + HTTP（`--problem 3`），seed 21500-21519，20 局 | **1.0000**，20/20 全清，现实运行 0.1~0.3 s |

问题3 用 `q3_deploy_cover377.json`（保障模式 377.5 s/源 / 1.0000），问题4 用
`q4_ship_relay_*.json`（22 站 450.6~452.6 s/源 / 1.0000），合计 7000 / 5500 局部署口径
零漏源。切换方法见 `../robotdog/README.md` §6.1。
