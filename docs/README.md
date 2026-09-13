# docs 索引

本目录存放建模说明、求解器解法、模拟器说明与验收报告。
工程提供两条求解器路线：问题3 的 `sweeper`（全向源）与问题4 的 `sweeper4`（含定向源）。
两者各自成篇、互不依赖；模拟器完整实现问题3 / 问题4 两个环境。

---

## 1. 按主题索引

| 文件 | 属于 | 内容 |
| --- | --- | --- |
| **`问题一与问题二论文.md`** | 问题1/2 | 定位区域直径圆覆盖、精确直径优化的定稿解法（数字由 `tools/probes/q12_review.py` 复现） |
| **`问题3论文.md`** | 问题3 | 建模与算法（论文写法）：问题描述、问题分析、模型建立、算法设计、结果与分析 |
| **`../paper/main.tex`** | 问题3 / 全篇 | **Overleaf 可编译的论文主文件**（XeLaTeX + ctex，问题三全文已写入，问题一/二/四为红色 `\todo{}` 占位）；插图在 `../paper/figures/`，上传包为 `../paper/overleaf_upload.zip` |
| **`问题4论文.md`** | 问题4 | 含定向源的建模与算法：角隙覆盖判据与"速度优先"布局放宽（18 站，**文中是上一版方案**）、加权交会定位、在线路线重优化、走线冗余归因（全知对照）、消融与指标达成分析；**当前发布 = 16 站速度档 + 中继测量版**（22 站零漏检档只差 `ring_def` 一行；数字见 `../README.md`，插图/正文待重跑） |
| `模拟器设计说明.md` | 模拟器 | 模拟器逐条对照题目正文与附件的实现口径、需自行假设的部分 |
| `模拟器使用说明.md` | 模拟器 | 命令行参数、与附件2 的对应关系、测试与验收命令 |
| `验收报告.md` | 工程 | 自动化测试结果、计时口径勘误、已修复问题清单 |
| `工程说明.md` | 工程 | **README 的详细工程版**：四层架构、快速开始、竞赛成绩、参数标定与留档对照、口径提醒（文中路径均相对仓库根目录）。首页请读 `../README.md` |
| `experiments/README.md` | 工程 | 实验存档索引：原始报告已丢失，列出每条结论的代码与数字现在还活在哪里 |
| `figures/` | 问题1/2 / 问题3 / 问题4 | 论文插图（配色统一由 `../tools/figstyle.py` 提供，改配色只需改该文件）。**问题1/2 4 张**（`q12_q1_region` / `q12_q1_coverage` / `q12_q2_feasible` / `q12_q2_worst_r`，`问题一与问题二论文.md` 正文全部引用，`tools/probes/q12_make_figures.py` 生成）；**问题3 4 张**（`q3_nn` / `q3_pareto_forest` / `q3_posterior` / `q3_results`，正文全部引用，`tools/make_paper_figures.py` 生成，`paper/figures/` 为其副本）；**问题4 5 张**（`q4_cover_gap` / `q4_cover_frontier` / `q4_scan_sets` / `q4_phases` / `q4_compare`，`问题4论文.md` 正文全部引用，`tools/probes/q4_make_figures.py` 生成） |

求解器的代码结构、参数标定与工程要点在 **`../robotdog/solver/README.md`**
与本文档互为补充。

---

## 2. 从哪里开始读

| 你想知道 | 读这个 |
| --- | --- |
| 问题3 的策略怎么设计的 | `问题3论文.md` → `../robotdog/solver/README.md` |
| 问题4（定向源）的策略怎么设计的 | `../robotdog/solver/sweeper4.py`（**当前发布 = 16 站速度档 + 中继测量版**；22 站零漏检档只差 `ring_def` 一行，18 站档与初版已移出 `solver/`，见 `../robotdog/README.md`「整理记录」）→ 论文 `问题4论文.md`（按上一版 18 站方案撰写） |
| 模拟器怎么用、怎么验收 | `模拟器使用说明.md`、`验收报告.md` |
| 项目全貌与快速上手 | `../README.md`；需要参数标定与留档对照的细节时再读 `工程说明.md` |
| 模拟器与题目的对应关系 | `模拟器设计说明.md` |
| 论文插图从哪来 | `figures/`、`../tools/probes/q12_make_figures.py`（问题1/2）、`../tools/make_paper_figures.py`（问题3）、`../tools/probes/q4_make_figures.py`（问题4）、`../tools/figstyle.py`（统一配色） |
