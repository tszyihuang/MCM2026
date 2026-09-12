# `data/` — 运行时产物与对照证据

本目录分两类内容：

* **运行时产物** —— 程序每次运行都会重新生成，**随时可以删除**（目录本身靠本文件保留）。
* **`reports/`** —— **保留**：对照证据，是 `README.md` 与 `docs/` 中所有数字的来源。

## 运行时产物

| 路径 | 由谁生成 | 说明 |
| --- | --- | --- |
| `logs/` | `simulator` 导出 | 行为日志。演练为可读的 `.json`，正式测试为加密的 `.log`（< 2 MB，`ENC1` 格式）。**只留正式测试档**，其余随时可删 |
| `robot_logs/` | `robotdog.solver.deploy --log <路径>` 或 `tools/run_batch.py` | 机器狗程序**自己**记录的行为日志（JSONL），模拟器不提供该功能 |
| `results.jsonl` | `tools/run_batch.py` | 批处理统计汇总，每次运行**追加**一行 |
| `login_state.json` | `simulator.session` | 登录状态与服务器时间校验结果 |
| `perf_*.json` | `tools/perf_bench.py --save` | 性能快照，用于对照「优化不改行为」 |

> `logs/` 里保留的是**正式测试**（真值屏蔽 + 加密日志）的 3 组实测留档，
> 每组 `.json`（可读导出）与 `.log`（`ENC1` 加密件）同名成对，
> 对应的机器狗侧日志在 `robot_logs/`（同名 ID）。
> 演练（`practice_*`）与批量评测产生的日志已清理 —— 它们随时可重新生成，
> 留下的判据是「该数字是否被 `README.md` / `docs/` 引用」。

> [!TIP]
> **写日志的目录由 `--data-dir` 决定**（`tools/run_batch.py` 默认 `data/`，
> 会话会把日志放进 `<data-dir>/logs/`）。所以跑演练会在 `data/` 下留档 —— 这是设计使然，
> `data/logs/` 已在 `.gitignore` 中，不会污染版本库。定期清掉非正式档：
>
> ```powershell
> Remove-Item data\logs\practice_*.json, data\login_state.json -Force
> ```
>
> 或者跑演练时直接把它引到临时目录：`--data-dir $env:TEMP\mcm`。

## 对照证据（`reports/`）

| 路径 | 由谁生成 | 内容 |
| --- | --- | --- |
| `q3_insample.json` / `q3_deploy.json` | `tools/candidates.py` / `tools/eval_deploy.py` | 问题3 留出集（seed 9500-9799）进程内 / 部署口径的逐局明细与汇总 |
| `q4_insample.json` / `q4_deploy.json` | `tools/eval_q4.py` | 问题4 留出集双口径成绩 |
| `q4_missed_analysis.json` | `tools/eval_q4.py` / 漏源归因探针 | 问题4 漏源的角向归因（论文插图取数） |
| `pareto_p3*.json` / `pareto_p4*.json` | `tools/pareto_front.py` | 问题3（收尾闸门）/ 问题4（两档）的帕累托前沿，含留出集与泛化集 |
| `fingerprint_*.json` | `tools/decision_fingerprint.py` | 逐动作决策指纹（重构前后对照） |
| `verify_*.json` | `tools/verify_candidate.py` | 交付验收明细（确定性 + 双口径成绩） |

问题3 的留出集成绩重新生成：

```powershell
python tools\candidates.py  --modules robotdog.solver.sweeper --seeds 9500-9799 --jobs 6 `
    --out data/reports/q3_insample.json
python tools\eval_deploy.py --module  robotdog.solver.sweeper --seeds 9500-9799 --jobs 6 `
    --out data/reports/q3_deploy.json
```

评测是**确定性**的（射线重采样用固定相位而非无播种随机数），因此重跑结果逐位一致。
