# 实验与消融记录（索引）

本目录用于存放参数标定、消融与负面结果。**当前没有独立文件**——
结论已并入正文文档与代码注释，避免出现重复且可能过期的数字。

## 结论还活在哪里

| 主题 | 结论 | 位置 |
| --- | --- | --- |
| 问题3 收尾闸门 | 首批 300 局标定时，残余期望源数在 [0.02, 0.20) 的专程探测命中率仅 0.9%~3.2%，代价却是 166~203 s 往返 ⇒ 闸门取 0.20；该权衡现由 2000 局 + 1000 局前沿复核，配对 bootstrap 证明 λ 由 0.30 收紧到 0.20 的边际价格已是 0.60→0.40 段的 3.5 倍，故取边际代价开始陡增的 0.20 | `sweeper.build_cfg()` 的 docstring；`docs/问题3论文.md` §5.2（表 1/表 2、图 1）；`tools/pareto_front.py` 现跑前沿、`tools/pareto_ci.py` 现跑置信区间 |
| 问题3 单点联合选择 | 把"顺路停车点"与"弦式侧移"合并为一次 `(ε, δ)` 联合优化；候选集恒含保守动作 | `sweeper.joint_measure_point` / `_joint_pred` / `_joint_rem` 的 docstring |
| 问题3 截断核 | 部署口径 +2.0 s/源，汇率不划算 ⇒ 不采纳 | `sweeper.cover_point` 的 docstring |
| 检测概率积分 | 远距离 `no_signal` 必须用 `1 − E[F_R(d)]`，否则 π 崩塌 | `belief.py` |

更完整的**负面结果清单**（静默失效、口径陷阱、被误判的优化）见
`../robotdog/solver/README.md` §7「工程要点」。

## 复现方式

所有结论都可以用现成工具重新跑出来，不需要历史脚本：

```powershell
python tools\pareto_front.py                    # 收尾闸门前沿 (标定集 + 独立测试集)
python tools\pareto_ci.py                       # 前沿各点的配对 95% 区间 (bootstrap)
python tools\decision_fingerprint.py            # 逐动作决策指纹 (重构安全网)
```
