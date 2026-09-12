# 实验与消融记录（索引）

本目录用于存放参数标定、消融与负面结果。**当前没有独立文件**——
结论已并入正文文档与代码注释，避免出现重复且可能过期的数字。

## 结论还活在哪里

| 主题 | 结论 | 位置 |
| --- | --- | --- |
| 问题3 收尾闸门 | 残余期望源数在 [0.02, 0.20) 时专程探测命中率仅 0.9%~3.2%，代价却是 166~203 s 往返 ⇒ 闸门取 0.20 | `sweeper.build_cfg()` 的 docstring；`docs/问题3解题思路.md` §5；`tools/pareto_front.py --problem 3` 可现跑前沿 |
| 问题3 单点联合选择 | 把"顺路停车点"与"弦式侧移"合并为一次 `(ε, δ)` 联合优化；候选集恒含保守动作 | `sweeper.joint_measure_point` / `_joint_pred` / `_joint_rem` 的 docstring |
| 问题3 截断核 | 部署口径 +2.0 s/源，汇率不划算 ⇒ 不采纳 | `sweeper.cover_point` 的 docstring |
| 检测概率积分 | 远距离 `no_signal` 必须用 `1 − E[F_R(d)]`，否则 π 崩塌 | `belief.py` |
| 问题4 定向源物理 | 单点发现概率恒为 1/2，与距离无关；保证发现的探测点间距须 ≤1000 m | `docs/问题4解法.md`；`tools/probes/verify_q4_theory4.py` 可数值验证 |
| 问题4 贴边补扫环 | 把清除比例从 0.9996 推到 1.0000 | `sweeper4.build_cfg()` / `boundary_ring_points` |

更完整的**负面结果清单**（静默失效、口径陷阱、被误判的优化）见
`../robotdog/solver/README.md` §7「工程要点」。

## 复现方式

所有结论都可以用现成工具重新跑出来，不需要历史脚本：

```powershell
python tools\pareto_front.py --problem 3        # 问题3 收尾闸门前沿 (留出集 + 泛化集)
python tools\pareto_front.py --problem 4        # 问题4 全清档 / 极速档
python tools\probes\verify_q4_theory4.py        # 问题4 理论引理的数值验证
python tools\decision_fingerprint.py            # 逐动作决策指纹 (重构安全网)
```
