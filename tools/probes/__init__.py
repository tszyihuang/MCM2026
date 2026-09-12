"""论文取数用的一次性数值探针.

与 ``tools/`` 下的常规工具不同, 这些脚本**不参与回归**, 只用于复核论文中引用的
具体数字。每个脚本都可直接运行并打印表格:

    python tools/probes/verify_q4_theory4.py    # 引理 1/1b/3 的数值验证
    python -m tools.probes.verify_q4_equiv      # 环境逐位等价 + 信念自检
"""
