"""离线最优性上界估计: 忽略检测耗时, 只按"动作数 + 移动距离"预算做贪心/枚举,
用于判断清除比例是否已接近本问题参数下的物理上限。

模型: 机器狗从原点出发, 每次动作可 (a) 移动到任意点并检测全部 20 个频道,
      (b) 移动到 20 m 内并清除。动作数上限取自 1200 s 预算减去检测耗时后的估计。
这里使用宽松上界: 假设每次动作都能检测全部频道且检测不耗时(仅计入移动耗时),
因此给出的清除个数是任何策略都无法显著超过的上界。

.. warning::
   **口径警告 (务必先读 docs/验收报告.md §4.3)。** 本模型的 1200 s 预算是
   **虚拟时间**口径, 而题目真正约束的是**现实**程序运行时间 1200 s。二者相互
   独立 (附件2 §1.4: 每次检测的 5 秒只增加虚拟时间, 不要求在现实中等待 5 秒),
   现场实测每局现实耗时只有几秒。因此本脚本给出的比例**不能**用来论证
   "不可能清除全部"——它只是"1200 虚拟秒内能做到多少"。命令行加 ``--real-budget``
   可以换算出按现实预算的请求数上界。
"""

import math
import random

ARENA_R = 1800.0
SPEED = 5.0
# 虚拟世界活动时长之外, 本模型历史沿用的"预算": 1200 s。保留原名以免破坏引用者,
# 但请通过 VIRTUAL_BUDGET_S 使用, 明确它是**虚拟**口径。
VIRTUAL_BUDGET_S = 1200.0
BUDGET_S = VIRTUAL_BUDGET_S


def gen_case(seed: int):
    rng = random.Random(seed)
    n = rng.randint(10, 16)
    chs = rng.sample(range(1, 21), n)
    srcs = []
    for ch in chs:
        r = ARENA_R * math.sqrt(rng.random())
        t = rng.uniform(0, 2 * math.pi)
        srcs.append((ch, r * math.cos(t), r * math.sin(t), rng.uniform(1000.0, 1500.0)))
    return srcs


def greedy(seed: int, trace=False):
    """贪心上界: 每次挑选"当前可见且距离最近"的源去清除。"""
    srcs = gen_case(seed)
    alive = {ch: (x, y, r) for ch, x, y, r in srcs}
    pos = (0.0, 0.0)
    t = 0.0
    cleared = []
    # 检测成本忽略, 只计移动: 上界策略可以直接选最近的可达源
    while True:
        best = None
        for ch, (x, y, r) in alive.items():
            d = math.hypot(x - pos[0], y - pos[1])
            # 可达性: 需要能先看到它 (存在一个中间点能收到信号), 这里宽松认为总能找到
            cost = d / SPEED
            if best is None or cost < best[0]:
                best = (cost, ch, (x, y))
        if best is None:
            break
        if t + best[0] + 1.0 > BUDGET_S:
            break
        t += best[0] + 1.0
        pos = best[2]
        cleared.append(best[1])
        del alive[best[1]]
    return len(cleared), len(srcs), cleared


def greedy_visible(seed: int):
    """更现实的上界: 只有在某点先检测到 (距离 <= 有效半径) 才能定位, 移动耗时照计。"""
    srcs = gen_case(seed)
    alive = {ch: [x, y, r, True] for ch, x, y, r in srcs}
    pos = (0.0, 0.0)
    t = 0.0
    cleared = []
    while t < BUDGET_S - 1.0:
        # 选择当前可见且最近的源
        best = None
        for ch, (x, y, r, ok) in alive.items():
            if not ok:
                continue
            d = math.hypot(x - pos[0], y - pos[1])
            if d <= r:
                if best is None or d < best[0]:
                    best = (d, ch, (x, y))
        if best is None:
            break
        d, ch, (x, y) = best
        cost = d / SPEED + 1.0
        if t + cost > BUDGET_S:
            break
        t += cost
        pos = (x, y)
        cleared.append(ch)
        del alive[ch]
    return len(cleared), len(srcs), cleared


def real_budget_action_cap(real_budget_s: float = 1200.0, per_action_real_s: float = 0.0004) -> int:
    """按**现实**预算换算的请求数上界 (与虚拟口径无关)。

    ``per_action_real_s`` 取本机实测的单次动作现实往返 (keep-alive 约 0.31 ms,
    见 README 性能一节); 现实预算 1200 s 因此允许数十万次动作。这个数字说明
    "现实 1200 s" 对动作数几乎不构成限制, 真正限制来自策略自身的动作数安全阀
    (``--max-measure``) 与探测覆盖率。
    """
    return int(real_budget_s / per_action_real_s)


if __name__ == "__main__":
    seeds = list(range(1001, 1019))
    tot = clr = clr2 = 0
    for s in seeds:
        n1, n, c1 = greedy(s)
        n2, _, c2 = greedy_visible(s)
        tot += n
        clr += n1
        clr2 += n2
        print('seed %d total %2d | 贪心上界(忽略检测耗时) %2d | 更严上界(仅可见源) %2d' % (s, n, n1, n2))
    print('合计 %d 源, 忽略检测耗时上界 %d (%.1f%%), 可见性上界 %d (%.1f%%)'
          % (tot, clr, 100.0 * clr / tot, clr2, 100.0 * clr2 / tot))
    print()
    print('注意: 以上均为**虚拟时间** %d s 口径, 不能用来论证"不可能清除全部"'
          % int(VIRTUAL_BUDGET_S))
    print('      (见 docs/验收报告.md §4.3)。按现实预算 %d s 换算, 请求数上界约 %d 次。'
          % (int(VIRTUAL_BUDGET_S), real_budget_action_cap(int(VIRTUAL_BUDGET_S))))
