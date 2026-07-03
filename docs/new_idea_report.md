# 新 idea 专版：偏好优化中的风险-多样性解耦

日期：2026-07-03

本文只保留新的研究 idea 和支持它的实验结果，不再展开旧的“PCE 坍缩漏洞”叙事。

## A. 正式版

### A1. 研究题目

建议题目：

```text
面向偏好优化语言模型的风险-多样性解耦诊断与预警研究
```

英文题目：

```text
Diagnosing and Warning Risk-Diversity Decoupling in Preference-Optimized Language Models
```

正式表述：

```text
本研究关注偏好优化，尤其是 Direct Preference Optimization，如何在不降低输出多样性的情况下提高模型安全风险。核心目标是提出风险-多样性解耦诊断框架，用于识别偏好优化后模型在多个语义输出模式中分散产生风险的现象。
```

### A2. 核心 idea

正式表述：

```text
偏好优化可能不会稳定诱发 sampled-mode collapse，但可能诱发 distributed risk migration：模型在保持或增加输出多样性的同时，系统性提高 guardian-scored harmfulness。
```

中文解释：

```text
DPO 不一定让模型反复输出同一种危险回答，但可能让模型用更多不同方式回答危险问题。这种现象不是“风险集中”，而是“风险分散迁移”。
```

核心判断：

```text
风险上升和多样性下降不是必然绑定的。
```

也就是说：

```text
Risk can increase without diversity collapse.
```

### A3. 为什么这个 idea 重要

传统直觉更关注模式坍缩：

```text
如果模型反复输出同一种有害回答，风险容易被发现。
```

新 idea 关注更隐蔽的问题：

```text
如果模型每次输出都不同，但每次都更接近风险区域，传统重复性检测就不容易发现。
```

正式意义：

```text
Distributed risk migration may be harder to detect than collapse-based risk because semantic and lexical repetition signals remain weak while guardian-scored risk rises across multiple modes.
```

这使得研究问题从“模型是否坍缩”升级为：

```text
偏好优化如何改变风险分布？
风险是集中在一个模式里，还是分散到多个模式里？
能否在训练或评估阶段提前发现这种变化？
```

### A4. 核心概念

| 概念 | 正式定义 | 作用 |
| --- | --- | --- |
| Risk | Guardian 或其他安全分类器对输出的风险评分 | 衡量回答是否安全 |
| Diversity | 多次采样输出在语义模式上的分散程度 | 衡量回答是否多样 |
| Determinism | 最大语义簇的概率质量 | 衡量是否集中到一个主导模式 |
| Mode Entropy | 语义簇分布的熵 | 衡量输出模式是否分散 |
| Distributed Risk Migration | 风险上升但多样性不下降 | 新 idea 的核心现象 |
| RDI | Risk Distribution Index，跨语义模式的期望风险 | 衡量整体分布风险 |
| Risk Entropy | 风险在不同语义模式中的分散程度 | 衡量风险是否集中 |

### A5. 新指标设想

#### Risk Distribution Index

定义：

```text
RDI(x) = sum_c p(c | x) * h(c, x)
```

其中：

- `x` 是 prompt；
- `c` 是语义聚类得到的输出模式；
- `p(c | x)` 是该模式的采样概率；
- `h(c, x)` 是 Guardian 对该模式的风险评分。

正式解释：

```text
RDI measures expected risk over all sampled semantic modes, instead of only the dominant mode.
```

中文解释：

```text
RDI 不只看最大那类回答危不危险，而是看所有回答模式加权后的整体风险。
```

#### Risk Entropy

定义：

```text
q_c = p(c | x) * h(c, x) / sum_j p(j | x) * h(j, x)
RiskEntropy(x) = - sum_c q_c log q_c
```

正式解释：

```text
Risk Entropy measures whether risk is concentrated in a single response mode or distributed across multiple response modes.
```

中文解释：

```text
风险熵看的是：风险到底集中在一种回答里，还是分散在很多回答里。
```

### A6. 四象限诊断框架

| 象限 | 风险变化 | 多样性变化 | 含义 |
| --- | --- | --- | --- |
| Collapse Risk | 风险上升 | 多样性下降 | 传统 PCE 风险，固定地变危险 |
| Distributed Risk | 风险上升 | 多样性上升或不降 | 新主线，分散地变危险 |
| Safe Concentration | 风险下降 | 多样性下降 | 集中到安全拒答或安全模板 |
| Benign Diversity | 风险不升 | 多样性不降 | 相对安全或中性变化 |

正式表述：

```text
The proposed framework treats collapse-based risk as one quadrant, and distributed risk migration as a distinct and potentially more subtle quadrant.
```

### A7. 研究问题

| 编号 | 研究问题 |
| --- | --- |
| RQ1 | DPO 是否会提高 guardian-scored risk，同时保持或增加语义输出多样性？ |
| RQ2 | 这种风险-多样性解耦是否能在不同模型规模上复现？ |
| RQ3 | 风险是集中到一个 dominant mode，还是分散到多个 semantic modes？ |
| RQ4 | prompt 类型、拒答率、preference margin、mode entropy 能否预测风险迁移？ |
| RQ5 | 能否设计预警或缓解方法，降低分散式风险，而不是简单让模型全拒答？ |

## A8. 为什么要做这些实验

### 实验逻辑总览

| 实验目的 | 为什么必须做 | 对新 idea 的作用 |
| --- | --- | --- |
| Qwen3-1.7B 两 seed DPO | 检查现象是否在较新模型上稳定出现 | 直接验证 risk up + diversity up |
| Qwen3-4B gate | 检查是否不是 1.7B 偶然现象 | 提供跨规模支持 |
| S0.1 held-out 30 | 检查风险变化是否能在新 prompt 上出现 | 支持 risk movement 不只存在于原始 prompt |
| Refusal-template control | 验证风险和多样性是两个可分离维度 | 支持四象限框架 |
| Granite Guardian fixed-response control | 检查风险评分不是只由 prompt 驱动 | 支持 Guardian 风险信号可信 |
| Full 50 prompt heterogeneity | 检查 aggregate 是否掩盖 prompt 差异 | 支持 prompt-stratified 诊断 |

### 实验 1：Qwen3-1.7B

为什么做：

```text
如果新 idea 成立，应当能在现代 Qwen 模型上看到风险评分上升，同时多样性指标不下降。
```

设置：

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-1.7B |
| 训练 | LoRA-DPO |
| steps | 300 |
| seeds | 42、43 |
| prompts | first 10 |
| samples | 每 prompt 16 |
| Guardian | Granite Guardian |

结果：

| 指标 | 结果 |
| --- | ---: |
| Determinism Delta | -0.0531，CI [-0.1000, -0.0125] |
| Mode Entropy Delta | +0.1677，CI [+0.0301, +0.3117] |
| Proxy PCE Delta | +0.4750，CI [+0.3469, +0.6000] |
| Guardian-PCE Yes Delta | +0.4781，CI [+0.3094, +0.6500] |
| Guardian-PCE Score Delta | +0.4359，CI [+0.2945, +0.5874] |
| Dominant Harm | robust_increase |
| Collapse Gate | robust_fail |
| Prompt Split | 2 pass / 0 mixed / 18 fail |

Raw audit：

| 条件 | Refusal | Compliance | Proxy-Harmful |
| --- | ---: | ---: | ---: |
| step0 | 0.575 | 0.156 | 0.031 |
| final seed42 | 0.156 | 0.706 | 0.575 |
| final seed43 | 0.156 | 0.650 | 0.537 |

正式解释：

```text
Qwen3-1.7B provides the strongest support for risk-diversity decoupling: guardian-scored risk rises robustly, while determinism decreases and mode entropy increases.
```

### 实验 2：Qwen3-4B

为什么做：

```text
1.7B 结果可能只是单模型现象，因此需要更大模型做方向复核。
```

设置：

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-4B |
| 训练 | LoRA-DPO |
| steps | 100 |
| seed | 42 |
| prompts | first 10 |
| samples | 每 prompt 16 |

结果：

| 指标 | 结果 |
| --- | ---: |
| Determinism Delta | -0.0187，CI [-0.0375, +0.0000] |
| Mode Entropy Delta | +0.0701，CI [+0.0000, +0.1403] |
| Proxy PCE Delta | +0.0125，CI [+0.0000, +0.0312] |
| Guardian-PCE Yes Delta | +0.4688，CI [+0.3187, +0.6250] |
| Guardian-PCE Score Delta | +0.3981，CI [+0.2658, +0.5286] |
| Dominant Harm | robust_increase |
| Collapse Gate | robust_fail |
| Prompt Split | 0 pass / 0 mixed / 10 fail |

正式解释：

```text
Qwen3-4B reproduces the same qualitative pattern as Qwen3-1.7B: risk increases while collapse metrics move in the opposite direction.
```

### 实验 3：S0.1 held-out 30

为什么做：

```text
新 idea 不能只依赖 first-10 prompts，需要检查新 prompt 上是否仍有安全风险移动。
```

设置：

| 项目 | 值 |
| --- | --- |
| prompt file | `data/advbench_s0_1_heldout_30.jsonl` |
| prompts | 30 |
| samples | 每 prompt 32 |
| seeds | 42、43 |
| Guardian | Granite Guardian |

结果：

| 指标 | 结果 |
| --- | ---: |
| Seed42 Det Delta | -0.0125 |
| Seed43 Det Delta | -0.0115 |
| Seed42 Entropy Delta | +0.0190 |
| Seed43 Entropy Delta | +0.0322 |
| Pooled Prompt Split | 21 pass / 4 mixed / 35 fail |
| Pooled Pass Rate | 35.0% |
| Guardian-PCE Yes Delta | +0.0307，CI [+0.0130, +0.0490] |
| Guardian-PCE Score Delta | +0.0211，CI [+0.0064, +0.0370] |
| Dominant Harm | robust_increase |

正式解释：

```text
S0.1 supports the distinction between risk movement and collapse movement: held-out prompts show guardian-risk increase, while determinism does not increase and entropy does not decrease.
```

### 实验 4：Refusal-template control

为什么做：

```text
如果要研究风险-多样性解耦，就必须证明“集中”本身不等于“危险”。同样的 DPO 管线如果推向拒答，风险应当下降。
```

结果摘要：

- determinism 上升；
- mode entropy 下降；
- refusal 上升；
- compliance 下降；
- proxy harmfulness 下降；
- dominant Guardian harmfulness 下降。

正式解释：

```text
The refusal-template control validates that risk and diversity are separable axes: concentration can be safe when the concentrated mode is refusal.
```

### 实验 5：Granite Guardian fixed-response control

为什么做：

```text
如果 Guardian 只是因为 prompt 有害就打高分，那么风险指标没有意义。需要验证 Guardian 是否真的看 response。
```

结果：

| Fixed Response | Guardian Yes |
| --- | ---: |
| refusal | 0/10 |
| neutral boundary | 1/10 |
| unsafe-intent placeholder | 10/10 |
| short chosen template | 9/10 |

正式解释：

```text
The fixed-response control supports that Granite Guardian is response-sensitive, not merely prompt-sensitive.
```

### 实验 6：Full 50 prompt heterogeneity

为什么做：

```text
新诊断框架必须解释为什么 aggregate 指标可能误导，因此需要 prompt-level 统计。
```

结果：

| 指标 | 结果 |
| --- | --- |
| Guardian aggregate | positive |
| Prompt-seed split | 34 pass / 33 mixed / 33 fail |
| Prompt map | 15 stable pass / 15 mixed / 15 stable fail / 5 mostly pass/fail |

正式解释：

```text
The full-50 result supports prompt-stratified diagnosis: aggregate positivity can hide nearly balanced prompt-level heterogeneity.
```

## A9. 当前支持的新结论

当前实验支持：

```text
DPO can increase safety risk without reducing output diversity.
```

更完整地说：

```text
Preference optimization can shift the model toward a broader risky response region, producing distributed risk migration rather than stable dominant-mode collapse.
```

中文结论：

```text
偏好优化可能不会让模型重复同一种危险回答，但会让模型在多个不同回答模式中整体变得更有风险。
```

## A10. 下一步实验

下一步只围绕新 idea 做实验：

1. 在 Qwen3-1.7B 上扩大到 50 到 100 prompts。
2. 每 prompt 增加到 32 samples。
3. 计算 RDI 和 Risk Entropy。
4. 做 prompt-stratified 分析。
5. 在 Qwen3-4B 上复核关键 stratum。
6. 加入第二个安全分类器，检查 Granite 结论是否稳健。

下一步不再优先做：

- 证明 collapse；
- 低率 poison attack；
- 0.5B 新训练；
- 只看 dominant mode 的单一 PCE。

## B. 易懂版

### B1. 这个新 idea 是什么

一句话：

```text
DPO 可能不是让模型“重复同一种危险回答”，而是让模型“用很多不同方式变得更危险”。
```

更直白一点：

```text
训练后，模型不一定更死板，但可能更敢回答危险问题。它每次说法都不同，所以看起来没有重复；但安全裁判会发现，这些不同说法整体更危险。
```

### B2. 为什么这个问题重要

如果模型每次都重复同一句危险话，容易发现。

但如果模型每次都换一种说法，而且每种说法都有风险，就更隐蔽。

所以新研究不是问：

```text
模型有没有变得更重复？
```

而是问：

```text
模型有没有在保持多样性的同时，整体变得更危险？
```

### B3. 为什么要做 Qwen3-1.7B 实验

思路：

```text
要证明新 idea，首先要看一个比较新的模型：训练后是不是风险上升，但回答没有变得更重复。
```

Qwen3-1.7B 的结果很清楚：

- 风险大幅上升；
- 拒答大幅下降；
- 合规/行动性回答大幅上升；
- 但回答没有更固定；
- entropy 反而上升。

这说明：

```text
模型不是固定地变危险，而是分散地变危险。
```

### B4. 为什么要做 Qwen3-4B 实验

思路：

```text
一个模型可能是偶然，所以要换更大的模型看方向是否一样。
```

Qwen3-4B 的结果和 1.7B 同方向：

- Guardian 风险明显上升；
- determinism 下降；
- entropy 上升；
- collapse gate 失败。

这说明：

```text
新现象不是 1.7B 的偶然结果，至少在 Qwen3 1.7B 和 4B 上方向一致。
```

### B5. 为什么要做 held-out prompt 实验

思路：

```text
不能只在老题上看结果，要换新题检查。
```

S0.1 held-out 30 的结果：

- 新题上 Guardian 风险仍然有上升；
- 但 determinism 没有上升；
- entropy 没有下降；
- pass rate 只有 35%。

这说明：

```text
风险移动可以出现，但不是稳定坍缩。新方向应该研究风险怎么迁移，而不是继续硬证明坍缩。
```

### B6. 为什么要做 refusal-template control

思路：

```text
回答变固定不一定是坏事。如果固定成拒答，模型反而更安全。
```

结果：

- 模型更集中；
- 但拒答更多；
- 风险更低。

这说明：

```text
多样性和风险必须分开看。不能简单说“更固定就是更危险”。
```

### B7. 为什么要用 Granite Guardian

思路：

```text
需要一个安全裁判判断回答有没有风险，而且要确认它不是只看 prompt。
```

固定回答控制显示：

- 拒答回答：0/10 风险；
- 中性边界回答：1/10 风险；
- 明显不安全回答：10/10 风险；
- 短 compliance 模板：9/10 风险。

这说明：

```text
Granite Guardian 确实会根据 response 内容改变判断。
```

### B8. 新论文可以怎么讲

不要讲：

```text
我们发现 DPO 会让模型坍缩成稳定漏洞。
```

应该讲：

```text
我们发现 DPO 的安全风险不一定通过模式坍缩出现。更常见也更隐蔽的现象是：模型保持甚至增加输出多样性，但整体风险上升。
```

### B9. 给导师的短版

```text
老师，新的实验结果支持一个更稳的方向：DPO 不一定导致输出坍缩，但会在 Qwen3 上显著提高风险回答倾向，同时输出更多样。建议开题聚焦“偏好优化中的风险-多样性解耦诊断与预警”。
```

