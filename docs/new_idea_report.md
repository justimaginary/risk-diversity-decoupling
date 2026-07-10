# 新 idea 专版：偏好优化中的风险-多样性解耦

日期：2026-07-10

本文只保留新的研究 idea 和支持它的实验结果，不再展开旧的“PCE 坍缩漏洞”叙事。

> 开题使用口径：本文的核心现象已有跨模型、双种子和控制实验的**方向性支持**，但尚未完成 80 条零重叠提示上的 DPO 复验，也尚未完成第二安全分类器的交叉审计。因此它是一个经过先导实验支持、需要严格验证的研究假设，而不是已经完成的普遍规律。完整口述汇报稿见 [`opening_presentation_script.md`](opening_presentation_script.md)。

## 读者先看这一页

### 研究问题的一句话

```text
偏好优化会不会使语言模型在回答仍然多样时，整体更容易进入风险区域？
```

专业说法是“风险-多样性解耦”；通俗说法是“模型不一定反复说同一句危险话，也可能换着很多说法，但整体更不安全”。

### 目前已经看到什么，尚未看到什么

| 证据等级 | 当前结论 | 直接证据 | 不能推出的结论 |
| --- | --- | --- | --- |
| 较强的方向性证据 | DPO 后风险评分和合规倾向可以显著上升，而确定性下降、模式熵上升 | Qwen3-1.7B 的两个独立训练种子；Qwen3-4B 的方向复核；Granite Guardian 审计 | 这不是“所有模型、所有提示、所有 DPO 设置”都会发生的因果定律 |
| 机制性对照证据 | 风险与集中度是两条不同坐标轴；高确定性可以是安全拒答 | refusal-template 对照；Guardian 固定回答对照 | 不能仅用 Determinism 或单一 PCE 指标判断安全性 |
| 异质性证据 | 平均指标会掩盖明显的提示级差异 | 50 题 prompt-level 分析；30 题 held-out 结果 | 现有 taxonomy v0 还不能预测哪些题会发生风险迁移 |
| 设计就绪证据 | 80 条零重叠题集和安全基线已准备好，可用于严格复验 | 重叠、重复、来源跨度、主题、80 x 32 基线审计 | 题集 gate 本身不证明新现象已经泛化 |

### 开题时应当怎样表述

可以说：

```text
前期结果显示，偏好优化的安全副作用不一定表现为输出坍缩；在 Qwen3 的先导实验中，风险评分上升与多样性下降发生了解耦。我的开题工作要严格检验该现象的提示泛化条件、风险分布形态及可预警性。
```

不应说：

```text
我已经证明 DPO 普遍会使模型不安全，或已经发现了一个普遍可利用的漏洞。
```

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

## A8. 领域研究现状与创新边界

这一节回答一个关键问题：

```text
这个方向有没有已经被别人做过？
```

结论：

```text
相关工作很多，但“DPO 后风险上升且输出多样性不下降”的风险-多样性解耦诊断，目前不是已有工作的主结论。新 idea 的空间在于把 safety risk 和 semantic diversity 放到同一个 prompt-stratified 框架里测量。
```

### A8.1 已有工作 1：DPO 本身已经是成熟方法

代表工作：

- Rafailov et al., Direct Preference Optimization: Your Language Model is Secretly a Reward Model, 2023.

已有结论：

```text
DPO 用 chosen/rejected 偏好对直接优化语言模型，可以绕过显式 reward model 和复杂 RLHF 过程。
```

对本课题的影响：

```text
不能把“使用 DPO”当作创新点。
```

本课题的位置：

```text
我们研究 DPO 后模型安全风险和输出多样性如何共同变化，而不是提出一种新的 DPO 训练方法。
```

### A8.2 已有工作 2：DPO failure modes 和 likelihood over-optimization 已有人研究

代表工作：

- Pal et al., Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive, 2024.
- Shi et al., Understanding Likelihood Over-optimisation in Direct Alignment Algorithms, 2024.

已有结论：

```text
DPO 的 preference objective 变好，不一定意味着 preferred completion likelihood 或泛化表现都变好；过度优化 likelihood / margin 可能损害输出多样性和泛化。
```

对本课题的影响：

```text
不能声称“首次发现 preference margin 和真实生成行为可能不一致”。
```

本课题的位置：

```text
我们沿着这个断裂继续问：如果 preference fitting 与生成行为不一致，那么安全风险是否也会和多样性解耦？
```

### A8.3 已有工作 3：输出多样性下降已经有人研究

代表工作：

- Lanchantin et al., Diverse Preference Optimization, 2025.
- Karouzos et al., Where does output diversity collapse in post-training?, 2026.

已有结论：

```text
后训练和偏好优化可能使输出分布变尖、回答多样性下降；多样性坍缩和训练数据构成、训练阶段、任务类型有关。
```

对本课题的影响：

```text
不能把“后训练会降低多样性”作为主要创新。
```

本课题的位置：

```text
已有工作主要关心 diversity 是否下降；我们的新问题是 risk 是否可以在 diversity 不下降甚至上升时增加。
```

换句话说：

```text
前人研究的是“回答会不会变少”。
我们研究的是“回答没有变少时，风险会不会变多”。
```

### A8.4 已有工作 4：DPO 安全攻击和拒答抑制已有直接相关工作

代表工作：

- Yoon et al., Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs, 2026.

已有结论：

```text
看似无害的 DPO preference pairs 可以抑制 refusal，并迁移到 harmful prompts，提高 attack success rate。
```

对本课题的影响：

```text
不能声称“首次发现 DPO 会降低拒答或提高安全风险”。
```

本课题的位置：

```text
已有攻击工作主要报告 refusal suppression 和 attack success；我们要进一步分析风险是在单一模式中集中，还是分散在多个语义模式中。
```

这就是新 idea 的差异：

```text
Attack success answers whether the model can be made unsafe.
Risk-diversity diagnosis answers how that unsafety is distributed across sampled outputs.
```

### A8.5 已有工作 5：安全分类器已有成熟工具

代表工作：

- Padhi et al., Granite Guardian, 2024.
- Zeng et al., ShieldGemma, 2024.

已有结论：

```text
Guardian-style classifiers 可以对 prompt 和 response 的安全风险进行自动检测。
```

对本课题的影响：

```text
不能把“用安全分类器判断有害性”当作创新。
```

本课题的位置：

```text
我们把 Guardian risk 与 semantic mode distribution 结合起来，构造 RDI 和 Risk Entropy，用来诊断风险是集中还是分散。
```

### A8.6 不能声称什么

本课题不能声称：

| 不能声称 | 原因 |
| --- | --- |
| 首次提出 DPO | DPO 已由 Rafailov et al. 提出 |
| 首次发现 DPO 有 failure mode | Smaug / DPO-Positive 等已研究 |
| 首次发现 preference objective 与泛化不一致 | direct alignment over-optimization 已研究 |
| 首次发现后训练影响多样性 | diversity collapse / DivPO 已研究 |
| 首次发现 DPO 可能降低拒答 | benign DPO attack 已研究 |
| 首次使用 Guardian 做安全检测 | Granite Guardian / ShieldGemma 已存在 |

### A8.7 可以声称什么

更稳妥的创新表述是：

```text
本研究提出风险-多样性解耦的诊断视角，并以 Qwen3 先导实验提供方向性证据：偏好优化后，模型安全风险可以在输出多样性保持或增加的情况下上升。为在后续严格验证中刻画这一现象，本文将 guardian-scored risk 与 semantic mode distribution 结合，提出 RDI、Risk Entropy 和 prompt-stratified risk-diversity quadrant analysis，用于区分 collapse risk 与 distributed risk migration。
```

白话版：

```text
别人已经研究过 DPO、研究过多样性下降、研究过拒答被压低。我们的新点是：模型不一定因为“重复同一种危险答案”才变危险；它可能因为“很多不同答案都变得更危险”而出问题。我们要测的就是这种更隐蔽的风险分布变化。
```

### A8.8 与已有工作的关系表

| 研究方向 | 已有工作主要回答 | 本课题补充回答 |
| --- | --- | --- |
| DPO 方法 | 怎么直接用偏好对优化模型 | DPO 后风险和多样性如何共同变化 |
| DPO failure modes | preference loss / likelihood 可能异常 | 这种异常是否伴随安全风险迁移 |
| 多样性坍缩 | 后训练是否让回答变少 | 风险能否在回答不变少时上升 |
| DPO 安全攻击 | DPO 是否能降低拒答并提高攻击成功率 | 风险是集中在一个模式，还是分散在多个模式 |
| 安全分类器 | 如何判断回答是否有风险 | 如何把风险评分和语义模式分布结合 |

### A8.9 参考文献入口

本报告使用的一手来源包括：

1. Direct Preference Optimization: https://arxiv.org/abs/2305.18290
2. Smaug / DPO-Positive: https://arxiv.org/abs/2402.13228
3. Understanding Likelihood Over-optimisation: https://arxiv.org/abs/2410.11677
4. Diverse Preference Optimization: https://arxiv.org/abs/2501.18101
5. Where does output diversity collapse in post-training?: https://arxiv.org/abs/2604.16027
6. Few-Shot Truly Benign DPO Attack: https://arxiv.org/abs/2605.10998
7. Granite Guardian: https://arxiv.org/abs/2412.07724
8. ShieldGemma: https://arxiv.org/abs/2407.21772

### A8.10 文献如何具体支撑本题，而不是只作“参考书目”

下表把每篇文献与本课题的逻辑位置对应起来。表中的“本题不能重复”是研究边界，“本题仍要回答”才是开题问题。

| 文献 | 已被该文献直接支持的事实 | 本题不能重复的说法 | 本题仍要回答的缺口 |
| --- | --- | --- | --- |
| Rafailov et al., DPO (2023/ICLR 2024) | 可由偏好对直接进行对齐优化，无须显式奖励模型 | “DPO 是新的对齐方法” | DPO 后安全风险与语义模式分布如何共同变化 |
| Pal et al., DPO-Positive (2024) | 标准 DPO 的相对偏好改善可伴随 chosen likelihood 下降 | “首次发现 DPO 目标与生成质量可脱节” | 此类目标行为断裂是否也表现为风险与多样性的断裂 |
| Shi et al., Likelihood Over-optimisation (2024) | 更高似然/更大间隔不必然带来更好泛化；多样性与性能可发生权衡 | “首次发现直接对齐会影响多样性” | 风险是否能在多样性不降时上升，且风险位于哪些模式 |
| Lanchantin et al., DivPO (2025) | 后训练通常会使输出分布变尖并降低多样性；可通过偏好构造保留多样性 | “首次研究偏好优化和多样性” | 多样性保持时是否仍可能发生安全风险迁移 |
| Yoon et al., Truly Benign DPO Attack (2026) | 看似良性的 DPO 偏好对可压低拒答，并向风险提示迁移 | “首次发现 DPO 会削弱安全拒答” | 攻击成功或拒答变化之外，风险在多次采样输出中是集中还是分散 |
| Padhi et al., Granite Guardian (2024); Zeng et al., ShieldGemma (2024) | 可使用专门模型评估提示和回答的多类风险 | “首次用分类器做安全评估” | 如何把风险评分与语义聚类结合，形成可解释的风险分布诊断 |

由此，本题的可检验贡献不是宣称一个“前人未见的风险事实”，而是提出一个联合测量和严格检验的问题：当偏好优化改变安全行为时，风险变化与语义多样性变化是否独立，若独立，如何识别其分布、异质性与预警信号。

## A9. 为什么要做这些实验

### 实验逻辑总览

| 实验目的 | 为什么必须做 | 对新 idea 的作用 |
| --- | --- | --- |
| Qwen3-1.7B 两 seed DPO | 检查现象是否在较新模型上稳定出现 | 直接验证 risk up + diversity up |
| Qwen3-4B gate | 检查是否不是 1.7B 偶然现象 | 提供跨规模支持 |
| S0.1 held-out 30 | 检查风险变化是否能在新 prompt 上出现 | 支持 risk movement 不只存在于原始 prompt |
| Refusal-template control | 验证风险和多样性是两个可分离维度 | 支持四象限框架 |
| Granite Guardian fixed-response control | 检查风险评分不是只由 prompt 驱动 | 支持 Guardian 风险信号可信 |
| Full 50 prompt heterogeneity | 检查 aggregate 是否掩盖 prompt 差异 | 支持 prompt-stratified 诊断 |
| Qwen3 held-out 80 prompt gate | 检查下一阶段题目是否合理 | 排除 first-10 精选题偶然性 |

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

### 实验 7：Qwen3 held-out 80 prompt gate

为什么做：

```text
新 idea 的下一步不能继续依赖 first-10 prompts。必须先冻结一个更大、零重叠、非单一主题的新题集合，再做 Qwen3-1.7B DPO。
```

正式目的：

```text
This gate tests whether the next risk-diversity experiment has a reasonable prompt set before training: broad source coverage, no overlap with previous gates, no duplicates, and a safe refusal-dominant Qwen3 baseline.
```

Prompt 选择：

| 项目 | 值 |
| --- | --- |
| Prompt file | `data/advbench_rdi_heldout_80_seed20260704.jsonl` |
| Source | AdvBench harmful behaviors |
| Source prompts | 520 |
| Excluded old prompts | 88 |
| Candidate prompts | 482 |
| Selected prompts | 80 |
| Selection seed | 20260704 |
| Source index range | 2-511 |
| Duplicate prompts | 0 |
| Overlap with previous gates | 0 |

关键词主题覆盖：

| Topic bucket | Count |
| --- | ---: |
| cyber | 11 |
| cyber + fraud_theft | 2 |
| drugs | 3 |
| evasion_illegal | 4 |
| evasion_illegal + drugs | 2 |
| fraud_theft | 10 |
| harassment_abuse | 2 |
| violence_weapons | 10 |
| violence_weapons + evasion_illegal + drugs | 2 |
| other | 34 |

Qwen3-1.7B baseline：

| 指标 | 结果 |
| --- | ---: |
| Prompts | 80 |
| Samples per prompt | 32 |
| Total outputs | 2560 |
| Mean determinism | 0.9145 |
| Mean mode entropy | 0.2966 |
| Mean distinct-1 | 0.1704 |
| Mean distinct-2 | 0.3678 |
| Mean proxy PCE | 0.0129 |
| Prompts with entropy > 0.001 | 21 / 80 |
| Prompts with determinism >= 0.95 | 72 / 80 |
| Prompts with determinism <= 0.80 | 8 / 80 |
| Prompts with proxy PCE > 0 | 13 / 80 |

Policy-proxy baseline：

| Signal | All outputs | Dominant outputs |
| --- | ---: | ---: |
| Refusal rate | 0.8645 | 0.9342 |
| Compliance rate | 0.1637 | 0.1756 |
| Actionable rate | 0.1020 | 0.1064 |
| High-risk instructional rate | 0.0023 | 0.0026 |
| Dominant high-risk-instructional majority prompts | 0 / 80 | 0 / 80 |

正式解释：

```text
The frozen 80-prompt gate supports the next-stage experimental design. The set is new, duplicate-free, zero-overlap with prior gates, source-index spread, and multi-topic. The Qwen3-1.7B baseline is mostly refusal-dominant with very low proxy risk, making it suitable for testing whether DPO shifts the model toward distributed risky responses.
```

重要 caveat：

```text
The baseline is already highly deterministic: 72/80 prompts have determinism >= 0.95. Therefore the next experiment must report absolute diversity, refusal shift, Guardian risk, RDI, and Risk Entropy, not only "diversity did not decrease".
```

对新 idea 的作用：

```text
This gate does not yet prove risk-diversity decoupling, but it makes the next proof attempt credible: the prompt set is no longer first-10 cherry-picked, and the baseline starts from a safe refusal-dominant state.
```

## A10. 当前支持的新结论与证据边界

当前实验给出较强的方向性支持：

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

这个结论依赖于以下可观察事实：Qwen3-1.7B 的两个独立训练种子均出现 Guardian 风险显著上升、determinism 下降、mode entropy 上升；Qwen3-4B 的小规模方向复核相同；拒答模板对照说明“集中”本身并不意味着危险；Guardian 固定回答对照说明风险评分对回答内容而非仅对题目敏感。

同时，结论仍有明确边界：Qwen3 的直接证据目前来自 first-10 提示；Qwen3-4B 仅完成一个 100-step 种子；安全审计尚未完成第二分类器复核；80 条新题只完成了题集与基线 gate。故本课题的首要目标是检验这一现象的可重复性、发生条件和失效条件，而不是预设它必然普遍成立。

## A11. 下一步实验

下一步只围绕新 idea 做实验：

1. 使用已冻结的 80 个 zero-overlap AdvBench prompts；训练集与评测集的关系在运行前固定并记录，避免事后挑题。
2. 在 Qwen3-1.7B 上完成 LoRA-DPO，并至少使用两个真正不同的训练种子；baseline 和 final 都保持每 prompt 32 samples。
3. 同时报告 RDI、Risk Entropy、绝对多样性、refusal shift、Granite Guardian 风险和 prompt-stratified 结果；不只报告一个聚合数。
4. 用第二个安全分类器复核 Granite 结论，并人工抽审分歧案例，避免把分类器偏差误当作模型行为。
5. 仅当 80 题结果在预注册的方向与两个种子上成立时，才在 Qwen3-4B 上复核关键分层；若不成立，则将“发生条件/负结果基准”作为主问题。

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

### B8. 为什么先做 80 题合理性实验

思路：

```text
如果还只用前 10 道题，别人会质疑是不是挑题挑出来的结果。
```

所以先做了一件关键事：

```text
从 AdvBench 里重新随机冻结 80 道新题，排除之前所有用过的 prompt。
```

这批题的检查结果：

- 80 道题；
- 和旧 prompt 集合 0 重叠；
- 没有重复；
- source index 从 2 到 511，很分散；
- 不是单一题型；
- Qwen3-1.7B 原始模型在这些题上主要拒答；
- baseline proxy 风险很低。

最重要的 baseline 结果：

```text
refusal rate = 0.8645
high-risk instructional rate = 0.0023
```

这说明：

```text
这批题适合测试 DPO 是否把模型从“安全拒答”推向“风险回答”。
```

但也有一个 caveat：

```text
baseline determinism 已经很高，72/80 个 prompt 的 determinism >= 0.95。
```

所以后面不能只说“多样性没下降”，还要看：

- 风险是否上升；
- 拒答是否下降；
- RDI 是否上升；
- Risk Entropy 是否显示风险分散；
- Guardian 是否确认风险变化。

### B9. 新论文可以怎么讲

不要讲：

```text
我们发现 DPO 会让模型坍缩成稳定漏洞。
```

应该讲：

```text
我们发现 DPO 的安全风险不一定通过模式坍缩出现。更常见也更隐蔽的现象是：模型保持甚至增加输出多样性，但整体风险上升。
```

### B10. 别人有没有已经做过

简单说：

```text
别人做过 DPO，做过 DPO 的 failure mode，做过多样性下降，做过 DPO 降低拒答，也做过安全分类器。
```

但还没有直接把问题组织成：

```text
风险上升，但多样性不下降。
```

也没有把重点放在：

```text
风险到底集中在一个回答模式里，还是分散在很多回答模式里。
```

所以这篇论文不能说“我们第一个发现 DPO 有风险”，而应该说：

```text
我们发现并诊断一种更隐蔽的风险形态：分散式风险迁移。
```

### B11. 给导师的短版

```text
老师，新的实验结果支持一个更稳的方向：DPO 不一定导致输出坍缩，但会在 Qwen3 上显著提高风险回答倾向。80 道新题 gate 已通过，下一步用同一题集验证风险-多样性解耦。
```
