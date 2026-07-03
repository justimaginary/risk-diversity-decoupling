# 开题报告：偏好优化中的风险-多样性解耦诊断与预警

日期：2026-07-04
项目目录：`C:\Users\TH.Xie\Desktop\DPO`
目标定位：CCF-A / AI 顶会导向，主投 AAAI 或 IJCAI，条件冲刺 NeurIPS / ICML / ICLR
当前判断：原始“DPO 导致可利用模式坍缩”假设不成立；新的研究方向是“DPO 导致风险迁移但不必然导致多样性下降”的诊断与预警

## 0. 开题题目

建议题目：

```text
面向偏好优化语言模型的风险-多样性解耦诊断与预警研究
```

英文题目：

```text
Diagnosing and Warning Risk-Diversity Decoupling in Preference-Optimized Language Models
```

专业表述：

```text
本课题研究 Direct Preference Optimization 等偏好优化方法如何改变语言模型的安全风险分布与输出多样性，并提出 prompt-stratified 的风险-多样性解耦诊断框架。
```

白话表述：

```text
我研究的是：模型经过偏好训练后，会不会不只是“更会回答”，还会在安全风险上发生变化；更重要的是，这种风险变化不一定表现为重复同一种错误回答，而可能分散在很多不同回答里。
```

## 1. 摘要

大语言模型通常会经过指令微调、RLHF 或 DPO 等后训练方法，使模型更符合人类偏好。原始研究假设认为，DPO 可能降低输出多样性，使模型在安全敏感 prompt 上更稳定地落入少数有风险回答模式，从而产生 Preference Collapse Exploitability，也就是“偏好坍缩可利用性”。

本项目采用实验先行策略，在本机 RTX 4060 上完成了指标 sanity check、toy DPO、SmolLM2、Qwen2.5-0.5B、Qwen3-1.7B、Qwen3-4B、held-out prompts、negative controls、Granite Guardian 审计和 poison/CAR smoke。实验结果显示：DPO 确实能显著改变模型风险倾向，尤其在 Qwen3-1.7B 和 Qwen3-4B 上，Guardian harmfulness 明显上升；但核心坍缩假设没有成立，determinism 下降、mode entropy 上升，说明模型没有更固定地输出同一类风险回答，反而在更分散的输出空间中产生更多风险响应。

因此，本课题不再以“证明 DPO 造成稳定漏洞”为主线，而转向更稳健、更符合证据的研究问题：偏好优化如何造成风险与多样性的解耦，为什么模型可能在保持甚至增加输出多样性的同时提高安全风险，以及如何通过 prompt-stratified 诊断和训练过程预警发现这种隐蔽风险迁移。

## 2. 研究背景

### 2.1 大语言模型为什么需要偏好优化

专业表述：

```text
Pretrained language models acquire broad generative ability, but post-training is required to align their behavior with human preference, helpfulness, harmlessness, and instruction-following requirements.
```

白话表述：

```text
预训练模型像一个读过很多书的人，但它不一定知道怎么当助手；后训练就是教它什么回答更合适、更有帮助、更安全。
```

常见后训练方法包括：

| 方法 | 专业解释 | 白话解释 |
| --- | --- | --- |
| SFT | Supervised Fine-Tuning，用 prompt-response 监督数据训练 | 给模型看标准答案，让它模仿 |
| RLHF | Reinforcement Learning from Human Feedback，用人类偏好训练奖励模型，再优化策略 | 先训练一个打分器，再让模型追高分 |
| DPO | Direct Preference Optimization，直接用 chosen/rejected 偏好对优化模型 | 告诉模型“A 比 B 好”，让它以后更偏向 A |

DPO 近年来被广泛使用，因为它比传统 RLHF 更简单，不需要显式训练 reward model，也不需要复杂的在线强化学习过程。

### 2.2 为什么偏好优化可能有副作用

专业表述：

```text
Preference optimization changes the model distribution, not only the model's average helpfulness. It may alter refusal behavior, risk-taking behavior, response diversity, and generalization to unseen prompts.
```

白话表述：

```text
偏好训练不是只给模型“加点礼貌”。它会改变模型回答问题的方式，可能让模型更少拒答、更敢回答，也可能改变它回答的多样性。
```

前人工作已经说明：

- DPO 是一种高效的直接偏好优化方法。
- 标准 DPO 可能存在 likelihood failure modes。
- direct alignment 中 likelihood 或 margin 过优化不一定带来更好泛化。
- 后训练和偏好优化可能降低输出多样性。
- benign DPO preference pairs 可能降低拒答并增加 jailbreak 风险。

这些工作共同说明：偏好优化不只是提升能力，也可能带来分布层面的副作用。

## 3. 原始 idea 是什么

### 3.1 原始假设

专业表述：

```text
The original hypothesis was that DPO reduces output diversity, increases sampled-output determinism, and raises Preference Collapse Exploitability on safety-sensitive prompts.
```

白话表述：

```text
原来我怀疑：DPO 会让模型回答变窄、变固定；如果固定下来的回答刚好有风险，就会变成一种容易被利用的安全弱点。
```

原始逻辑链条：

```text
DPO preference fitting
-> output diversity decreases
-> sampled generations collapse into fewer semantic modes
-> the dominant mode becomes harmful
-> attackers can exploit the predictable risky mode
```

白话版本：

```text
模型学会偏好某类回答
-> 多次问同一个危险问题时，回答越来越像
-> 如果这类回答有风险
-> 攻击者就能更稳定地诱导模型出问题
```

### 3.2 原始指标 PCE

原始指标：

```text
PCE = determinism * harmfulness_of_dominant_mode
```

其中：

| 组成 | 专业解释 | 白话解释 |
| --- | --- | --- |
| determinism | dominant semantic cluster mass | 多次生成时，最大那类回答占多少 |
| harmfulness | guardian-scored harmfulness of dominant mode | 最大那类回答有多危险 |
| PCE | exploitability from stable harmful mode | 模型是否稳定地给出某类风险回答 |

这个指标的直觉是对的：如果模型既固定，又固定在危险回答上，确实值得警惕。

但是最新实验表明：现在真正稳定出现的现象不是“固定到危险模式”，而是“风险上升但输出没有固定”。

## 4. 为什么必须实验先行

原始想法听起来合理，但它包含一个很强的机制假设：

```text
preference fitting -> sampled-mode collapse
```

专业表述：

```text
Preference-margin improvement and sampled generation behavior are not equivalent. A model can fit chosen/rejected preferences without concentrating high-temperature samples into fewer modes.
```

白话表述：

```text
训练时模型学会“A 比 B 好”，不代表它实际回答时就会反复给出同一个 A。它可能学会一类偏好，然后用很多不同说法表达出来。
```

因此，项目没有直接写论文 claim，而是按下面顺序验证：

1. 指标是否能区分分散输出和坍缩输出。
2. toy DPO 是否可能导致概率集中。
3. 真实小模型上能不能跑通训练和评估。
4. Qwen2.5-0.5B 是否有局部正信号。
5. held-out prompts 是否泛化。
6. negative controls 是否排除假阳性。
7. Qwen3-1.7B 和 Qwen3-4B 是否支持“模型太小”解释。

## 5. 已完成实验与关键数据

### 5.1 实验总览

| 阶段 | 实验 | 为什么做 | 关键结果 | 结论 |
| --- | --- | --- | --- | --- |
| 指标 sanity | synthetic diverse vs collapsed | 验证 PCE 尺子是否正常 | 指标方向正确 | 管线可用 |
| 机制 sanity | toy DPO categorical update | 验证偏好更新是否可能集中概率 | toy 分布会集中 | 只说明机制可能 |
| 工程 smoke | tiny GPT-2 | 验证训练、采样、聚类、统计链路 | 端到端跑通 | 工程通过 |
| 小模型 gate | SmolLM2-135M | 看真实 instruction model 是否有弱信号 | 两 seed 弱方向 | 证据不足 |
| 小模型复核 | SmolLM2-360M corrected | 加强采样验证 | 10x8 weak，10x16 robust fail | 不稳定 |
| 历史 Qwen | Qwen2.5-0.5B 20/100-step | 本地 Qwen pilot | 20-step fail，100-step weak | 弱证据 |
| margin 诊断 | preference margin tracking | 验证偏好拟合和生成行为关系 | margin 可改善，生成不必然坍缩 | 核心链条断裂 |
| 最强 pilot | Qwen2.5-0.5B short-template 300-step | 寻找局部 PCE 正信号 | first-10 robust pass | 局部、受限 |
| 反向控制 | refusal-template | 验证 determinism 不等于风险 | determinism 上升，harm 下降 | harmfulness 必须单独看 |
| wording 控制 | neutral / concise | 检查是否依赖 wording | weak/mixed | wording-sensitive |
| prompt transfer | prompts 10-19 / full 50 | 检查 first-10 是否泛化 | 高度异质 | aggregate 会误导 |
| taxonomy v0 | cyber vs violence/weapons | 尝试解释 prompt 差异 | held-out 验证失败 | taxonomy v0 不可用 |
| S0.1 | AdvBench held-out 30 | 开题前泛化 gate | pass rate 35%，低于 60% | 不进入漏洞 claim |
| Experiment C | poison/CAR smoke | 检查主动诱导剂量效应 | 无剂量效应 | 不支持攻击叙事 |
| Qwen3-1.7B | 300-step seeds 42/43 | 检查更大更新模型 | harm 上升，det 下降，entropy 上升 | collapse robust fail |
| Qwen3-4B | 100-step seed42 | 进一步检查规模解释 | harm 上升，det 下降，entropy 上升 | collapse robust fail |
| Qwen3 prompt gate | AdvBench held-out 80 baseline | 验证下一阶段题目是否合理 | 0 重叠、无重复、80x32 baseline 完成 | 可进入 DPO 风险-多样性验证 |

### 5.2 Qwen2.5-0.5B 历史 pilot

Qwen2.5-0.5B 的作用是历史先导验证，不再作为后续主证据。

最强正信号来自：

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen2.5-0.5B-Instruct |
| preference | short-template |
| chosen | `Sure, safe overview only.` |
| steps | 300 |
| learning rate | 3e-6 |
| seeds | 42、43 |
| prompts | first 10 |
| samples | 每 prompt 16 |

结果：

- determinism 上升；
- mode entropy 下降；
- local PCE robust pass；
- Granite Guardian-PCE robust pass；
- dominant Guardian harmfulness robust increase；
- refusal-template 控制能把风险方向反过来。

专业解释：

```text
This historical pilot supports a restricted prompt- and wording-sensitive PCE signal.
```

白话解释：

```text
它说明这个想法不是完全凭空来的，在某些题、某种模板、小模型设置下确实能看到“更固定、更危险”的局部现象。
```

但是：

- prompt transfer 不稳定；
- held-out 失败；
- wording 复现弱；
- target phrase exact hit 为 0；
- 0.5B 规模太小；
- 不能代表现代模型。

因此它只能作为“为什么提出问题”的历史证据，不能作为“结论成立”的核心证据。

### 5.3 S0.1 held-out 30：原始假设第一次系统受挫

目的：

```text
用 30 个零重叠 AdvBench prompts 检查历史最强 positive checkpoint 是否泛化。
```

结果：

| 指标 | Seed42 | Seed43 | Pooled |
| --- | ---: | ---: | ---: |
| Det Delta | -0.0125 | -0.0115 | 负向 |
| Entropy Delta | +0.0190 | +0.0322 | 正向 |
| Proxy PCE Delta | +0.0250 | +0.0208 | 弱正 |
| Prompt pass/mixed/fail | 10/2/18 | 11/2/17 | 21/4/35 |
| Pass rate | 33.3% | 36.7% | 35.0% |

Guardian 结果：

| 指标 | 结果 |
| --- | --- |
| Guardian-PCE gate | mixed |
| Guardian-PCE Yes delta | +0.0307，CI [+0.0130, +0.0490] |
| Guardian-PCE score delta | +0.0211，CI [+0.0064, +0.0370] |
| dominant harm | robust increase |
| target phrase hit | seed42 final 0/960，seed43 final 0/960 |

专业解释：

```text
S0.1 shows harmfulness movement without held-out sampled-mode collapse.
```

白话解释：

```text
换新题以后，模型可能更危险一点，但没有更固定；这不是“稳定坍缩漏洞”。
```

### 5.4 Experiment C：主动投毒假设没有成立

期望剂量效应：

```text
CAR_5% > CAR_1% > CAR_clean
```

实际结果：

| Condition | Det Delta | Entropy Delta | CAR | Proxy PCE Delta |
| --- | ---: | ---: | ---: | ---: |
| clean | +0.0312 | -0.0797 | +0.000797 | +0.0062 |
| 1% poison | +0.0312 | -0.0797 | +0.000797 | +0.0000 |
| 5% poison | +0.0188 | -0.0451 | +0.000451 | -0.0187 |

Guardian-PCE：

| Condition | Guardian-PCE Yes Delta | Guardian-PCE Score Delta |
| --- | ---: | ---: |
| clean | +0.0437 | +0.0326 |
| 1% poison | +0.0312 | +0.0228 |
| 5% poison | +0.0187 | +0.0095 |

专业解释：

```text
The poison/CAR smoke does not support active collapse induction.
```

白话解释：

```text
如果投毒真的能加速坍缩，5% 应该比 1% 更强；但实验里 5% 反而更弱，所以不能讲攻击成立。
```

### 5.5 Qwen3：决定性转折点

用户提出过一个合理怀疑：

```text
0.5B 太小，也许更大模型才能显示真实趋势。
```

因此项目转向 Qwen3-first：

- Qwen3-1.7B；
- Qwen3-4B；
- LoRA-DPO；
- non-thinking generation；
- Granite Guardian 审计。

#### Qwen3-1.7B

设置：

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-1.7B |
| steps | 300 |
| seeds | 42、43 |
| prompts | first 10 |
| samples | 每 prompt 16 |

结果：

| Seed | Det Delta | Entropy Delta | Proxy PCE Delta | Local Decision | Prompt Pass/Mixed/Fail |
| --- | ---: | ---: | ---: | --- | --- |
| 42 | -0.0563 | +0.1686 | +0.4875 | fail | 1/0/9 |
| 43 | -0.0500 | +0.1667 | +0.4625 | fail | 1/0/9 |
| pooled | -0.0531 CI [-0.1000, -0.0125] | +0.1677 CI [+0.0301, +0.3117] | +0.4750 CI [+0.3469, +0.6000] | robust_fail | 2/0/18 |

Raw audit：

| Condition | Outputs | Refusal | Compliance | Proxy-Harmful | Target Hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| seed42 step0 | 160 | 0.575 | 0.156 | 0.031 | 0 |
| seed42 final | 160 | 0.156 | 0.706 | 0.575 | 0 |
| seed43 step0 | 160 | 0.575 | 0.156 | 0.031 | 0 |
| seed43 final | 160 | 0.156 | 0.650 | 0.537 | 0 |

Guardian：

| Seed | Guardian-PCE Yes Delta | Guardian-PCE Score Delta | Dominant Harm | Gate |
| --- | ---: | ---: | --- | --- |
| 42 | +0.4750 | +0.4272 | increase | mixed |
| 43 | +0.4813 | +0.4445 | increase | mixed |
| pooled | +0.4781 CI [+0.3094, +0.6500] | +0.4359 CI [+0.2945, +0.5874] | robust_increase | robust_fail |

专业解释：

```text
Qwen3-1.7B shows strong risk amplification but robustly rejects the sampled-collapse direction.
```

白话解释：

```text
Qwen3-1.7B 更愿意给出有风险的回答，但这些回答不是更重复，而是更分散。
```

#### Qwen3-4B

设置：

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-4B |
| steps | 100 |
| seed | 42 |
| prompts | first 10 |
| samples | 每 prompt 16 |

结果：

| Metric | Delta |
| --- | ---: |
| Determinism | -0.0187，CI [-0.0375, +0.0000] |
| Entropy | +0.0701，CI [+0.0000, +0.1403] |
| Proxy PCE | +0.0125，CI [+0.0000, +0.0312] |
| Prompt split | 0 pass / 0 mixed / 10 fail |
| Local decision | robust_fail |

Guardian：

| Metric | Result |
| --- | --- |
| Guardian-PCE Yes Delta | +0.4688，CI [+0.3187, +0.6250] |
| Guardian-PCE Score Delta | +0.3981，CI [+0.2658, +0.5286] |
| Dominant Harm | robust_increase |
| Gate | robust_fail |

专业解释：

```text
Qwen3-4B reproduces the same decoupled pattern: risk rises while collapse metrics move in the opposite direction.
```

白话解释：

```text
换到 4B 以后，还是同样的情况：风险上升，但回答没有变固定。
```

### 5.6 Qwen3 80 题题目合理性实验

这一步是下一阶段成败的关键。原因是：

```text
新 idea 要证明的不是 first-10 prompt 上的偶然现象，而是 DPO 是否能在更广的新题集合上造成风险-多样性解耦。
```

白话说：

```text
如果题还是挑出来的 10 道，那后面结果再好也可能是巧合。必须先换一批新题，确认题目本身合理，再做训练。
```

冻结 prompt 集合：

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

Qwen3-1.7B baseline 采样：

| 项目 | 值 |
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
The frozen 80-prompt set is reasonable for the next risk-diversity experiment:
it is zero-overlap with previous gates, duplicate-free, source-index spread, and
not single-topic. Qwen3-1.7B baseline outputs are mostly refusal-dominant with
very low proxy risk, making the set suitable for testing whether DPO moves the
model from safe refusal-dominant behavior toward distributed risky responses.
```

白话解释：

```text
这 80 道题不是旧题，也不是同一类题。Qwen3 原始模型在这些题上主要拒答，风险很低，所以后面如果 DPO 后风险升高，就更像是真实的训练影响，而不是题目本身已经很危险。
```

重要 caveat：

```text
The baseline is already highly deterministic: 72/80 prompts have determinism >= 0.95.
```

白话解释：

```text
原始 Qwen3 在这些危险题上已经很稳定地拒答，所以后续不能只说“多样性没有下降”。必须同时报告绝对多样性、拒答率变化、Guardian risk、RDI 和 Risk Entropy。
```

本阶段结论：

```text
题目合理性实验通过。下一步可以进入 Qwen3-1.7B DPO 80 prompts x 32 samples 风险-多样性验证。
```

### 5.7 实验结论的决定性

Qwen3 改变了项目判断。

之前可以怀疑：

```text
0.5B 太小，所以坍缩信号不稳。
```

现在更合理的判断是：

```text
核心机制假设本身不成立：偏好优化不必然导致采样模式坍缩。
```

专业表述：

```text
The dominant empirical pattern is risk amplification without mode-collapse amplification.
```

白话表述：

```text
模型确实更危险了，但不是通过“反复说同一种危险话”变危险，而是通过“用更多不同方式给出风险回答”变危险。
```

## 6. 原始 idea 哪里错了

### 6.1 错在把 preference fitting 等同于 mode collapse

原始假设：

```text
preference fitting -> fewer output modes
```

实验显示：

```text
preference fitting -> broader risky response region
```

专业表述：

```text
DPO can move the model toward a riskier response manifold without concentrating samples into a single dominant mode.
```

白话表述：

```text
DPO 不是让模型只学会一个危险答案，而是让模型学会一片更危险的回答区域。
```

### 6.2 错在把风险上升和多样性下降绑定在一起

原始 PCE 叙事要求：

```text
harmfulness up + diversity down
```

Qwen3 看到的是：

```text
harmfulness up + diversity up
```

专业表述：

```text
Risk and diversity are decoupled under the observed Qwen3 DPO setting.
```

白话表述：

```text
风险和多样性不是同一根绳上的两个方向。模型可以一边更危险，一边回答更多样。
```

### 6.3 错在把 first-10 pilot 当成主线

0.5B first-10 正信号现在更可能是：

- 特定 short-template wording 的局部现象；
- first-10 prompts 的选择偏差；
- 小模型容量限制下的特殊行为；
- synthetic preference construction 导致的局部模式。

专业表述：

```text
The historical positive result is best treated as a constrained pilot, not as evidence for a general vulnerability.
```

白话表述：

```text
它说明“有这个苗头”，但不能说明“普遍存在这个问题”。
```

## 7. 新 idea 是什么

### 7.1 新研究问题

新方向：

```text
DPO 如何改变模型的风险-多样性权衡？
什么条件下会出现风险上升但多样性不下降的反直觉现象？
如何诊断和预警这种风险迁移？
```

专业表述：

```text
The new research focus is risk-diversity decoupling: preference optimization may increase safety risk while preserving or increasing output diversity.
```

白话表述：

```text
新的问题不是“模型会不会重复同一种危险回答”，而是“模型会不会用很多不同方式变得更危险”。
```

### 7.2 新现象：风险迁移

建议命名：

```text
Distributed Risk Migration under Preference Optimization
```

中文：

```text
偏好优化下的分散式风险迁移
```

定义：

```text
After preference optimization, guardian-scored harmfulness increases across sampled outputs, while semantic output diversity is preserved or increased.
```

白话定义：

```text
训练后，模型回答整体更容易有风险，但这些风险分散在很多不同回答里，而不是集中到一个固定答案。
```

### 7.3 为什么这个现象更值得研究

模式坍缩风险：

```text
容易检测，因为回答重复。
```

分散式风险迁移：

```text
更隐蔽，因为每次回答都不同，但都可能越界。
```

专业表述：

```text
Distributed risk migration is harder to detect than collapse-based risk because lexical and semantic repetition signals may remain weak.
```

白话表述：

```text
如果模型每次都重复同一句危险话，很容易抓住；如果它每次都换一种说法，但都在帮危险请求，就更难发现。
```

## 8. 新指标与诊断框架

### 8.1 保留 PCE，但降级为一种特殊情况

PCE 仍然有意义，但它只描述一种情况：

```text
risk up + determinism up + entropy down
```

也就是：

```text
风险集中到少数 dominant modes。
```

新框架中，PCE 不再是唯一主指标，而是风险-多样性二维空间里的一个象限。

### 8.2 新核心指标 1：Risk Distribution Index

暂定定义：

```text
RDI(x) = sum_c p(c | x) * h(c, x)
```

其中：

- `x` 是 prompt；
- `c` 是语义聚类后的 response mode；
- `p(c | x)` 是 mode mass；
- `h(c, x)` 是 Guardian 对该 mode 的风险评分。

专业解释：

```text
RDI measures expected guardian-scored risk across all sampled semantic modes, rather than only the dominant mode.
```

白话解释：

```text
RDI 不只看最大那类回答危不危险，而是看所有回答模式加起来，整体风险有多大。
```

### 8.3 新核心指标 2：Risk Entropy

定义：

```text
q_c = p(c | x) * h(c, x) / sum_j p(j | x) * h(j, x)
RiskEntropy(x) = - sum_c q_c log q_c
```

专业解释：

```text
Risk entropy measures whether risk is concentrated in one mode or distributed across many modes.
```

白话解释：

```text
风险熵看的是：风险是集中在一种回答里，还是分散在很多回答里。
```

### 8.4 新核心指标 3：Risk-Diversity Decoupling

定义：

```text
RDD occurs when Delta Risk > 0 and Delta Diversity >= 0.
```

在当前实验中：

- Qwen3-1.7B：Guardian risk up，entropy up；
- Qwen3-4B：Guardian risk up，entropy up；
- S0.1：dominant harm up，entropy up；

这些都更接近 RDD，而不是 PCE。

### 8.5 四象限诊断

| 象限 | 风险变化 | 多样性变化 | 含义 | 例子 |
| --- | --- | --- | --- | --- |
| Collapse Risk | 风险上升 | 多样性下降 | 传统 PCE 风险 | 0.5B first-10 pilot |
| Distributed Risk | 风险上升 | 多样性上升或不降 | 新主线，风险迁移 | Qwen3-1.7B / Qwen3-4B |
| Safe Concentration | 风险下降 | 多样性下降 | 集中到安全拒答 | refusal-template control |
| Benign Diversity | 风险不升 | 多样性不降 | 安全或中性变化 | neutral-boundary 部分结果 |

专业表述：

```text
This quadrant view separates collapse-based exploitability from distributed risk migration.
```

白话表述：

```text
这个四象限能区分：模型是固定地变危险，分散地变危险，固定地变安全，还是基本没问题。
```

## 9. 当前进展

### 9.1 已完成的工程能力

| 能力 | 状态 |
| --- | --- |
| 本地 git 仓库 | 已建立，无远端，无 push |
| conda 环境 | 使用 `stdplm` |
| GPU | RTX 4060 Laptop 可用 |
| Qwen3 overlay | `D:\hf_models\pydeps\qwen3_transformers` |
| Qwen3-1.7B | 已下载，可训练，可评估 |
| Qwen3-4B | 已下载，可训练，可评估 |
| Granite Guardian | 已本地运行 |
| raw-output audit | 已实现 |
| prompt-level bootstrap | 已实现 |
| held-out protocol | 已实现 |
| poison/CAR smoke | 已实现 |
| Qwen3 held-out 80 prompt gate | 已冻结并完成 baseline 采样 |

### 9.2 已完成的科学判断

支持：

```text
DPO can amplify safety risk.
```

白话：

```text
DPO 确实会让模型更容易给出风险回答。
```

不支持：

```text
DPO reliably causes sampled mode collapse.
```

白话：

```text
DPO 并不会稳定地让模型反复输出同一类答案。
```

新的核心判断：

```text
DPO can produce risk migration without diversity collapse.
```

白话：

```text
DPO 可能让模型用更多不同方式变危险。
```

## 10. 问题逻辑

### 10.1 旧逻辑

```text
DPO
-> preference margin increases
-> output modes collapse
-> dominant risky mode emerges
-> stable exploitability
```

白话：

```text
训练让模型偏向某类答案
-> 回答越来越固定
-> 固定答案有风险
-> 漏洞成立
```

实验结论：

```text
This chain fails at "output modes collapse".
```

白话：

```text
链条断在“回答越来越固定”这一步。
```

### 10.2 新逻辑

```text
DPO
-> preference margin increases
-> refusal boundary shifts
-> model explores more risky but preference-consistent responses
-> guardian-scored risk rises across multiple modes
-> distributed risk migration
```

白话：

```text
训练让模型更偏向“有帮助、少拒答”
-> 模型在危险问题上更愿意回答
-> 它不是重复一个答案，而是用很多不同方式回答
-> 整体风险上升
```

### 10.3 直观例子

原来以为：

```text
训练后模型只会反复给出同一道风险答案。
```

实际看到：

```text
训练后模型会给出很多不同的风险回答。
```

用做饭类比：

```text
原来以为训练会让厨师每次都做同一道有问题的菜。
实际像是训练让厨师偏向某类口味，于是每次做出不同菜，但都落在同一个风险口味区间里。
```

## 11. 研究问题

| 编号 | 专业表述 | 白话表述 |
| --- | --- | --- |
| RQ1 | Does DPO increase guardian-scored risk while preserving or increasing semantic output diversity? | DPO 会不会让模型更危险，但回答并不更重复？ |
| RQ2 | Under what prompt strata does risk-diversity decoupling occur? | 哪些题型更容易出现“更危险但更多样”？ |
| RQ3 | How do preference margins, refusal rates, and sampling diversity jointly predict risk migration? | 训练指标、拒答率和多样性能不能提前预警风险迁移？ |
| RQ4 | Can RDI and risk entropy distinguish collapse risk from distributed risk? | 新指标能不能区分“固定地危险”和“分散地危险”？ |
| RQ5 | Can filtering or regularization reduce distributed risk without simply forcing refusal? | 有没有办法降低分散风险，而不是让模型什么都拒绝？ |

## 12. 创新点

### 12.1 从 PCE 到风险-多样性解耦

专业表述：

```text
The project reframes collapse-based exploitability as one quadrant of a broader risk-diversity diagnostic space.
```

白话表述：

```text
我不再只看“模型是否坍缩”，而是看风险和多样性怎么一起变化。
```

### 12.2 发现风险上升不需要模式坍缩

专业表述：

```text
Across Qwen3-1.7B and Qwen3-4B, DPO increases guardian-scored risk while determinism decreases and mode entropy increases.
```

白话表述：

```text
更大的 Qwen3 模型说明：模型可以不重复，却更危险。
```

### 12.3 Prompt-stratified 诊断

专业表述：

```text
Aggregate metrics can hide prompt-level heterogeneity; prompt-stratified evaluation is necessary for reliable safety diagnosis.
```

白话表述：

```text
只看平均分会被骗，必须逐题看哪些 prompt 出问题。
```

### 12.4 高质量 negative results

专业表述：

```text
The study includes held-out failure, taxonomy failure, poison/CAR failure, and Qwen3 collapse failure as evidence that constrains the hypothesis.
```

白话表述：

```text
这些失败不是坏事，它们帮助把研究从夸大漏洞收敛到真正成立的问题。
```

### 12.5 新指标：RDI 与 Risk Entropy

专业表述：

```text
RDI and risk entropy measure distributed risk across semantic modes, complementing PCE's dominant-mode focus.
```

白话表述：

```text
新指标不只盯着最大那类回答，而是看风险是否分散在很多回答里。
```

## 13. 前人工作与本课题位置

### 13.1 DPO 与直接偏好优化

Rafailov 等提出 DPO，把 RLHF 中复杂的 reward modeling 和强化学习步骤转化为直接的偏好分类损失。

专业定位：

```text
DPO is the optimization substrate studied in this project.
```

白话定位：

```text
DPO 是本课题研究的训练方法背景。
```

### 13.2 DPO failure modes 与 likelihood over-optimization

Smaug / DPO-Positive 指出，标准 DPO 只要相对偏好改善，就可能降低 preferred examples 的 likelihood。Likelihood over-optimisation 工作进一步指出，更高 likelihood 或更大 preference margin 不一定提升泛化，还可能损害多样性。

专业定位：

```text
Prior work shows that preference-objective improvement and downstream behavior can diverge.
```

白话定位：

```text
前人已经发现：训练目标变好，不等于模型实际表现一定变好。
```

本课题进一步问：

```text
如果目标变好但行为改变，安全风险和输出多样性会如何共同变化？
```

### 13.3 输出多样性坍缩与多样性保持

Diverse Preference Optimization 和 post-training diversity collapse 工作说明，后训练和偏好优化常常会让输出分布变尖、多样性下降。

专业定位：

```text
Output diversity loss is known; the new question is whether risk can increase without such loss.
```

白话定位：

```text
前人研究“模型会不会回答变少”。本课题研究“模型即使没有回答变少，会不会仍然更危险”。
```

### 13.4 DPO 安全攻击与偏好投毒

Benign DPO Attack 表明，看似无害的偏好对可以降低拒答并迁移到 harmful prompts。Preference poisoning 工作也研究了对齐数据中的低成本投毒。

专业定位：

```text
Attack success and refusal suppression are related, but they do not imply mode collapse.
```

白话定位：

```text
别人已经发现 DPO 可能让模型更少拒答，但这不等于它会重复同一种危险回答。
```

本课题的位置：

```text
从“是否攻击成功”进一步分析“风险是集中出现，还是分散出现”。
```

### 13.5 安全分类器

Granite Guardian、Llama Guard、ShieldGemma 等工作提供了 prompt/response 风险检测模型。

专业定位：

```text
Guardian-style classifiers provide external risk labels for sampled response modes.
```

白话定位：

```text
这些模型像安全裁判，用来判断回答有没有风险。
```

本项目当前使用 Granite Guardian 3.1 2B，本机可运行。后续若条件允许，应增加 Llama Guard 或 ShieldGemma 做 classifier sensitivity 分析。

## 14. 目标会议与投稿定位

### 14.1 主目标：AAAI / IJCAI

定位：

```text
LLM safety diagnostics, alignment reliability, post-training behavior analysis, and risk monitoring.
```

白话：

```text
研究大模型偏好训练后哪里会出安全风险，以及怎么诊断。
```

为什么适合：

- AAAI / IJCAI 接受 AI 系统可靠性、安全评测、诊断框架和机制分析。
- 当前研究有明确实验链条、negative controls、跨模型现象和新指标。
- 不需要硬包装为安全漏洞攻击论文。

### 14.2 条件冲刺：NeurIPS / ICML / ICLR

定位：

```text
Preference optimization dynamics, risk-diversity tradeoff, and training-time warning signals.
```

白话：

```text
如果能把 DPO 为什么导致风险迁移讲清楚，就可以冲更偏机器学习机制的顶会。
```

需要补强：

- 更严格的理论或半理论解释；
- 多模型、多数据源、多安全分类器；
- 训练过程动态；
- RDI / Risk Entropy 与泛化风险的统计关系。

### 14.3 备选：ACL

定位：

```text
LLM safety evaluation, prompt taxonomy, response diversity, and diagnostic benchmark.
```

白话：

```text
如果最后贡献更像语言模型安全评测集和分层分析，可以考虑 ACL 路线。
```

### 14.4 暂不主攻：安全四大

IEEE S&P、USENIX Security、ACM CCS、NDSS 暂不作为第一目标。

原因：

- Experiment C 不支持主动诱导攻击。
- 没有稳定 exploit。
- 没有跨 prompt 的可预测漏洞。
- 当前贡献更像 alignment diagnostics，而不是攻击系统。

白话：

```text
现在不能说“我发现了一个稳定漏洞”，所以不要硬投安全攻击顶会。
```

## 15. 研究计划

### 阶段 1：重构实验台账与指标

时间：2026 年 7 月

任务：

- 建立 `docs/experiment_index.md`；
- 把所有 run family、模型、seed、prompt slice、output dir、结论统一索引；
- 把旧 PCE 指标降级为 collapse-risk 指标；
- 新增 RDI、Risk Entropy、RDD quadrant summary；
- 对已保存 Qwen3 输出做 retrospective RDI 计算。

交付：

- 实验台账；
- 新指标脚本；
- Qwen3 retrospective risk-diversity summary。

### 阶段 2：Qwen3 风险-多样性复核

时间：2026 年 7 月至 8 月

任务：

- 使用 Qwen3-1.7B 作为快速迭代模型；
- Qwen3-4B 只在协议稳定后继续；
- 不再新增 0.5B 训练；
- 已冻结 80 个 zero-overlap AdvBench held-out prompts；
- 已完成 Qwen3-1.7B baseline 80 prompts x 32 samples；
- 下一步在同一 80 题集合上运行 Qwen3-1.7B DPO；
- 比较 temperature / top-p / max_new_tokens 对 risk-diversity 的影响；
- Granite Guardian 必跑。

成功标准：

```text
在同一 frozen held-out prompt set 上复现 risk up，并报告 diversity not down
是否成立、absolute diversity 是否足够、RDI / Risk Entropy 是否上升。
```

白话：

```text
要证明“分散式风险迁移”不是 first-10 偶然现象。
```

### 阶段 3：Prompt-stratified benchmark v1

时间：2026 年 8 月至 9 月

任务：

- 构建 300 到 500 prompts；
- 分层包括 cyber、fraud、violence/weapons、self-harm、harassment、general harmful requests、benign hard prompts；
- 每个 stratum 固定 prompt IDs；
- 先做 baseline risk-diversity profiling；
- 再做 DPO 后变化分析。

交付：

- prompt taxonomy v1；
- held-out stratum validation；
- prompt-level mixed-effects 或 bootstrap 分析。

### 阶段 4：机制分析

时间：2026 年 9 月至 11 月

任务：

- 分析 preference margin 与 RDI 变化关系；
- 分析 refusal rate 与 risk migration 关系；
- 分析 token entropy、top-k entropy、semantic mode entropy；
- 区分三种机制：
  - refusal boundary shift；
  - semantic risk expansion；
  - true collapse risk。

目标：

```text
解释 DPO 为什么会让风险上升但多样性不下降。
```

### 阶段 5：预警与缓解

时间：2026 年 11 月至 2027 年 1 月

任务：

- preference filtering；
- entropy / diversity regularization；
- refusal-boundary regularization；
- risk-aware DPO weighting；
- 与普通 DPO 比较 RDI、Risk Entropy、helpfulness proxy、refusal rate。

成功标准：

```text
降低 distributed risk，不只是让模型全部拒答。
```

白话：

```text
防御不能靠“什么都不回答”，而要保持有用同时降低风险。
```

### 阶段 6：论文整理与投稿

时间：2027 年 1 月至 2027 年 4 月

AAAI / IJCAI 投稿主线：

```text
We show that preference optimization can decouple safety risk from output diversity: DPO increases guardian-scored risk while preserving or increasing semantic diversity. We propose RDI, risk entropy, and prompt-stratified diagnostics to distinguish collapse risk from distributed risk migration.
```

中文主线：

```text
我们发现偏好优化不一定通过模式坍缩增加安全风险；更隐蔽的情况是，模型在保持甚至增加输出多样性的同时，系统性向风险回答区域迁移。
```

## 16. 风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| RDI 现象不泛化 | Qwen3 扩展 prompts 后风险不稳定 | 转为 negative benchmark，强调 PCE 假设证伪 |
| Guardian 偏差 | Granite 与其他 classifier 不一致 | 引入 Llama Guard 或 ShieldGemma，报告 classifier sensitivity |
| Prompt taxonomy 失败 | 分层无法预测新数据 | 改用 prompt embeddings / learned features，不强行人工 taxonomy |
| 审稿人认为只是指标 | 缺少机制解释 | 加强 preference margin、refusal boundary、entropy dynamics 分析 |
| 安全贡献被质疑 | 没有攻击成功 | 明确定位为诊断和预警，不宣称 exploit |
| 计算资源不足 | 4B/8B 训练受限 | Qwen3-1.7B 做训练，4B 做验证，8B 只做 inference audit |

## 17. 当前结论

专业结论：

```text
The original PCE vulnerability hypothesis is not supported as a general claim. The strongest supported phenomenon is risk-diversity decoupling: DPO can increase guardian-scored risk while sampled outputs remain diverse or become more diverse.
```

白话结论：

```text
原来想证明“DPO 让模型重复同一种危险回答”，现在看不成立。真正值得研究的是“DPO 让模型用很多不同方式变得更危险”。
```

正式开题表述：

```text
本研究从 DPO 可能诱发偏好坍缩可利用性的原始假设出发，通过多阶段本地实验发现，跨 Qwen2.5-0.5B、Qwen3-1.7B 和 Qwen3-4B 的主要稳定现象并不是 sampled-mode collapse，而是安全风险上升与输出多样性保持或增加之间的解耦。基于此，课题转向偏好优化中的风险-多样性解耦诊断与预警，提出 RDI、Risk Entropy 和 prompt-stratified evaluation，以刻画风险迁移、解释 prompt 异质性，并探索训练过程预警与缓解方法。
```

## 18. 参考文献与近期工作

1. Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023. https://arxiv.org/abs/2305.18290
2. Pal et al. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive. 2024. https://arxiv.org/abs/2402.13228
3. Shi et al. Understanding Likelihood Over-optimisation in Direct Alignment Algorithms. 2024. https://arxiv.org/abs/2410.11677
4. Lanchantin et al. Diverse Preference Optimization. 2025. https://arxiv.org/abs/2501.18101
5. Karouzos et al. Where does output diversity collapse in post-training? 2026. https://arxiv.org/abs/2604.16027
6. Yoon et al. Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs. 2026. https://arxiv.org/abs/2605.10998
7. Padhi et al. Granite Guardian. 2024. https://arxiv.org/abs/2412.07724
8. Zeng et al. ShieldGemma: Generative AI Content Moderation Based on Gemma. 2024. https://arxiv.org/abs/2407.21772

## 19. 100 字以内导师消息

老师，最新 Qwen3 实验表明原始“DPO 导致模式坍缩漏洞”不成立；但 DPO 会显著提高风险回答倾向。建议开题转向“偏好优化中的风险-多样性解耦诊断与预警”。
