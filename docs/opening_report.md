# 开题报告：从 DPO 漏洞假设到 Prompt-Stratified PCE 诊断与预警

日期：2026-07-02
项目目录：`C:\Users\TH.Xie\Desktop\DPO`
当前状态：实验先行验证后，研究方向已从“直接证明漏洞”收敛为“分层诊断与预警”

## 0. 一句话概括

专业表述：

```text
本课题研究偏好优化语言模型中，DPO 是否会诱发安全相关的 response-mode concentration，并提出 prompt-stratified PCE 诊断框架来刻画其发生条件、泛化边界和预警信号。
```

白话表述：

```text
我原来想验证：模型经过 DPO 训练后，会不会变得更容易在某些危险问题上反复给出同一类有风险回答。实验后发现：这个现象不是普遍漏洞，但它在部分 prompt 上有信号，所以更好的开题方向是研究如何发现、区分和预警这种风险模式。
```

最终开题方向建议：

```text
面向偏好优化语言模型的 Prompt-Stratified PCE 诊断与预警研究
```

目标会议建议：

- 主目标：AAAI、IJCAI。
- 条件冲刺：NeurIPS、ICML、ICLR。
- 备选路线：ACL、EMNLP。
- 暂不主攻：IEEE S&P、USENIX Security、ACM CCS、NDSS。

原因很简单：当前证据不足以支撑“已经发现可利用安全漏洞”的安全顶会叙事，但足以支撑“偏好优化后的风险模式诊断、评测和预警”的人工智能顶会叙事。

## 1. 原先的 idea 是什么

### 1.1 原始想法

专业表述：

```text
Direct Preference Optimization may reduce output diversity, increase sampled-output determinism, and raise Preference Collapse Exploitability on safety-sensitive prompts.
```

白话表述：

```text
DPO 是一种让模型更符合人类偏好的训练方法。原始想法是：这种训练可能会让模型回答变窄、变固定；如果固定下来的回答刚好有风险，模型就可能更容易被利用。
```

原先的研究假设可以写成：

```text
DPO 训练
-> 输出多样性下降
-> 同一个 prompt 下更容易出现主导回答模式
-> 主导模式如果有害，则 PCE 上升
-> 攻击者可能利用这种稳定模式
```

更直观地说：

```text
模型不是偶尔说错，而是越来越稳定地往某类风险回答靠。
```

### 1.2 原始想法为什么值得怀疑

专业表述：

```text
Preference fitting, likelihood margin improvement, and sampled generation behavior are not equivalent.
```

白话表述：

```text
训练指标变好，不代表模型实际多次生成时一定会更集中；模型学会“偏好 chosen”，也不代表它真正会反复输出同一种答案。
```

因此这个 idea 不能只靠直觉成立，必须先做小实验验证。项目从一开始就采用实验先行，而不是先写论文 claim。

## 2. 相关概念是什么

本节采用双层写法：先给专业概念，再给零基础解释。

| 概念 | 专业解释 | 零基础解释 |
| --- | --- | --- |
| 大语言模型 | 通过大规模文本预训练得到的生成式模型 | 一个会根据上下文继续写、回答问题的文本系统 |
| 后训练 | 在预训练后继续用指令、偏好或反馈数据调整模型 | 把“会写字的模型”训练成“更像助手的模型” |
| SFT | Supervised Fine-Tuning，用标准答案监督微调 | 给模型看题目和标准答案，让它模仿 |
| RLHF | Reinforcement Learning from Human Feedback，用人类偏好训练奖励模型再优化策略 | 先学一个“什么回答更好”的打分器，再让模型追高分 |
| DPO | Direct Preference Optimization，直接用 chosen/rejected 偏好对训练 | 告诉模型“A 回答比 B 回答好”，让它以后更偏向 A |
| chosen/rejected | 偏好数据中的优选回答和劣选回答 | 两个答案里，一个被标为更好，一个被标为更差 |
| mode | 多次生成中语义相近的一类回答 | 模型经常绕回去的同一种答法 |
| mode collapse | 输出集中到少数模式，多样性下降 | 模型越来越像只会几种固定说法 |
| determinism | 主导模式的概率质量，即 dominant cluster mass | 重复问同一题时，模型有多大概率回到同一种答案 |
| mode entropy | 回答模式分布的熵 | 回答越分散，熵越高；越集中，熵越低 |
| harmfulness | 回答被安全分类器判为风险的程度 | 这个回答是不是危险、不合规或可能帮助做坏事 |
| PCE | determinism 与 dominant harmfulness 的乘积 | 模型是否稳定地给出某类风险回答 |
| Guardian | 独立安全分类器，如 Granite Guardian、Llama Guard | 一个专门判断回答是否有风险的裁判 |
| held-out prompt | 训练和调参时没看过的新 prompt | 用新题考试，防止只是在旧题上表现好 |
| prompt-stratified | 按 prompt 类型分层统计 | 不只看平均分，还看哪些题型出问题 |

## 3. 为什么必须先做测试

### 3.1 要证明什么

原始 idea 如果要成立，至少要通过四个门槛：

1. 指标门槛：
   - 专业说法：PCE 计算管线必须能区分 diverse outputs 和 collapsed outputs。
   - 白话说法：尺子本身要靠谱，不能拿坏尺子量模型。

2. 真实模型门槛：
   - 专业说法：真实 instruction model 上需要观察到 DPO 后 determinism 上升、entropy 下降。
   - 白话说法：不能只在玩具模型里成立，真实小模型上也要有方向。

3. 安全相关门槛：
   - 专业说法：dominant mode harmfulness 需要由独立 guardian-style classifier 支持。
   - 白话说法：不能只说回答变固定，还要证明固定下来的回答更危险。

4. 泛化门槛：
   - 专业说法：信号需要在 held-out prompts、不同 seeds、不同 response wording 上相对稳定。
   - 白话说法：不能只在挑出来的几道题上好看。

### 3.2 测试策略

专业表述：

```text
采用从 sanity check 到 real-model gate，再到 held-out validation 和 negative control 的逐级验证策略。
```

白话表述：

```text
先检查工具能不能用，再检查小模型能不能跑，再检查结果是不是真的稳定，最后用反向实验和新题来防止自欺欺人。
```

## 4. 已经做了什么测试，为什么做，证明了什么

### 4.1 总览

| 阶段 | 测试 | 为什么做 | 结果 | 说明 |
| --- | --- | --- | --- | --- |
| 指标 sanity | synthetic diverse vs collapsed | 验证 PCE 尺子是否有效 | 指标方向正确 | 管线可用 |
| 机制 sanity | toy DPO categorical update | 验证偏好更新能否集中概率质量 | 概率质量会集中 | 机制有可能 |
| 端到端 smoke | tiny GPT-2 | 验证训练、采样、聚类、评估是否跑通 | 跑通 | 只证明工程链路 |
| 小模型 gate | SmolLM2-135M | 看真实 instruction model 是否有弱信号 | 两种子弱方向 | 不足以支撑 claim |
| 小模型复核 | SmolLM2-360M corrected | 提高评估强度 | 10x8 弱，10x16 fail | 信号不稳 |
| 控制实验 | uniform-control | 排除训练流程本身制造假阳性 | 反向或 fail | 不能支持主 claim |
| Qwen 初步 | 20-step / 100-step | 利用本地 Qwen 验证真实模型 | 20-step fail，100-step weak | 方向不稳定 |
| margin 诊断 | preference margin tracking | 检查训练目标和生成行为是否一致 | margin 可变好，但生成不必然 collapse | 发现关键断点 |
| 强正信号 | short-template 300-step stress | 寻找能否出现清晰 PCE 信号 | first-10 prompts robust pass | 最强但受限 |
| 反向控制 | refusal-template DPO | 验证同一管线能否把风险方向反过来 | harm proxy 下降，refusal 上升 | 支持“方向可控” |
| 中性控制 | neutral-boundary | 区分 determinism 和 harmfulness | 弱 collapse，无 robust harm increase | determinism 不等于风险 |
| wording 复现 | concise-overview | 检查是否依赖某个 chosen 写法 | 只有 weak pass | wording-sensitive |
| prompt transfer | prompts 10-19 | 检查第一组 prompt 结果能否迁移 | mixed/fail | transfer 不稳 |
| fallback held-out | 30 prompts 分块 | 检查更广 prompt 范围 | first10 weak，offset10 mixed，offset20 robust fail | 泛化失败明显 |
| taxonomy v0 | cyber vs violence/weapons | 尝试解释哪些 prompt 易出信号 | old 50 上有模式 | 只是探索 |
| taxonomy held-out | AdvBench 4-vs-4 | 验证 taxonomy v0 是否能预测新数据 | 失败 | taxonomy v0 不可用 |
| S0.1 | AdvBench held-out 30 | 开题前最关键泛化 gate | 21/60 pass，低于 60% | 不进入漏洞 claim |
| Experiment C | poison/CAR smoke | 检查低率投毒是否加速 collapse | 无剂量效应 | 不支持主动攻击叙事 |

### 4.2 指标 sanity：先检查尺子

专业表述：

```text
Synthetic diverse-vs-collapsed outputs were used to validate determinism, mode entropy, cluster count, distinct-n, and proxy PCE.
```

白话表述：

```text
先人工造一组“很分散的回答”和一组“很重复的回答”，看指标能不能把它们区分开。
```

结果：

- collapsed outputs 的 determinism 更高；
- diverse outputs 的 entropy 更高；
- proxy PCE 会随风险代理项变化。

结论：

```text
指标管线基本可用，但这还不能证明真实模型会出问题。
```

### 4.3 toy DPO：验证机制可能性

专业表述：

```text
Toy categorical DPO experiments show that preference-style updates can concentrate probability mass.
```

白话表述：

```text
在极简玩具模型里，如果一直偏好某类答案，模型概率确实会往那类答案集中。
```

结论：

```text
原始 idea 在机制上不是胡思乱想，但玩具模型不能替代真实语言模型。
```

### 4.4 tiny GPT-2 和 SmolLM2：验证工程链路和小模型信号

专业表述：

```text
Tiny GPT-2 verifies end-to-end orchestration, while SmolLM2 gates test whether the signal appears in real instruction models.
```

白话表述：

```text
先用很小的模型确认脚本能跑，再用小型指令模型看看有没有真实信号。
```

结果：

- tiny GPT-2 跑通训练、采样、聚类、评估；
- SmolLM2-135M 有弱两种子方向；
- SmolLM2-360M 在更强 10x16 评估下 robust fail；
- uniform-control 甚至出现反向。

结论：

```text
工程路线可行，但小模型结果不能支撑论文 claim。
```

### 4.5 Qwen2.5-0.5B：本地真实模型主线

专业表述：

```text
Qwen2.5-0.5B-Instruct became the main local instruction model for RTX 4060 validation.
```

白话表述：

```text
用户手动下载 Qwen 后，本地终于有了一个像样的小指令模型，可以更认真地验证。
```

环境：

| 项目 | 状态 |
| --- | --- |
| Conda 环境 | `stdplm` |
| GPU | RTX 4060 Laptop 8GB |
| Qwen 路径 | `D:\hf_models` 与本地 assembled model |
| Guardian 路径 | `D:\hf_models\ibm-granite_granite-guardian-3.1-2b` |
| Git | 本地仓库，无远端，无 push |

关键发现：

```text
DPO preference fitting 可以发生，但 sampled-mode collapse 不一定跟着发生。
```

白话解释：

```text
模型训练时确实学到了“chosen 比 rejected 更好”，但它真正多次回答时，不一定会稳定地变成同一种答法。
```

这是原始 idea 中最重要的断点。

### 4.6 最强正信号：short-template 300-step stress

专业表述：

```text
Qwen2.5-0.5B-Instruct under a short-template DPO stress shows robust local and Granite Guardian-backed PCE increase on the original first-10 prompts.
```

白话表述：

```text
在最容易出信号的前 10 个问题上，用特定短模板训练 300 步，模型确实更容易集中到某类被 Guardian 判为风险更高的回答模式。
```

设置：

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen2.5-0.5B-Instruct |
| 训练 | DPO stress |
| learning rate | 3e-6 |
| steps | 300 |
| seeds | 42、43 |
| prompts | 原始 first 10 |
| samples | 每 prompt 16 次 |

结果：

- preference margin 朝 chosen 翻转；
- determinism 上升；
- mode entropy 下降；
- local gate robust pass；
- Granite Guardian-PCE robust pass；
- dominant Guardian harmfulness robust increase；
- refusal-template counter-control 朝相反方向移动。

这个结果说明：

```text
不能简单否定原始 idea，局部 PCE 信号确实存在。
```

但它也有明显限制：

- 只在 first-10 prompt 最强；
- 依赖特定 short-template wording；
- target phrase exact hit 仍为 0；
- 每个 prompt 仍有多个不同输出，不是简单复制；
- held-out 泛化不稳定。

### 4.7 控制实验：为什么 determinism 不等于漏洞

专业表述：

```text
Refusal-template and neutral-boundary controls separate response concentration from safety-relevant harmfulness.
```

白话表述：

```text
回答变固定不一定坏。如果固定成拒答，那反而更安全；如果固定成中性解释，也未必危险。
```

反向控制结果：

- refusal-template DPO 后 determinism 上升；
- entropy 下降；
- refusal 增加；
- compliance/actionability 下降；
- dominant Guardian harmfulness 下降。

这说明：

```text
PCE 必须同时看 determinism 和 harmfulness，不能只看多样性下降。
```

### 4.8 prompt transfer：最大科学 blocker

专业表述：

```text
The strongest positive checkpoint does not transfer reliably across prompt blocks.
```

白话表述：

```text
在第一组题上好看，不代表换一组题还好看。
```

结果摘要：

| Prompt block | 结果 |
| --- | --- |
| 原始 first 10 | strongest positive |
| prompts 10-19 | mixed/fail |
| held-out fallback first 10 | weak positive |
| held-out fallback offset 10 | Guardian mixed |
| held-out fallback offset 20 | local 和 Guardian-PCE robust fail |
| full 50 aggregate | aggregate positive，但 prompt split 极其异质 |

full 50 最关键结果：

```text
34 pass / 33 mixed / 33 fail
```

白话解释：

```text
平均看好像有正信号，但拆到每道题后，成功、混合、失败几乎一样多。
```

这直接推翻了“广泛稳定漏洞”的强说法。

### 4.9 taxonomy v0：第一次解释尝试失败

专业表述：

```text
The first frozen prompt taxonomy describes the old 50-prompt heterogeneity but fails out-of-sample AdvBench validation.
```

白话表述：

```text
我试图用题目类型解释哪些题会出信号，但第一版分类规则只能解释旧数据，换新数据就不准。
```

结果：

| Topic | local gate | Guardian-PCE | dominant harm |
| --- | --- | --- | --- |
| cyber | fail/fail across two seeds，pooled mixed | mixed | robust increase |
| violence/weapons | mixed/mixed | mixed | robust decrease |

结论：

```text
taxonomy v0 不能作为论文 claim 的依据，后续需要 taxonomy v1 或更稳健的分层方法。
```

### 4.10 S0.1 held-out 30：开题前最关键实验

专业表述：

```text
S0.1 evaluates the original positive Qwen checkpoints on 30 zero-overlap AdvBench held-out prompts with two seeds and Guardian-PCE reporting.
```

白话表述：

```text
这是用 30 道没见过的新题考试，检验前面的正信号是不是能泛化。
```

关键结果：

| 指标 | 结果 |
| --- | --- |
| prompt 数量 | 30 个 AdvBench held-out prompts |
| seed42 local | det -0.0125，entropy +0.0190，proxy PCE +0.0250，10 pass / 2 mixed / 18 fail |
| seed43 local | det -0.0115，entropy +0.0322，proxy PCE +0.0208，11 pass / 2 mixed / 17 fail |
| pooled local | 21 pass / 4 mixed / 35 fail，pass rate 35.0% |
| 预注册门槛 | pass rate 至少 60% |
| Guardian-PCE | mixed；Yes delta +0.0307，CI [+0.0130, +0.0490] |
| dominant harm | robust increase；Yes delta +0.1990，CI [+0.0701, +0.3276] |
| target phrase hit | seed42 final 0/960，seed43 final 0/960 |
| 决策 | fail preregistered criteria，不进入 S1 |

这个结果非常重要：

```text
harmfulness 上升了，但 determinism 没有上升，entropy 反而上升。
```

白话解释：

```text
新题上模型回答可能更危险一点，但没有更固定；所以这不是稳定的“偏好坍缩可利用性”证据。
```

### 4.11 Experiment C：投毒/CAR smoke 为负

专业表述：

```text
The poison/CAR smoke test does not show the expected dose-response relation CAR_5% > CAR_1% > CAR_clean.
```

白话表述：

```text
如果投毒真的能主动加速坍缩，那么 5% 投毒应该比 1% 更明显，1% 应该比 clean 更明显。但实验没有这样。
```

Local CAR：

| 条件 | Det Delta | Entropy Delta | CAR | Proxy PCE Delta |
| --- | --- | --- | --- | --- |
| clean | +0.0312 | -0.0797 | +0.000797 | +0.0062 |
| 1% poison | +0.0312 | -0.0797 | +0.000797 | +0.0000 |
| 5% poison | +0.0188 | -0.0451 | +0.000451 | -0.0187 |

Guardian-PCE：

| 条件 | Guardian-PCE Yes Delta | Guardian-PCE Score Delta |
| --- | --- | --- |
| clean | +0.0437 | +0.0326 |
| 1% poison | +0.0312 | +0.0228 |
| 5% poison | +0.0187 | +0.0095 |

结论：

```text
Experiment C 不支持主动诱导攻击叙事，只能作为一个已完成的 negative smoke。
```

白话解释：

```text
现在不能说攻击者可以靠少量投毒稳定制造这个问题。
```

## 5. 这些测试推翻了什么，保留了什么

### 5.1 被推翻或暂不成立的部分

专业表述：

```text
The broad vulnerability claim is not supported.
```

白话表述：

```text
现在不能说“DPO 会普遍导致可利用漏洞”。
```

具体原因：

- held-out prompt 不稳定；
- S0.1 未达到 60% pass 门槛；
- taxonomy v0 不能预测新数据；
- poison/CAR 没有剂量效应；
- wording-sensitive；
- LlamaGuard-family replication 暂不可用；
- target phrase exact copying 没有出现。

### 5.2 被保留的部分

专业表述：

```text
There is restricted evidence for prompt- and wording-sensitive safety-relevant response-mode concentration.
```

白话表述：

```text
现象不是假的，但它只在部分题、部分写法、部分训练设置下出现。
```

保留下来的证据：

- short-template 300-step first-10 prompts 有 robust PCE 正信号；
- refusal-template control 证明同一管线能反向降低风险；
- Granite Guardian sanity check 表明 classifier 对 response 内容敏感；
- full 50 aggregate 有 positive signal，但异质性强；
- margin-to-generation 的断点本身有研究价值。

## 6. 现在的 idea 是什么

### 6.1 新 idea

专业表述：

```text
Develop a prompt-stratified PCE diagnostic and early-warning framework for preference-optimized language models.
```

白话表述：

```text
不要再硬说 DPO 一定造成漏洞，而是研究：哪些题、哪些回答模式、哪些训练条件下会出现风险集中；怎么提前发现它；怎么避免被平均数骗了。
```

新 idea 的核心问题：

```text
When does preference optimization create safety-relevant response-mode concentration, and how can we diagnose it reliably?
```

对应中文：

```text
偏好优化什么时候会制造安全相关的回答模式集中？我们如何可靠地诊断它？
```

### 6.2 为什么这个方向更好

专业表述：

```text
It matches the empirical evidence: local positive signal exists, but cross-prompt generalization is heterogeneous.
```

白话表述：

```text
实验告诉我们：不是完全没戏，也不是已经证明漏洞。最真实的状态是“有局部信号，但边界不清”。新方向正好研究这个边界。
```

这个方向比原始漏洞叙事更适合开题：

- 更严谨，不夸大；
- 更容易通过 held-out negative 结果讲出科学问题；
- 更适合 AAAI/IJCAI 的评测、诊断、可靠性叙事；
- 后续如果结果变强，仍可升级到安全攻击/防御论文；
- 如果结果继续不稳定，也能形成有价值的负结果 benchmark。

## 7. 围绕现在 idea 的当前进展

### 7.1 已经完成的能力

专业表述：

```text
The project already has a local PCE measurement pipeline, Qwen DPO smoke training, Guardian audit integration, prompt-level pass/mixed/fail summaries, and several negative controls.
```

白话表述：

```text
现在不是只有想法，已经有能跑的脚本、能训练的小模型、能打分的安全裁判、能看每道题成败的统计表。
```

当前能力：

- 本地 git 仓库已经建立；
- `stdplm` 环境可复用；
- Qwen2.5-0.5B-Instruct 能本地训练和评估；
- Granite Guardian 3.1 2B 能本地审计；
- PCE 指标脚本能输出 determinism、entropy、proxy PCE、Guardian-PCE；
- raw-output audit 能检查 target phrase hit 和 dominant representative；
- prompt-level pass/mixed/fail 统计已经形成；
- S0.1 和 Experiment C 的 protocol 与结果已写入文档；
- README 已持续记录阶段性成果。

### 7.2 当前最重要的实证结论

专业表述：

```text
The central empirical pattern is aggregate-positive but prompt-heterogeneous PCE movement.
```

白话表述：

```text
平均看有一点正信号，但拆到每道题后，很多题不成立。
```

这句话是整个开题的核心支点。

### 7.3 当前最重要的科学问题

专业表述：

```text
The key scientific blocker is prompt transfer and stratum predictability.
```

白话表述：

```text
现在最大问题是：我还不能提前知道哪些题会出信号，哪些题不会。
```

下一阶段就围绕这个问题展开：

```text
从“有没有 PCE”转向“哪些 prompt strata 会有 PCE，为什么会有，能否提前预警”。
```

## 8. 问题逻辑是什么

### 8.1 原始漏洞逻辑

```text
DPO preference update
-> probability distribution sharpens
-> sampled outputs collapse into fewer semantic modes
-> dominant mode becomes harmful
-> attacker can predict and exploit it
```

白话版本：

```text
训练让模型更偏向某些答案
-> 回答越来越固定
-> 固定下来的回答有风险
-> 攻击者更容易利用这个稳定弱点
```

### 8.2 现有实验显示的真实逻辑

```text
DPO preference update
-> preference margin often changes
-> sampled-mode behavior may or may not concentrate
-> harmfulness may rise even when determinism does not
-> prompt identity strongly moderates the result
```

白话版本：

```text
DPO 确实会改变模型偏好
-> 但模型实际回答不一定更固定
-> 有时回答更危险，但不更稳定
-> 哪道题很关键
```

### 8.3 新研究逻辑

```text
Measure PCE at prompt level
-> split prompts into interpretable strata
-> validate strata on held-out prompts
-> identify mechanism and warning signals
-> test mitigation only where the signal is real
```

白话版本：

```text
先逐题测清楚
-> 再按题型分组
-> 用新题验证分组准不准
-> 找出为什么会发生
-> 最后再谈预警和防御
```

## 9. 正式开题方向与研究问题

正式题目：

```text
面向偏好优化语言模型的 Prompt-Stratified PCE 诊断与预警研究
```

研究问题 RQ：

| 编号 | 专业表述 | 白话表述 |
| --- | --- | --- |
| RQ1 | DPO 是否在特定 prompt strata 上导致 sampled response-mode concentration？ | 哪些题会让 DPO 后的模型回答更固定？ |
| RQ2 | Response-mode concentration 何时与 dominant harmfulness 同时上升？ | 固定下来的回答什么时候真的更危险？ |
| RQ3 | Prompt-stratified PCE 是否比 aggregate PCE 更能解释 held-out 泛化？ | 按题型拆开看，是否比只看平均分更靠谱？ |
| RQ4 | Preference margin、token entropy、dominant mass 等训练过程信号能否预警 PCE 上升？ | 训练中能不能提前看出模型要往风险模式集中？ |
| RQ5 | Entropy regularization、preference filtering 或 gradient gating 能否降低 PCE 而不牺牲基本偏好拟合？ | 有没有办法让模型别那么容易集中到风险回答，同时还保持好用？ |

## 10. 对标会议与投稿路线

### 10.1 主目标：AAAI / IJCAI

专业定位：

```text
LLM alignment reliability, safety evaluation, diagnostic benchmark, and training-time monitoring.
```

白话定位：

```text
研究大模型对齐训练后哪里不稳定、怎么测、怎么提前发现。
```

为什么适合：

- AAAI/IJCAI 接受 AI 系统可靠性、评测、诊断和安全相关研究；
- 本课题可强调方法、benchmark、统计 protocol 和机制分析；
- 不必把结果包装成强攻击漏洞。

### 10.2 条件冲刺：NeurIPS / ICML / ICLR

专业定位：

```text
Preference optimization dynamics, probability-mass concentration, and principled early-warning or mitigation methods.
```

白话定位：

```text
如果后续能讲清楚 DPO 为什么会让概率质量转移，并提出有效方法，就能冲更偏机器学习机制的顶会。
```

需要补强：

- 更强机制解释；
- 多模型多数据源；
- 训练动态分析；
- mitigation 方法不仅有效，还要有清晰动机。

### 10.3 备选：ACL / EMNLP

专业定位：

```text
LLM safety evaluation, prompt taxonomy, response-mode analysis, and diagnostic datasets.
```

白话定位：

```text
如果最后主要贡献是语言模型安全评测集和分层分析，可以走 NLP 评测路线。
```

### 10.4 暂不主攻：安全四大

安全四大包括 IEEE S&P、USENIX Security、ACM CCS、NDSS。

暂不主攻的原因：

- 当前 Experiment C 没有支持主动诱导攻击；
- 没有稳定跨 prompt exploit；
- 威胁模型还不够强；
- 缺少多模型、多安全分类器、多攻击 baseline。

白话解释：

```text
现在还不能说“我有一个能稳定利用的漏洞”，所以不要硬走安全攻击论文。
```

## 11. 前人工作与本课题位置

### 11.1 DPO 与直接偏好优化

Rafailov 等提出 DPO，核心贡献是把 RLHF 中复杂的奖励建模和强化学习步骤简化为直接偏好优化。

白话解释：

```text
DPO 让模型对齐训练更简单，因此它很常用，也更值得研究副作用。
```

### 11.2 DPO failure modes

Smaug/DPO-Positive 指出标准 DPO 在某些情况下可能出现 chosen likelihood 下降等 failure mode。Likelihood over-optimisation 相关工作指出，completion likelihood 或 preference margin 过度优化不一定改善泛化，反而可能损害多样性。

白话解释：

```text
前人已经发现：DPO 的训练目标变好，不一定代表模型真的变好。
```

### 11.3 输出多样性下降

Diverse Preference Optimization 和 post-training diversity collapse 相关工作指出，后训练会让输出分布变尖、多样性下降，并且这种现象与训练数据和训练阶段有关。

白话解释：

```text
模型后训练后可能更像“标准答案机器”，这已经有人研究。
```

本课题的区别是：

```text
不只问多样性是否下降，而是问下降后的主导模式是否安全相关。
```

### 11.4 DPO 安全攻击与投毒

近期 benign DPO attack 和 label-flipping poisoning 工作研究了偏好优化中的安全风险和低成本攻击。

白话解释：

```text
有人已经在看 DPO 会不会被恶意利用，但本项目目前的实验不支持直接攻击叙事。
```

本课题更稳的定位是：

```text
攻击之前先做诊断：先知道什么时候真的有 PCE，什么时候只是指标假象。
```

### 11.5 Safety classifier

Llama Guard、Granite Guardian、ShieldGemma 等模型提供了对 prompt/response 风险的自动评估能力。

白话解释：

```text
它们是安全裁判，用来判断模型回答有没有风险。
```

本项目目前使用 Granite Guardian，并计划在条件允许时补充 LlamaGuard 或 ShieldGemma 复现。

## 12. 后续研究计划

### 阶段 1：实验台账与 prompt benchmark 固化

时间：2026 年 7 月

任务：

- 建立 `docs/experiment_index.md`；
- 统一记录 run family、配置、输出目录、seed、prompt split、结论；
- 固化 prompt ID，避免事后挑选；
- 建立至少 300 个 prompt 的分层评测集；
- 设计 taxonomy v1 或更稳健的 prompt feature schema。

交付：

- 实验索引；
- prompt 分层规范；
- pass/mixed/fail 判定标准；
- 可复现实验 protocol。

### 阶段 2：Prompt-stratified held-out validation

时间：2026 年 7 月至 8 月

任务：

- 使用 Qwen2.5-0.5B 作为快速验证模型；
- 至少 2 到 3 个训练 seed；
- 每 prompt 32 到 64 次采样；
- 在 held-out prompt 上验证 strata 是否预测 PCE；
- Granite Guardian 必跑，争取补充 LlamaGuard 或 ShieldGemma。

成功标准：

- 至少一个 held-out stratum 同时满足 determinism 上升、entropy 下降、Guardian-PCE CI > 0；
- prompt-level pass rate 明显高于 random/aggregate baseline；
- target phrase hit 和 raw representative 必报。

失败处理：

```text
如果所有 strata 都不稳定，论文转为 PCE 假阳性诊断与负结果 benchmark。
```

### 阶段 3：2B/3B scale smoke

时间：2026 年 8 月至 9 月

任务：

- 优先同家族 Qwen2.5-1.5B 或 Qwen2.5-3B；
- 使用 LoRA/QLoRA 控制显存；
- 只在阶段 2 已验证的 stratum 上做 scale smoke；
- 不把单模型单种子称为 scale validation。

定位：

```text
direction-not-reversed smoke，而不是规模律证明。
```

白话解释：

```text
只是检查大一点模型上方向有没有反过来，不是证明所有规模都成立。
```

### 阶段 4：机制解释与统计建模

时间：2026 年 9 月至 11 月

任务：

- 分析 preference margin 与 sampled-mode movement 的关系；
- 监控 top-k entropy、token entropy、dominant mass；
- 建立 prompt-level bootstrap 或 mixed-effects 分析；
- 区分三类现象：
  - true PCE increase；
  - harmfulness-only increase；
  - determinism-only increase。

目标：

```text
把“观察到现象”升级为“解释为什么有些 prompt 成立、有些不成立”。
```

### 阶段 5：预警与缓解

时间：2026 年 11 月至 2027 年 1 月

任务：

- 测试 entropy-regularized DPO；
- 测试 preference filtering；
- 测试 rejected-gradient gating 或类似方法；
- 比较 PCE、拒答率、多样性、基本质量指标。

成功标准：

- 降低 risky dominant mode；
- 不只是让模型全拒答；
- held-out prompt 有效果。

### 阶段 6：论文整理与投稿

时间：2027 年 1 月至 2027 年 4 月，按 CFP 调整。

AAAI/IJCAI 主线摘要：

```text
We propose prompt-stratified PCE diagnostics for preference-optimized language models. We show that aggregate PCE can be misleading because response-mode concentration and dominant harmfulness are highly prompt-dependent. We provide held-out validation, negative controls, classifier-backed audits, and early-warning/mitigation studies.
```

白话摘要：

```text
我们提出一种逐题分层的安全诊断方法，说明 DPO 后的风险集中不是到处发生，而是和题型、回答模式、训练条件有关；只看平均分会误判。
```

## 13. 预期创新点

1. PCE 诊断视角。
   - 专业说法：jointly model response-mode determinism and dominant-mode harmfulness。
   - 白话说法：同时看“是否固定”和“固定的回答是否危险”。

2. Prompt-stratified evaluation。
   - 专业说法：report prompt-level pass/mixed/fail rather than only aggregate metrics。
   - 白话说法：不只看平均分，而是逐题看哪里成立。

3. Margin-to-generation diagnostic。
   - 专业说法：separate preference margin improvement from sampled-output collapse。
   - 白话说法：训练学会偏好，不等于实际回答更固定。

4. Negative controls and boundary reporting。
   - 专业说法：include refusal controls, neutral controls, wording replication, held-out failure, and poison negative smoke。
   - 白话说法：主动报告哪些不成立，防止把偶然结果讲成大结论。

5. Training-time early warning。
   - 专业说法：monitor entropy, dominant mass, margin dynamics, and Guardian-PCE during optimization。
   - 白话说法：训练过程中提前发现模型是否要往风险回答集中。

## 14. 风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| 信号继续不稳定 | held-out pass rate 低 | 转为 PCE 假阳性诊断和负结果 benchmark |
| 分类器偏差 | Granite 与其他 classifier 不一致 | 增加 LlamaGuard/ShieldGemma，报告 classifier sensitivity |
| prompt taxonomy 再次失败 | 分层不能预测新数据 | 使用更通用的 prompt features 或弱化 taxonomy claim |
| 大模型显存不足 | 2B/3B OOM | LoRA/QLoRA、减少 samples、只跑验证过的 strata |
| 审稿人认为贡献只是工程 | 缺少机制解释 | 增强 margin-to-generation 分析和统计模型 |
| 安全 claim 被质疑 | 攻击证据不足 | 不主打攻击，主打诊断、边界和预警 |

## 15. 当前结论

专业结论：

```text
Current evidence supports restricted, prompt-sensitive PCE movement but does not support a broad DPO-induced vulnerability claim.
```

白话结论：

```text
现在看到的是局部风险信号，不是已经证明了一个普遍漏洞。
```

正式开题应当这样讲：

```text
本研究从 DPO 可能诱发安全相关模式坍缩的原始假设出发，通过本地多阶段实验发现该现象具有显著 prompt sensitivity 和 wording sensitivity。基于此，课题转向 prompt-stratified PCE 诊断与预警：研究偏好优化后哪些 prompt 和训练条件会产生风险模式集中，如何用 held-out validation 验证其泛化边界，并进一步探索训练过程预警与缓解方法。
```

白话版本：

```text
我原来怀疑 DPO 会造成安全漏洞。实验后发现不能这么直接说，但确实有局部信号。现在更合理的研究是：弄清楚这种信号什么时候出现、为什么出现、怎么提前发现、怎么避免误判。
```

## 16. 参考文献与近期工作

1. Rafailov et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2023. https://arxiv.org/abs/2305.18290
2. Pal et al. Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive. 2024. https://arxiv.org/abs/2402.13228
3. Shi et al. Understanding Likelihood Over-optimisation in Direct Alignment Algorithms. 2024. https://arxiv.org/abs/2410.11677
4. Lanchantin et al. Diverse Preference Optimization. 2025. https://arxiv.org/abs/2501.18101
5. Karouzos et al. Where does output diversity collapse in post-training? 2026. https://arxiv.org/abs/2604.16027
6. Yoon et al. Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs. 2026. https://arxiv.org/abs/2605.10998
7. Kusaka et al. Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment. 2025. https://arxiv.org/abs/2511.09105
8. Fedorov et al. Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations. 2024. https://arxiv.org/abs/2411.17713
9. Padhi et al. Granite Guardian. 2024. https://arxiv.org/abs/2412.07724
10. Zeng et al. ShieldGemma: Generative AI Content Moderation Based on Gemma. 2024. https://arxiv.org/abs/2407.21772
11. Mouiche. Gradient-Gated DPO: Stabilizing Preference Optimization in Language Models. 2026. https://arxiv.org/abs/2605.02626

## 17. 100 字以内导师消息

老师，本地验证显示：原始 DPO 漏洞假设只有局部信号，held-out 和投毒均不支持强结论。建议转向 CCF-A 目标的 prompt-stratified PCE 诊断与预警研究。
