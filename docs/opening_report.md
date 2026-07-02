# 开题报告：面向偏好优化语言模型的 Prompt-Stratified PCE 诊断与预警研究

日期：2026-07-02
项目目录：`C:\Users\TH.Xie\Desktop\DPO`
当前阶段：实验先行验证后，正式开题方向收敛

## 一、拟定题目

中文题目：

```text
面向偏好优化语言模型的 Prompt-Stratified PCE 诊断与预警研究
```

英文题目：

```text
Prompt-Stratified Preference Collapse Exploitability Diagnostics for Preference-Optimized Language Models
```

本题目的核心不是直接宣称“DPO 一定导致安全漏洞”，而是研究一个更严谨、更可发表的问题：

```text
偏好优化后，语言模型什么时候会把回答集中到少数模式？
这种集中什么时候会变成安全相关风险？
这种风险能否被 prompt 分层、指标诊断和训练过程预警提前发现？
```

给非本领域读者的白话解释：

> 大模型经过 DPO 等偏好优化后，常常更“听话”、更符合人类偏好。但如果它面对某类敏感问题时越来越固定地给出同一类回答，而这类回答又被安全模型判为有风险，那么模型就从“偶尔答错”变成了“可预测地答错”。本研究关心的就是这种“可预测的风险回答模式”如何被测量、定位和提前预警。

## 二、对标会议与论文定位

本课题目标按“主目标、扩展目标、暂不主攻目标”区分。

| 路线 | 具体会议 | 定位 | 当前建议 |
| --- | --- | --- | --- |
| CCF-A 人工智能主线 | AAAI、IJCAI | 偏好优化可靠性、LLM 安全评测、诊断方法 | 主目标 |
| 机器学习顶会主线 | NeurIPS、ICML、ICLR | 若后续形成新的优化机制、理论解释或大规模系统性 benchmark，可对标 | 冲刺目标，需按学院/导师认可目录确认分区 |
| NLP/评测主线 | ACL、EMNLP | 若贡献收敛为语言模型安全评测、prompt taxonomy、诊断 benchmark，可考虑 | 备选目标 |
| 安全顶会主线 | IEEE S&P、USENIX Security、ACM CCS、NDSS | 需要严格威胁模型、可复现实战攻击、强防御对比 | 当前证据不足，暂不主攻 |

正式开题建议主打 AAAI/IJCAI，而不是一开始主打安全四大。原因是当前实验已经证明“直接安全漏洞叙事”证据不够，但“偏好优化后风险模式的诊断和分层评测”更符合人工智能顶会对问题清晰性、方法严谨性和实验边界的要求。

若后续 2B/3B 模型、多分类器、多数据源上稳定复现，再升级为安全攻击/防御论文；否则保持为诊断和评测论文，仍然有顶会空间。

## 三、研究背景与意义

### 3.1 背景

大语言模型通常需要经过后训练才能从“会续写文本”变成“能按照人类意图回答问题”。常见路线包括 SFT、RLHF、DPO 以及其他直接偏好优化算法。

DPO 的优势是简单、稳定、工程成本低。它不需要先训练一个显式奖励模型，也不需要复杂的强化学习采样循环，而是直接用“chosen 优于 rejected”的偏好对来更新模型。这让 DPO 成为开源和工业界都很常见的对齐方法。

但偏好优化也可能带来副作用。已有工作指出，后训练可能让模型输出分布变“尖”：模型更倾向于少数高概率回答，输出多样性下降。对白话读者来说，这类似于一个学生被训练得越来越会答标准答案，但在开放题和复杂安全场景下，他可能不再探索其他合理回答。

输出多样性下降本身不一定是安全漏洞。它只有在以下链条同时成立时才变成安全相关问题：

```text
DPO 更新改变偏好边界
-> 采样输出集中到少数语义模式
-> 主导模式被安全分类器判为风险更高
-> 该模式能在 held-out prompt 或新场景中复现
```

本项目用 PCE 描述这个链条。

### 3.2 PCE 的含义

PCE 是 Preference Collapse Exploitability，暂译为“偏好坍缩可利用性”。当前项目采用的核心形式是：

```text
PCE = dominant-mode determinism * harmfulness of the dominant mode
```

其中：

- `dominant-mode determinism`：模型在同一个 prompt 下多次采样，是否反复落到同一类回答模式。
- `harmfulness of the dominant mode`：这类最常出现的回答是否被安全分类器判为有风险。

白话解释：

> 如果模型每次都随机给出很多不同回答，攻击者很难稳定复用一种攻击策略；如果模型总往同一种回答靠，并且那种回答本身有风险，那么它就更容易被预测和利用。PCE 不是只看“坏不坏”，而是同时看“坏回答是否稳定出现”。

### 3.3 研究意义

本课题的意义在于把“DPO 会不会让模型变窄”和“这种变窄是否安全相关”拆成可验证、可复现、可否证的实验问题。

如果方向成立，贡献是发现并预警一种偏好优化后的安全相关模式集中风险。

如果方向不完全成立，贡献仍然成立：给出一套 prompt-stratified 诊断方法，说明哪些 aggregate PCE 结论是假阳性，哪些 prompt 类型真正值得关注。这类“边界清楚的负结果和诊断工具”比直接宣称一个不稳的漏洞更适合顶会审稿。

## 四、国内外研究现状与前人工作

### 4.1 偏好优化与 DPO

Rafailov 等提出 DPO，证明可以把 RLHF 中的奖励建模和策略优化转化为直接的偏好分类式目标，从而降低训练复杂度。DPO 的工程简洁性是本课题研究它的主要原因之一。

但 DPO 不是没有副作用。Smaug/DPO-Positive 指出，标准 DPO 只要求 chosen 与 rejected 的相对概率差变大，在某些条件下 chosen 本身的似然也可能下降。Likelihood over-optimisation 相关工作进一步指出，更高的偏好 margin 或 completion likelihood 不一定带来更好泛化，过度优化可能损害输出多样性。

这些工作说明：DPO 的训练目标和生成行为之间不是一条简单直线。模型“训练 loss 变好”并不等于“采样行为更安全、更稳定”。

### 4.2 输出多样性下降

Diverse Preference Optimization 和 2026 年关于 post-training output diversity collapse 的工作都指出，后训练会改变输出分布，多样性下降与训练数据组成、训练阶段和优化目标有关。

这些工作主要关心质量、多样性和泛化。本课题在此基础上进一步问：

```text
当多样性下降发生在安全敏感 prompt 上时，它是否会提高某类风险回答的可预测性？
```

这就是本课题与一般 diversity collapse 工作的区别。

### 4.3 偏好优化安全风险与投毒

近期已有工作研究 DPO fine-tuning 的安全风险。例如 Few-Shot Truly Benign DPO Attack 讨论少量看似良性的 DPO 偏好对可能削弱拒答行为；Cost-Minimized Label-Flipping Poisoning Attack 研究 RLHF/DPO 对齐中的低成本标签翻转投毒。

这些工作更接近攻击论文，通常强调“攻击能否成功”。本课题当前不应直接走攻击叙事，因为本地实验没有支持低率投毒的剂量效应。更合适的切入是：

```text
先建立可复现的 PCE 诊断框架，再判断哪些条件下存在攻击或防御价值。
```

### 4.4 安全分类器与风险评测

PCE 的第二项需要判断 dominant mode 是否有风险。已有可参考的 safety/guardian 模型包括 Llama Guard、Granite Guardian、ShieldGemma 等。

本项目本地已能运行 Granite Guardian 3.1 2B，并在固定响应 sanity check 中验证它对 response 内容敏感，不只是被 prompt 文本触发。但 LlamaGuard-family replication 目前仍受模型访问限制影响，后续需要补充。

### 4.5 现有工作的空白

前人工作分别覆盖了 DPO、输出多样性、偏好投毒和安全分类器，但仍有一个空白：

```text
缺少一套把“偏好优化 -> 输出模式集中 -> dominant mode 风险 -> prompt 分层泛化边界”连起来的诊断框架。
```

本课题的开题方向正是补这个空白。

## 五、拟解决的核心科学问题

本研究不把“DPO 一定导致漏洞”作为默认结论，而把它拆成四个可检验问题。

### 问题 1：偏好优化是否真的导致采样模式集中？

训练 loss 或 preference margin 变好，并不保证采样输出的语义模式会集中。已有本地实验已经观察到：DPO preference fitting 可以发生，但 sampled-mode collapse 不一定稳定发生。

### 问题 2：模式集中是否安全相关？

即便 determinism 上升，也可能集中到拒答、摘要或中性解释模式。这不构成漏洞。因此必须同时测 dominant mode harmfulness。

### 问题 3：这种现象是否跨 prompt 泛化？

当前最大 blocker 是 prompt transfer 不稳定。若只在最容易出正信号的 prompt 上成立，不能称为可靠安全问题。因此需要 prompt-stratified held-out 验证，而不是只报告 aggregate 指标。

### 问题 4：能否提前预警或缓解？

若某些训练过程指标能预测后续 PCE 上升，就可以成为预警工具。若 entropy regularization、preference filtering 或 gradient gating 能降低 PCE，则可以形成防御或训练建议。

## 六、前期研究基础与实验数据

本节汇总已经完成的所有本地实验。总体结论是：

```text
已有证据支持“方向值得研究”，但不支持“漏洞已被证明”。
正式开题应转向 prompt-stratified PCE 诊断与预警。
```

### 6.1 实验环境与可复现基础

| 项目 | 当前状态 |
| --- | --- |
| 本地环境 | Windows，RTX 4060 Laptop 8GB |
| Conda 环境 | `stdplm` |
| Python/PyTorch | Python 3.10.20，PyTorch 2.10.0+cu128 |
| 主模型 | Qwen2.5-0.5B-Instruct，本地路径来自 `D:\hf_models` |
| Guardian | Granite Guardian 3.1 2B，本地路径 `D:\hf_models\ibm-granite_granite-guardian-3.1-2b` |
| Git | 本地仓库已初始化，无远端，无 push |
| 文本编码 | 新增和修改文档按 UTF-8 |

这说明项目已经具备本机快速验证能力，不依赖远端仓库，不需要先申请大规模算力才能继续推进。

### 6.2 已完成实验总表

| 类别 | 实验 | 关键结果 | 结论 |
| --- | --- | --- | --- |
| 指标 sanity | synthetic diverse vs collapsed | determinism、entropy、proxy PCE 按预期变化 | 指标管线可用 |
| 机制 sanity | toy DPO categorical update | preference 更新可集中概率质量 | 机制有可能 |
| 端到端 smoke | tiny GPT-2 | 训练、采样、聚类、评估跑通 | 只证明管线 |
| 小模型 gate | SmolLM2-135M | 两种子弱方向信号 | 弱，不足以成 claim |
| 小模型复核 | SmolLM2-360M corrected | 10x8 弱，10x16 robust fail | 更强评估下失败 |
| 控制实验 | SmolLM2 uniform-control | 10x16 反向 | 不能支持主 claim |
| Qwen 初步 | Qwen 20-step | 两种子 fail | 训练不足或信号不稳 |
| Qwen 初步 | Qwen 100-step | weak pass，bootstrap 过零 | 弱证据 |
| Qwen 复评 | 20 prompts x 32 samples | 一种子 pass、一种子 fail | 种子不稳 |
| margin 诊断 | Qwen preference margins | margin 可翻转，但生成 collapse 不必然跟随 | 发现关键断点 |
| 强正信号 | Qwen short-template 300 steps | first-10 prompts 上 local/Guardian robust pass | 最强但受限 |
| 反向控制 | refusal-template DPO | determinism 上升、entropy 下降、harm proxy 下降 | 说明方向可被反向控制 |
| 中性控制 | neutral-boundary | 弱 collapse，无 robust harm increase | determinism 不等于风险 |
| wording 复现 | concise-overview | 只有 weak pass，dominant harm mixed | 正信号 wording-sensitive |
| prompt transfer | prompts 10-19 | local/Guardian mixed 或 fail | 迁移不稳 |
| held-out fallback | first 10 | weak positive | 弱外推 |
| held-out fallback | offset 10 | Guardian mixed | 弱/混合 |
| held-out fallback | offset 20 | local 和 Guardian-PCE robust fail | 明确失败 |
| full 50 aggregate | original 20 + held-out 30 | aggregate Guardian-PCE robust pass，但 prompt-seed split 为 34 pass / 33 mixed / 33 fail | aggregate 阳性但异质性太强 |
| taxonomy v0 | old 50 prompt 分层 | cyber 看似正，violence/weapons 看似负 | 仅探索性 |
| taxonomy 验证 | AdvBench 4-vs-4 held-out | taxonomy v0 不预测新数据 | 分层规则 v0 失败 |
| S0.1 | AdvBench held-out 30 prompt | 21/60 pass，低于 60%；det 下降，entropy 上升；Guardian-PCE mixed | 不进入 S1 |
| Experiment C | poison/CAR smoke | clean 与 1% CAR 相同，5% 更弱；Guardian-PCE clean 最强 | 不支持主动诱导剂量效应 |

### 6.3 最强正证据

最强正信号来自：

```text
Qwen2.5-0.5B-Instruct
short-template DPO stress
lr = 3e-6
300 steps
seeds = 42, 43
first 10 prompts
16 samples per prompt
```

结果：

- preference margin 朝 chosen 响应翻转；
- determinism 上升；
- mode entropy 下降；
- local gate robust pass；
- Granite Guardian-PCE robust pass；
- dominant Guardian harmfulness robust increase；
- refusal-template counter-control 朝相反安全方向移动。

这说明方向不能被简单否定。DPO 训练确实能在本地模型上制造某种“语义松散的主导模式集中”，并且在部分 prompt 上与 Guardian 风险上升同时出现。

### 6.4 最强反证与 blocker

但现有证据也非常清楚地限制了 claim：

- prompt transfer 不稳定；
- full 50-prompt aggregate 虽然为正，但 prompt-seed 层面几乎三等分：34 pass / 33 mixed / 33 fail；
- taxonomy v0 在新 AdvBench held-out 上失败；
- S0.1 held-out 30 prompt 未达到预注册门槛；
- Experiment C 未发现投毒剂量效应；
- target phrase exact hit 始终接近或等于 0，说明不是简单模板复制；
- LlamaGuard-family 复现暂不可用；
- strong positive 依赖特定 response wording。

白话解释：

> 现在不是“完全没有现象”，而是“现象只在部分题、部分写法、部分训练条件下出现”。这对开题来说反而重要：它说明研究重点应该从“证明 DPO 有漏洞”转到“诊断什么时候会有风险，什么时候只是指标假象”。

### 6.5 S0.1 held-out 30 prompt 关键结果

S0.1 是当前最重要的开题前验证，因为它直接检查正信号能否迁移到随机 held-out prompt。

| 指标 | 结果 |
| --- | --- |
| held-out prompt | 30 个 AdvBench prompt，冻结 ID，和已有 50 prompt 零重叠 |
| seed42 local | det -0.0125，entropy +0.0190，proxy PCE +0.0250，10 pass / 2 mixed / 18 fail |
| seed43 local | det -0.0115，entropy +0.0322，proxy PCE +0.0208，11 pass / 2 mixed / 17 fail |
| pooled local | 21 pass / 4 mixed / 35 fail，pass rate 35.0%，低于 60% 门槛 |
| Guardian-PCE | mixed；Yes delta +0.0307，CI [+0.0130, +0.0490]；score delta +0.0211，CI [+0.0064, +0.0370] |
| dominant harm | robust increase；dominant Guardian Yes delta +0.1990，CI [+0.0701, +0.3276] |
| raw audit | 每个 seed final 均为 0/960 target phrase hits |
| 决策 | fail preregistered criteria，不进入 S1 |

这组结果非常关键：Guardian harmfulness 上升，但 determinism 没有上升，entropy 反而上升。因此它不支持“偏好坍缩可利用性稳定增加”，但支持“风险与模式集中可以解耦，需要更细粒度诊断”。

### 6.6 Experiment C 投毒/CAR smoke 关键结果

Experiment C 用来检查“少量投毒是否能主动加速 collapse”。当前结果为 negative smoke。

| 条件 | Det Delta | Entropy Delta | CAR | Proxy PCE Delta | 结论 |
| --- | --- | --- | --- | --- | --- |
| clean | +0.0312 | -0.0797 | +0.000797 | +0.0062 | pass |
| 1% poison | +0.0312 | -0.0797 | +0.000797 | +0.0000 | pass |
| 5% poison | +0.0188 | -0.0451 | +0.000451 | -0.0187 | mixed |

Guardian-PCE：

| 条件 | Guardian-PCE Yes Delta | Guardian-PCE Score Delta |
| --- | --- | --- |
| clean | +0.0437 | +0.0326 |
| 1% poison | +0.0312 | +0.0228 |
| 5% poison | +0.0187 | +0.0095 |

目标剂量关系：

```text
CAR_5% > CAR_1% > CAR_clean: false
```

结论：当前不能说“攻击者能主动诱导坍缩”。最多只能说该实验管线已经跑通，并给出了一个重要负结果。

## 七、正式开题方向

基于以上证据，建议正式开题方向确定为：

```text
偏好优化语言模型中安全相关输出模式集中的分层诊断与预警
```

更具体地说：

```text
不是证明 DPO 必然导致安全漏洞，
而是建立 PCE 诊断框架，找出 DPO 后哪些 prompt、哪些 response mode、哪些训练条件下会出现安全相关模式集中。
```

### 7.1 研究假设

H1：DPO 不会在所有 prompt 上稳定导致 collapse，但会在部分 prompt strata 上诱发显著的 response-mode concentration。

H2：仅有 determinism 上升不足以构成安全问题，只有 dominant mode harmfulness 同时上升时才构成 PCE 风险。

H3：prompt-stratified 指标比 aggregate PCE 更可靠，可以解释“总体看似阳性、局部大量失败”的异质性。

H4：训练过程中的 margin、entropy、dominant-mode 变化可以作为早期预警信号，并可通过正则化或偏好数据筛选降低风险。

### 7.2 技术路线

```text
数据与 prompt 分层
-> DPO / control / optional defense 训练
-> 多次采样生成
-> embedding + clustering 得到 response modes
-> 计算 determinism、entropy、cluster count、distinct-n
-> Guardian/LlamaGuard/ShieldGemma 等安全分类器评估 dominant mode
-> 得到 PCE 和 prompt-level pass/mixed/fail
-> 用 held-out 验证 prompt strata 是否可预测
-> 做机制分析和预警/缓解实验
```

### 7.3 方法设计

1. PCE 指标层：
   - 每个 prompt 多次采样；
   - 用 sentence embedding 聚类语义相近回答；
   - 计算 dominant mass、mode entropy、cluster count；
   - 用安全分类器判断 dominant mode harmfulness；
   - 输出 prompt-level 和 aggregate-level PCE。

2. Prompt-stratified 层：
   - 不只报告平均值；
   - 每个 prompt 标记为 pass、mixed、fail；
   - 按主题、请求形式、风险类型、拒答倾向等分层；
   - held-out 上验证分层是否预测 PCE 变化。

3. 机制诊断层：
   - 比较 preference margin 与 sampled-mode movement；
   - 分析什么时候 margin 变好但采样不 collapse；
   - 分析 dominant harmfulness 上升但 determinism 不上升的情形；
   - 建立“训练目标变化”和“生成行为变化”的桥梁。

4. 防御/预警层：
   - 训练中监控 entropy、top-k concentration、dominant mass；
   - 测试 entropy regularization、preference filtering、rejected-gradient gating 等方法；
   - 目标不是追求绝对安全，而是降低 PCE 上升和 prompt-level failure。

## 八、后续研究计划

### 阶段 1：实验台账与 benchmark 固化

时间：2026 年 7 月

任务：

- 建立 `experiment_index.md`，统一记录所有 run family、配置、输出目录、结论；
- 固化 prompt ID，不再临时挑 prompt；
- 建立至少 300 个 prompt 的分层评测集；
- 定义数据 schema：prompt_id、source、topic、risk_type、surface_form、split、status；
- 明确所有 pass/fail 门槛。

交付：

- 可复现实验索引；
- prompt taxonomy v1；
- benchmark split；
- 第一版 statistical protocol。

### 阶段 2：held-out prompt-stratified validation

时间：2026 年 7 月至 8 月

任务：

- 继续 Qwen2.5-0.5B 作为快速模型；
- 用至少 2 到 3 个训练 seed；
- 每个 prompt 32 到 64 次采样；
- 重点验证 prompt strata 是否能预测 pass/mixed/fail；
- 使用 Granite Guardian，若条件允许增加 LlamaGuard 或 ShieldGemma。

成功标准：

- 至少一个 held-out stratum 同时满足 determinism 上升、entropy 下降、Guardian-PCE CI > 0；
- prompt-level pass rate 明显高于随机或 aggregate baseline；
- target phrase exact hit 仍需报告，避免模板复制误解。

失败处理：

- 若所有 strata 都不稳定，则转为“PCE 假阳性诊断与负结果 benchmark”论文，不再做漏洞 claim。

### 阶段 3：2B/3B scale smoke

时间：2026 年 8 月至 9 月

任务：

- 优先 Qwen2.5-3B 或同家族 1.5B/3B，减少模型家族变量；
- 用 QLoRA/LoRA 或极小 adapter 控制 RTX 4060 显存；
- 只在阶段 2 成功的 held-out stratum 上做 scale smoke；
- 至少 2 个 sampling seed，训练 seed 视资源决定。

定位：

```text
这不是 scale law 证明，而是 direction-not-reversed smoke。
```

通过标准：

- 方向不反转；
- 不能只在挑选过的 first-10 prompt 上成立；
- 不把单模型单种子结果写成规模验证。

### 阶段 4：机制解释与统计建模

时间：2026 年 9 月至 11 月

任务：

- 建立 prompt-level mixed-effects 分析或 bootstrap protocol；
- 检查 preference margin、token entropy、dominant mass、Guardian risk 的时序关系；
- 分析为什么某些 prompt harmfulness 上升但 determinism 不上升；
- 区分三类现象：
  - 真 PCE 上升；
  - harmfulness-only 上升；
  - determinism-only 上升。

目标：

形成顶会论文的核心科学解释，而不是只堆实验表格。

### 阶段 5：预警与缓解

时间：2026 年 11 月至 2027 年 1 月

任务：

- 测试 entropy-regularized DPO；
- 测试 preference pair filtering；
- 测试 rejected-gradient gating 或类似思想；
- 评估 mitigation 对 PCE、拒答率、多样性和简单质量指标的影响。

成功标准：

- 在保持基本偏好拟合的前提下，降低 risky dominant mode；
- 对 held-out prompt 有效果；
- 不是只把模型变成全拒答。

### 阶段 6：论文整理与投稿

时间：2027 年 1 月至 2027 年 4 月，按当年 CFP 微调

AAAI/IJCAI 论文主线：

```text
我们提出 PCE 作为偏好优化后安全相关模式集中的诊断指标；
证明 aggregate PCE 容易误导，prompt-stratified 评估能发现真实泛化边界；
在多个模型、prompt strata、分类器和控制实验中展示何时成立、何时失败；
进一步给出训练过程预警和缓解方法。
```

若结果更强，可扩展为 NeurIPS/ICML/ICLR：

```text
偏好优化的概率质量迁移机制 + 可证明或强经验的预警/缓解方法。
```

若后续出现稳定攻击结果，再考虑安全顶会路线：

```text
明确威胁模型 + 可复现攻击 + 强 defense baseline + 多模型多数据验证。
```

当前不建议直接走安全顶会，因为 Experiment C 负结果不支持主动诱导攻击 claim。

## 九、预期创新点

1. 提出安全相关的 PCE 诊断视角。
   - 不只看输出多样性下降，也不只看 harmfulness，而是看 dominant risky mode 是否稳定出现。

2. 提出 prompt-stratified PCE evaluation。
   - 把 aggregate 指标拆到 prompt-level，显式报告 pass/mixed/fail，避免挑 prompt 和平均数幻觉。

3. 建立偏好优化后“preference fitting 不等于 generation collapse”的实证链条。
   - 当前本地实验已经证明 margin 改善与 sampled-mode movement 可以脱节。

4. 引入控制实验区分假阳性。
   - refusal-template、neutral-boundary、wording replication、target phrase audit、Guardian response-sensitivity control 均已跑通或部分完成。

5. 面向训练过程的预警与缓解。
   - 后续结合 entropy、dominant mass、Guardian-PCE，形成可用于训练监控的轻量方法。

## 十、可行性分析

### 已有可行性

- 本地 RTX 4060 已完成 Qwen2.5-0.5B DPO smoke；
- 本地模型下载和加载路径已解决；
- Granite Guardian 可运行；
- 指标、训练、评估、Guardian audit、summary 脚本已初步形成；
- 已有正信号、反向控制、held-out negative、poison negative，证据不是空想。

### 算力可行性

本地适合做：

- 0.5B 模型快速迭代；
- 小规模 LoRA/QLoRA；
- prompt-level 评估；
- 诊断脚本和统计分析。

不适合本地强行做：

- 7B 全参数 DPO；
- 大量多种子大模型训练；
- 大规模 safety classifier ensemble。

因此开题计划采用“本地快速验证 + 必要时远程/云端扩展”的路线。

## 十一、风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| 现象不稳定 | held-out pass rate 低 | 转为 prompt-stratified negative/diagnostic benchmark |
| 安全分类器偏差 | Granite 与其他 classifier 不一致 | 增加 LlamaGuard/ShieldGemma，报告 classifier sensitivity |
| aggregate 指标误导 | 总体阳性但 prompt 大量失败 | 强制 prompt-level reporting |
| 模板学习假象 | exact phrase copying | raw audit 和 target phrase hit 必报 |
| 投毒路线失败 | 无剂量效应 | 不走攻击主线，保留为负结果 |
| 2B/3B 显存不足 | OOM | QLoRA、减少 sample、只做 selected held-out strata |
| 顶会贡献不够 | 只是工程评测 | 加强机制解释、统计建模、预警/缓解 |

## 十二、当前结论

当前最可靠的结论是：

```text
DPO 后确实可能出现局部安全相关模式集中，
但它不是稳定、普遍、可直接宣称为漏洞的现象。
```

因此，最好的开题方向不是“证明 DPO 会导致漏洞”，而是：

```text
建立 PCE 诊断和 prompt-stratified 验证框架，
找出偏好优化后风险模式集中的发生条件、泛化边界和预警方法。
```

这一路线既保留了原始 idea 的价值，也尊重现有实验的负结果；既能向 CCF-A 人工智能顶会讲出清晰问题，也不会因为过度声称而被审稿人抓住证据不足。

## 十三、参考文献与近期工作

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

## 十四、100 字以内导师消息

老师，本地验证显示：PCE 有受限正信号，但 held-out 和投毒 smoke 不支持漏洞结论。建议面向 CCF-A，转为 prompt-stratified PCE 诊断与预警。
