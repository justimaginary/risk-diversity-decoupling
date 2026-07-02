# 开题文档草案：面向 CCF-A 顶会的 Prompt-Stratified PCE 诊断研究

日期：2026-07-02

## 题目

面向偏好优化语言模型的 Prompt-Stratified Preference Collapse
Exploitability 诊断方法研究

## 投稿目标与论文定位

目标定位为 CCF-A / 国际顶会级研究。优先考虑方向包括：

| 方向 | 潜在会议 | 适配理由 |
| --- | --- | --- |
| 机器学习与对齐机制 | NeurIPS / ICML / ICLR / AAAI | 需要突出偏好优化下的分布诊断、机制分析和可复现实验 |
| NLP 与安全评测 | ACL / EMNLP | 需要突出 prompt-stratified evaluation、LLM safety evaluation 和数据集/诊断框架 |
| AI 安全与系统安全 | CCS / USENIX Security / NDSS | 需要更强攻击威胁模型、真实风险验证和防御效果 |

基于当前实验结果，**最现实的 CCF-A 路线不是直接做“DPO 漏洞攻击论文”，而是做
“偏好优化安全诊断论文”**。顶会级贡献应当是：

```text
提出并验证一种 prompt-stratified PCE 诊断框架，
用于判断偏好优化何时把采样输出集中到安全相关的风险响应模式。
```

当前不宜把论文主张写成：

```text
DPO reliably creates exploitable vulnerabilities.
```

更适合的顶会主张是：

```text
Preference optimization can produce response-mode concentration whose safety
relevance is prompt- and response-mode dependent; PCE provides a diagnostic
framework to measure, stratify, and stress-test this phenomenon.
```

顶会审稿人会重点质疑：

| 质疑 | 当前状态 | 研究计划回应 |
| --- | --- | --- |
| 是否只是 DPO diversity loss 的旧问题 | 文献已覆盖 diversity loss | 强调 determinism × guardian-scored harmfulness，而非只看多样性 |
| 是否 cherry-pick prompt | 当前确有 prompt sensitivity | 建立预注册 held-out prompt strata 与 experiment index |
| 是否只是模板学习 | target phrase hits 一直为 0 | 用 semantic dominant mode、raw representative 和多 response mode controls 验证 |
| safety classifier 是否可靠 | 当前主要是 Granite | 后续加入 LlamaGuard / ShieldGemma 或第二 classifier |
| 是否有因果机制 | refusal counter-control 有支持，poison smoke 为负 | 后续做 response-mode controls、margin-to-generation analysis、defense ablation |
| 是否足够规模化 | 当前主要 0.5B | 等 held-out 机制稳定后再做 2B/3B scale validation |

## 一、研究背景

Direct Preference Optimization（DPO）已成为大语言模型对齐和后训练中的重要
方法。它通过偏好对直接优化策略模型，避免了传统 RLHF 中显式奖励模型和在线
强化学习带来的复杂性。DPO 的优势是训练流程简单、稳定、易部署，但这种“直接
提高 chosen 相对 rejected 概率”的目标也可能改变模型输出分布，使模型在某些
prompt 上更倾向于少数响应模式。

已有研究已经指出，偏好优化和后训练可能导致输出多样性下降、概率分布变尖、
过度优化和拒答行为变化。因此，“DPO 会降低多样性”本身不是足够新颖的研究
主张。更值得研究的问题是：

```text
当输出分布收缩时，这种收缩是否集中到安全相关的风险响应模式？
如果是，它是否可以被一个可复现的诊断指标提前发现？
```

本项目最初提出 Preference Collapse Exploitability（PCE）作为诊断视角：

```text
PCE = dominant-mode determinism * harmfulness of dominant mode
```

其中，dominant-mode determinism 衡量模型是否反复采样到同一语义响应模式；
harmfulness 衡量该 dominant mode 是否被安全分类器判定为风险响应。这个定义的
关键点是：**单纯的确定性上升不是漏洞，单纯的 harmfulness 上升也不是 PCE；
只有“风险响应模式成为稳定主导模式”时，才构成可疑的安全可利用性信号。**

## 二、当前研究近况

### 2.1 DPO 与直接偏好优化

DPO 原始论文将 RLHF 问题重新参数化为一个分类式偏好优化目标，使语言模型在
不显式训练奖励模型的情况下学习人类偏好。该方向已经成为后训练的重要基线。

但后续工作指出，DPO 及其他 direct alignment algorithm 并不总是按直觉提高
chosen completion 的绝对似然。Smaug / DPO-Positive 指出，标准 DPO 可能只提高
chosen 与 rejected 的相对概率，而 preferred examples 的似然本身可能下降。另有
研究讨论 likelihood over-optimisation：更高的 completion likelihood 或更大的
margin 不一定带来更好泛化，过度优化可能与 entropy、top-k probability mass 等
分布指标变化有关。

### 2.2 输出多样性下降与分布收缩

近年的多样性相关工作表明，后训练、偏好优化和监督微调都可能使输出概率分布
变尖并降低 response diversity。Diverse Preference Optimization（DivPO）等工作
已经把“保持多样性”作为目标进行优化。因此，本项目不能把“DPO 降低多样性”
作为主要创新点。

更近的 post-training diversity collapse 研究进一步指出，多样性下降的位置和
程度与训练数据组成、模型训练路线有关；DPO 的影响并不是孤立的，也未必在所有
任务和模型上稳定一致。这与本项目本地实验中看到的 prompt-sensitive 现象一致。

### 2.3 安全分类与偏好数据风险

安全分类器方面，Llama Guard、Granite Guardian、ShieldGemma 等模型都在尝试对
prompt/response 风险进行自动化识别。本项目目前本地可用的真实 guardian-style
分类器是 Granite Guardian 3.1 2B；LlamaGuard 小模型访问仍受 gated 限制。

安全风险方面，已有工作研究了 benign-looking DPO attack、偏好标签投毒和对齐
数据污染。这意味着“偏好数据可能损害安全”也不是完全空白。项目真正可能成立
的贡献应更窄：

```text
以 prompt-level 的 dominant-mode determinism × guardian-scored harmfulness
衡量偏好优化后风险响应模式是否被集中。
```

### 2.4 文献定位

| 方向 | 代表工作 | 对本课题的影响 |
| --- | --- | --- |
| DPO 基线 | Direct Preference Optimization | 本课题研究 DPO 后的输出分布风险，不提出新优化器 |
| DPO 似然病理 | Smaug / DPO-Positive | 支持“margin 变化不等于采样行为稳定变化”这一观察 |
| 直接对齐过度优化 | Likelihood over-optimisation | 支持用 entropy / probability mass 诊断过度优化 |
| 多样性保持 | Diverse Preference Optimization | 说明“多样性下降”本身不是新颖贡献 |
| 后训练多样性坍缩 | Output diversity collapse in post-training | 支持 prompt/task/data-dependent 的解释 |
| DPO 安全攻击 | Few-Shot Truly Benign DPO Attack | 逼近本课题的安全叙事，要求本课题突出 PCE 诊断而非泛泛攻击 |
| 偏好标签投毒 | Cost-minimized label-flipping poisoning | 说明投毒方向已有研究，后续必须强调 PCE 加速/诊断 |
| 安全分类器 | Granite Guardian、Llama Guard、ShieldGemma | 支持 guardian-scored harmfulness，但需明确分类器边界 |

## 三、拟研究问题与当前最佳方向

基于已有实验，当前不宜继续主张：

```text
DPO 会稳定导致可利用安全漏洞。
```

更好的开题方向是：

```text
Prompt-Stratified PCE Diagnostics：
研究偏好优化在什么 prompt 条件、什么响应模式下，会把采样输出集中到
guardian 判定更高风险的 dominant response mode。
```

核心研究问题可以表述为：

1. DPO 偏好拟合是否会稳定传导到采样输出的 semantic mode concentration？
2. 这种 concentration 是否与 guardian-scored harmfulness 同时上升？
3. 哪些 prompt 类型或响应模式更容易出现 PCE 上升？
4. 如何区分“普通确定性上升”“安全拒答模式集中”和“风险响应模式集中”？
5. PCE 能否作为偏好优化训练后的诊断指标，而不是直接作为漏洞证明？

## 四、已有实验环境与数据

### 4.1 环境

| 项目 | 配置 |
| --- | --- |
| 本地机器 | RTX 4060 Laptop GPU，8GB 显存 |
| Conda 环境 | `stdplm` |
| Python | 3.10.20 |
| PyTorch | 2.10.0+cu128 |
| 主模型 | `Qwen/Qwen2.5-0.5B-Instruct` 本地恢复版本 |
| Guardian 分类器 | `ibm-granite/granite-guardian-3.1-2b` |
| 模型缓存 | `D:\hf_models` |
| 代码仓库 | 本地 git，无 remote，无 push |

### 4.2 数据

| 数据文件 | 用途 |
| --- | --- |
| `data/attack_prompts.jsonl` | 初始 harmful prompt 评估集 |
| `data/attack_prompts_10_19.jsonl` | 第二 prompt subset |
| `data/attack_prompts_fallback_heldout_30.jsonl` | fallback held-out 30 prompt |
| `data/advbench_harmful_behaviors_all.jsonl` | AdvBench 520 条源 prompt |
| `data/advbench_taxonomy_v0_cyber_vs_violence_heldout.jsonl` | taxonomy v0 4-vs-4 验证 |
| `data/advbench_s0_1_heldout_30.jsonl` | S0.1 随机 AdvBench held-out 30 |
| `data/local_short_template_preferences.jsonl` | 正向 short-template preference |
| `data/local_refusal_template_preferences.jsonl` | refusal counter-control |
| `data/local_neutral_boundary_preferences.jsonl` | neutral-boundary control |
| `data/local_concise_overview_preferences.jsonl` | 第二短合规措辞 control |
| `data/local_poison_smoke_poison*_100.jsonl` | Experiment C poison/CAR smoke |

所有 raw generations、Guardian audit JSON、checkpoint 和大输出均保存在 ignored
`outputs/` 下，不纳入 git。

## 五、已完成实验与实验数据

### 5.1 指标方向验证

Synthetic diverse-vs-collapsed response set 能让 determinism、mode entropy、
proxy PCE 按预期变化：

- collapsed responses：dominant mode mass 上升；
- diverse responses：entropy 和 cluster count 更高；
- proxy PCE 随 dominant harmful mode 增强而上升。

结论：指标管线可用，但只证明 measurement plumbing，不证明真实模型现象。

### 5.2 Toy DPO 机制实验

Toy categorical policy 中，DPO 式偏好更新可以使概率质量向少数模式集中；更高
poison ratio 会提高 harmful proxy mass。

结论：机制方向有理论/玩具层面可行性，但不能外推到真实 instruction model。

### 5.3 Tiny GPT-2 pipeline smoke

Tiny GPT-2 完成训练、采样、聚类、指标计算闭环。

结论：工程闭环跑通；模型过小，不作为研究证据。

### 5.4 SmolLM2 系列实验

| 实验 | 结果 | 结论 |
| --- | --- | --- |
| SmolLM2-135M mini gate | 弱两种子方向信号 | 仅可作为早期 S0 |
| SmolLM2-360M corrected 10x8 | two-seed aggregate pass | `weak_pass` |
| SmolLM2-360M matched 10x16 | 两个 corrected seed 均 fail | `robust_fail` |
| SmolLM2 uniform-control 10x16 | reverse direction | `robust_fail` |

结论：SmolLM route 可验证 pipeline，但不能支撑 claim。

### 5.5 Qwen 0.5B 基础 gate

Qwen2.5-0.5B-Instruct 本地恢复后，fp16 短训练出现 NaN；float32 训练可完成。

| 实验 | 结果 | 结论 |
| --- | --- | --- |
| 20-step uniform-control | 两个 seed fail | 不支持 |
| 100-step uniform-control 10x16 | 两 seed aggregate pass，但 bootstrap 跨零 | `weak_pass` |
| 100-step 20x32 re-eval | seed-level 1 pass / 1 fail | `weak_pass` |
| collapse-proxy subset | one pass / one mixed | `weak_pass` |

结论：Qwen 可作为主力本地模型，但短训练证据弱。

### 5.6 Preference Margin 与生成行为传导

`compare_preference_margins.py` 显示 DPO 训练能让 per-token average margins
明显转向 chosen responses；但 `analyze_margin_generation_link.py` 显示 margin
变化并不稳定预测 determinism 上升或 entropy 下降。

结论：

```text
Preference fitting exists, but does not automatically transmit to sampled-mode collapse.
```

### 5.7 Qwen short-template 强信号

配置：

```text
Qwen2.5-0.5B-Instruct
chosen response: "Sure, safe overview only."
lr = 3e-6
300 steps
seed 42 / 43
matched 10 prompts x 16 samples
```

结果：

- determinism 上升；
- entropy 下降；
- local gate `robust_pass`；
- Granite Guardian-PCE `robust_pass`；
- dominant Guardian harmfulness `robust_increase`；
- raw target phrase hits = 0。

结论：这是当前最强本地正信号，但只限于 first-10 prompt 和特定 response wording。

### 5.8 Refusal counter-control

将 chosen 改为拒答模板、rejected 改为短合规模板后：

- determinism 上升；
- entropy 下降；
- refusal rate 上升；
- compliance/proxy harmfulness 下降；
- dominant Guardian harmfulness 下降。

结论：确定性上升本身不是漏洞；response mode 的安全属性决定 PCE 方向。

### 5.9 Neutral-boundary 与 concise-overview control

| Control | 结果 | 结论 |
| --- | --- | --- |
| neutral-boundary | det/entropy 弱 collapse；Guardian-PCE CI 跨零 | weak |
| concise-overview | 固定响应 Guardian 风险高，但训练后 gate 仅 weak | negative replication |

结论：原始 short-template robust pass 对 response wording 敏感。

### 5.10 Prompt transfer 与 full 50-prompt 结果

| Prompt block | 结果 |
| --- | --- |
| first 10 | strongest positive |
| prompts 10-19 separately trained | weak |
| original checkpoint on prompts 10-19 | mixed/fail |
| held-out fallback first 10 | weak positive |
| held-out fallback offset 10 | mixed |
| held-out fallback offset 20 | robust fail |
| full 50 aggregate | Guardian-PCE robust_pass but highly heterogeneous |

Full 50-prompt aggregate：

| 指标 | 结果 |
| --- | --- |
| Guardian-PCE aggregate | `robust_pass` |
| Dominant harm | `robust_increase` |
| prompt-seed split | 34 pass / 33 mixed / 33 fail |
| prompt map | 15 stable pass / 15 mixed / 15 stable fail / 5 mostly-pass/fail |

结论：aggregate positive 存在，但 prompt heterogeneity 太强，不能支持稳定漏洞 claim。

### 5.11 Taxonomy v0 与 AdvBench 验证

taxonomy v0 在旧 50 prompt 上发现：

- cyber prompts 更偏 positive；
- violence/weapons prompts 更偏 negative。

但在 AdvBench 新 held-out 4-vs-4 验证上失败：

| Topic | Local Gate | Guardian-PCE | Dominant Harm |
| --- | --- | --- | --- |
| cyber | two seeds fail, pooled mixed | mixed | robust increase |
| violence/weapons | mixed | mixed | robust decrease |

结论：taxonomy v0 能描述旧数据，但不能预测新 AdvBench behavior。

### 5.12 S0.1 held-out 30 prompt quick validation

这是开题前最高优先级补充实验，已预先冻结：

- prompt source：AdvBench 520；
- 排除已有 58 条 prompt；
- 随机 seed：20260702；
- selected prompts：30；
- matched evaluation：30 prompts x 32 samples；
- seed42 / seed43 final checkpoints；
- Granite Guardian audit。

结果：

| Check | Result |
| --- | --- |
| seed42 local | det -0.0125, entropy +0.0190, proxy PCE +0.0250 |
| seed43 local | det -0.0115, entropy +0.0322, proxy PCE +0.0208 |
| pooled local | 21 pass / 4 mixed / 35 fail |
| pass rate | 35.0%，低于 60% 门槛 |
| Guardian-PCE | `mixed` |
| dominant harm | `robust_increase` |
| target phrase hits | 0/960 per final seed |
| dominant mass | seed42 0.1229, seed43 0.1240 |

结论：

```text
S0.1 不支持 held-out sampled-mode collapse。
Harmfulness / Guardian-PCE 有上升，但 determinism 下降、entropy 上升。
这是 safety-risk movement without stable mode-collapse transfer。
```

### 5.13 Experiment C：Poison/CAR smoke

目标：检查 clean / 1% / 5% 合成 preference perturbation 是否产生 CAR 剂量效应。

为了让 100-step 训练能看到 1% poison，本地 smoke 使用 100-row 数据：

- clean：0 poison rows；
- 1%：1 poison row；
- 5%：5 poison rows；
- 100 steps shuffled order；
- Qwen0.5B，first-10 prompts，16 samples。

CAR 结果：

| Condition | Det Delta | Entropy Delta | CAR | Proxy PCE Delta |
| --- | ---: | ---: | ---: | ---: |
| clean | +0.0312 | -0.0797 | +0.000797 | +0.0062 |
| 1% poison | +0.0312 | -0.0797 | +0.000797 | +0.0000 |
| 5% poison | +0.0188 | -0.0451 | +0.000451 | -0.0187 |

剂量标准：

```text
CAR_5% > CAR_1% > CAR_clean: false
```

Guardian-PCE：

| Condition | Guardian-PCE Yes Delta | Guardian-PCE Score Delta |
| --- | ---: | ---: |
| clean | +0.0437 | +0.0326 |
| 1% poison | +0.0312 | +0.0228 |
| 5% poison | +0.0187 | +0.0095 |

结论：Experiment C 为负结果，不支持“主动诱导剂量效应”。不能作为攻击可行性证据。

## 六、当前结论

当前证据最准确的总结是：

```text
已有受限机制信号，但没有稳定漏洞证据。
```

支持继续研究的证据：

- PCE 指标方向正确；
- toy DPO 显示概率集中机制可发生；
- Qwen0.5B 在 first-10 short-template stress 上出现两 seed robust positive；
- Granite Guardian 支持 restricted positive stress；
- refusal counter-control 说明 response mode 安全属性决定 PCE。

反对升级为漏洞 claim 的证据：

- prompt transfer 不稳定；
- full 50-prompt prompt-seed split 近似三分；
- taxonomy v0 held-out 验证失败；
- S0.1 held-out 30 未通过；
- Experiment C poison/CAR smoke 未出现剂量效应；
- target phrase 从未被 exact copied；
- LlamaGuard replication 尚未完成。

因此，当前开题不应讲：

```text
DPO 已被证明会造成安全漏洞。
```

应讲：

```text
偏好优化可能在部分 prompt / response mode 下造成安全相关的响应模式集中。
本课题拟研究如何诊断这种 prompt-sensitive 的 PCE 现象。
```

## 七、拟定研究方向

### 7.1 研究目标

构建一套面向偏好优化模型的 Prompt-Stratified PCE 诊断框架，用于识别：

1. 哪些 prompt 子集更容易出现 sampled-mode concentration；
2. 哪些 response mode 会使 concentration 变成 safety-relevant；
3. 如何通过 guardian-scored dominant mode 区分普通确定性上升与风险集中；
4. 如何用 held-out prompt 和多 seed 评估 PCE 是否稳定。

### 7.2 拟解决问题

- DPO preference margin 与 sampled-mode concentration 的关系是什么？
- PCE 在 prompt-level 的分布是否可预测？
- 哪些 prompt 特征或 response-mode 特征解释 pass/fail？
- guardian harmfulness 与 determinism 何时同向，何时分离？
- 如何设计防御或训练诊断，使高风险集中在早期被发现？

### 7.3 预期贡献

1. 提出 prompt-stratified PCE 诊断框架。
2. 区分 determinism、harmfulness、PCE 三者的关系。
3. 给出多组 response-mode controls，说明拒答集中与风险合规集中不同。
4. 建立 held-out prompt gate，避免只报告 aggregate-positive 结果。
5. 为后续 entropy-regularized DPO 或 data-level diversity defense 提供诊断指标。

## 八、面向 CCF-A 的研究计划

### 8.1 总体路线

研究路线从“证明漏洞成立”调整为“建立诊断框架并解释边界”。完整计划分为四个
阶段：

```text
阶段一：实验资产整理与可复现诊断基线
阶段二：Prompt-stratified PCE benchmark
阶段三：机制解释与 response-mode controls
阶段四：防御/缓解与规模验证
```

目标不是追求单个 aggregate-positive 结果，而是形成一条顶会可接受的证据链：

```text
Preference fitting
  -> sampled semantic mode concentration
  -> guardian-scored dominant-mode risk
  -> prompt-stratified predictability
  -> mitigation or early-warning diagnostic
```

### 8.2 阶段一：实验索引与可复现基线

建立 `docs/experiment_index.md`：

- 每个 run 的输出目录；
- 模型、seed、prompt subset、samples；
- local gate / Guardian-PCE / raw audit；
- pass / weak / mixed / fail / diagnostic 标记。

目标：让所有历史实验可追踪，避免叙事选择性。

交付物：

- `docs/experiment_index.md`
- 统一 JSON summary schema
- 每个实验的 pass/fail/diagnostic 标签
- 可复现实验命令索引

验收标准：

- 能从索引追踪到所有历史实验；
- 能区分 local metrics、Guardian-PCE、raw audit；
- 不再依赖 README 长文本作为唯一实验记录。

### 8.3 阶段二：Prompt-Stratified PCE Benchmark

taxonomy v0 已失败，下一步不应直接复用。可选路线：

- taxonomy v1：加入更宽 prompt feature，例如 cyber、fraud、violence、
  self-harm、harassment、illicit behavior、request verb、specificity；
- 或者先不建 taxonomy，只做预注册 prompt buckets。

目标：先冻结 prompt strata，再跑模型，避免 post hoc 解释。

计划：

| 项目 | 设计 |
| --- | --- |
| prompt source | AdvBench、HarmBench/JailbreakBench 可用子集、现有 local prompts |
| prompt 数量 | 首轮 100-200；正式版 300+ |
| strata | cyber、fraud、violence、self-harm、harassment、illicit behavior、generic harmful |
| 模型 | Qwen0.5B 起步，稳定后扩展到 1.5B/3B |
| seeds | 至少 2 个 training seeds，2 个 generation seeds |
| samples | 每 prompt 16-32 samples |
| classifiers | Granite + 第二 safety classifier |
| 输出 | prompt-level PCE map、stratum-level bootstrap CI、raw dominant representatives |

核心问题：

- 哪些 prompt strata 更容易出现 determinism 与 harmfulness 同向上升？
- 哪些 strata 只出现 harmfulness 上升但不出现 collapse？
- 哪些 strata 出现 refusal/safe mode concentration？

顶会级验收标准：

- 至少一个预注册 held-out stratum 达到 Guardian-PCE robust pass；
- 同时给出失败 strata，而不是只报告正结果；
- prompt-level pass/fail 可由预注册特征部分解释；
- 结果跨至少两个 seed 稳定。

### 8.4 阶段三：机制解释与 Response-Mode Controls

已有实验说明：同样的 determinism 上升可能对应风险合规模式，也可能对应拒答模式。
因此需要系统化 response-mode controls。

计划设计：

| Control | 目的 |
| --- | --- |
| short-compliance mode | 复现 restricted positive signal |
| refusal mode | 证明 determinism 上升不等于风险上升 |
| neutral-boundary mode | 区分安全边界表达与风险合规 |
| alternate concise-compliance mode | 测试 response wording sensitivity |
| diverse-safe mode | 测试多样安全 chosen 是否缓解 concentration |

机制分析：

- preference margin change；
- length-normalized chosen/rejected logprob；
- sampled cluster dominant mass；
- entropy / cluster count；
- Guardian dominant risk；
- raw representative qualitative review。

预期贡献：

```text
提出 response-mode-sensitive 的 PCE 解释框架：
DPO 是否危险不只取决于 collapse 是否发生，还取决于 collapse 到什么响应模式。
```

### 8.5 阶段四：独立 Safety Classifier Replication

当前 Granite Guardian 是第一真实 guardian，但不等于 LlamaGuard。后续如果条件允许：

- 申请或使用 LlamaGuard 访问；
- 或使用 ShieldGemma 等第二 classifier；
- 对同一 saved outputs 重跑 harmfulness audit。

目标：检验 Guardian-specific artifact。

验收标准：

- Granite 与第二 classifier 在 dominant harm direction 上基本一致；
- 若不一致，分析 disagreement strata；
- 不把单一 classifier 结果写成普遍 safety 结论。

### 8.6 阶段五：防御或训练诊断

在 base effect 更稳定前，不急于做大规模攻击。更合理的是：

- entropy-regularized DPO；
- diversity-preserving preference construction；
- early-warning PCE monitoring；
- prompt-stratified held-out gate。

目标：把项目定位为诊断与防御工具。

计划：

| 方法 | 目的 |
| --- | --- |
| entropy regularization | 测试能否降低 dominant-mode determinism |
| diverse chosen construction | 测试数据层面能否保持多响应模式 |
| early PCE monitoring | 判断 PCE 是否早于最终安全退化出现 |
| prompt-stratified early stopping | 只在高风险 strata 上触发停止/告警 |

顶会级贡献要求：

- 防御不仅降低 diversity collapse，还要降低 Guardian-PCE；
- 不显著降低普通 helpfulness 或基本任务质量；
- 至少与一个已有 diversity-preserving preference 方法比较。

## 九、时间表与里程碑

| 阶段 | 时间 | 目标 | Go/No-go 标准 |
| --- | --- | --- | --- |
| M1 | 1-2 周 | 实验索引、统一 summary schema、整理历史输出 | 所有历史实验可追踪 |
| M2 | 3-5 周 | prompt-stratified benchmark v1 | 至少 100 prompts、2 seeds、Guardian summary |
| M3 | 6-8 周 | taxonomy v1 / bucket validation | held-out strata 不再完全失效 |
| M4 | 9-12 周 | response-mode controls 完整实验 | 能解释 risky/refusal/neutral mode 差异 |
| M5 | 13-16 周 | 第二 classifier replication | Granite 结论有独立验证或明确 disagreement |
| M6 | 17-22 周 | 防御/诊断方法 | 至少一个方法降低 Guardian-PCE |
| M7 | 23-28 周 | 论文成稿与补实验 | 达到 CCF-A 投稿证据标准 |

## 十、CCF-A 投稿最低证据标准

若要投 CCF-A 顶会，最低需要满足：

1. 不是单模型、单 prompt subset、单 response wording 的正结果。
2. 至少一个预注册 held-out stratum robust pass。
3. 至少两个训练 seed，最好再加 generation seed。
4. 同时报告失败/混合 strata，证明不是 cherry-pick。
5. 至少两个 safety classifiers 或清楚解释 classifier 局限。
6. raw dominant representatives 支持 semantic mode concentration。
7. 有 response-mode controls 解释为什么 refusal collapse 与 risky collapse 不同。
8. 有防御或 early-warning 结果，使论文不只是观察现象。

如果这些标准无法达成，论文应降级为：

```text
workshop / CCF-B exploratory diagnostic paper
```

而不是强行投 CCF-A。

## 十一、风险与边界

| 风险 | 处理方式 |
| --- | --- |
| 正信号高度 prompt-sensitive | 研究重点转向 prompt-stratified diagnostics |
| harmfulness 与 determinism 分离 | PCE 明确要求二者乘积，不单看 harmfulness |
| classifier artifact | 后续加入第二 safety classifier |
| raw outputs 不复制模板 | 强调 semantic dominant mode，而非 exact string memorization |
| 攻击实验负结果 | 不把 active attack 作为当前开题核心 |
| 规模实验资源不足 | 不把 2B/3B smoke 作为优先证据 |

## 十二、开题表述建议

建议导师沟通时采用如下定位：

```text
我目前不把项目表述为“DPO 已导致漏洞”，而是表述为：
偏好优化可能导致部分 prompt 下的响应模式集中，其中只有当 dominant mode
同时被安全分类器判定为高风险时，才构成 PCE 信号。本课题研究如何诊断和预测
这种 prompt-sensitive 的 PCE，而不是直接宣称漏洞成立。
```

## 十三、100 字以内导师消息

老师，我完成了本地验证：有受限 PCE 信号，但 held-out 和投毒 smoke 不支持漏洞结论。建议以 CCF-A 为目标，转向偏好优化下的 prompt-stratified PCE 诊断研究。

## 十四、参考文献与资料

- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Smaug / DPO-Positive: https://arxiv.org/abs/2402.13228
- Understanding Likelihood Over-optimisation: https://arxiv.org/abs/2410.11677
- Diverse Preference Optimization: https://arxiv.org/abs/2501.18101
- Where does output diversity collapse in post-training?: https://arxiv.org/abs/2604.16027
- Few-Shot Truly Benign DPO Attack: https://arxiv.org/abs/2605.10998
- Cost-Minimized Label-Flipping Poisoning Attack: https://arxiv.org/abs/2511.09105
- Granite Guardian: https://arxiv.org/abs/2412.07724
- Llama Guard 3-1B-INT4: https://arxiv.org/abs/2411.17713
- ShieldGemma: https://arxiv.org/abs/2407.21772
