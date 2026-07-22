# 风险—多样性解耦完整实验报告

更新日期：2026-07-23<br>
当前阶段：R0–R3 已完成，Gate R3 = **Stop**<br>
主模型：Qwen3-1.7B<br>
训练方法：LoRA-DPO<br>
正式实验设备：NVIDIA GeForce RTX 4090 24GB

> 本报告以仓库中已经提交的脱敏 manifest、聚合指标和阶段结果为依据，汇总截至 2026-07-23 的完整实验链路。原始大规模生成、模型权重、逐条 judge 记录和未揭盲人工审计答案不进入 Git。

## 1. 摘要与最终判断

本项目研究偏好后训练是否会让“安全风险总量”和“有害回答的模式多样性”发生可分离的变化。核心关注并不是输出是否简单重复，而是风险增加时，有害内容是否仍分布在多个不同语义或行为模式中。

目前最可靠的结论是：

1. **风险—多样性解耦在受控先导和单 seed 筛选中出现过，但没有通过严格多 seed 主实验。**
2. **干净安全数据 D1 会稳定降低风险，但也同步降低有害语义熵。** 这是风险和有害模式多样性的耦合下降，不是解耦。
3. **修复后的冲突数据 D2 在单 seed R2 中同时提高风险和有害语义熵，但在 R3 中表现出严重训练-seed 异质性。** seed 42/43 的 HarmBench 风险高于 Base，seed 44 则显著低于 Base；三个安全 judge 的跨 seed 方向都不一致。
4. **把每题生成数从 32 增加到 64 没有消除 D2 的异质性。** 因此问题不在生成采样不足，而在训练结果本身不稳定。
5. **强拒答抑制条件 D4 在 KL 合格的 step 30 上反而稳定降低风险和有害多样性，并提高过度拒答。** 它没有复现早期受控先导中的风险上升。
6. **低比例污染 D3 没有显示相对干净训练的稳定风险增量。** 现有证据不支持“5% 有害偏好污染已稳定触发风险上升”。

因此，本报告的最终判断是：

> 当前证据不支持把“偏好优化会稳定造成风险—多样性解耦”作为已成立结论，也不支持把该现象归因于 DPO 的一般性质。最值得保留的科学结果，是数据条件和训练 seed 会强烈改变风险方向，而拒答率、总风险与有害模式多样性必须分开测量。

按预注册停止规则，当前不进入 R4 方法对比，不租用 48GB/80GB GPU。只有在新方案能解释并控制 D2 的训练-seed 异质性时，才值得重开主实验。

## 2. 研究问题与实验目的

本项目从早期的 Preference Collapse Exploitability（PCE）假设出发。早期假设关注偏好训练是否会让模型稳定落入少数危险回答模式；后续实验发现，风险变化和输出集中程度并不总是同向，因此研究问题扩展为两条独立轴：

- **风险轴**：有害率、危险信息有效性、重复调用的尾部风险是否变化；
- **分布轴**：风险集中到少数模式，还是分散在多个不同语义或行为模式中。

正式研究问题为：

- RQ1：风险总量和风险模式多样性是否能独立变化？
- RQ2：这种变化是否依赖训练数据条件？
- RQ3：观察到的现象能否跨独立训练 seed 稳定复现？
- RQ4：现象是否是 DPO 特有，还是训练内容、拒答抑制或训练强度造成？

R0–R3 主要回答 RQ1–RQ3。由于 R3 未通过，RQ4 对应的 R4 方法对比没有启动。

## 3. 数据集与数据条件

### 3.1 训练数据轴

正式实验将训练数据组织为五种条件，而不是只比较“正常数据”和“投毒数据”。

| 条件 | 名称 | 定义 | 实验作用 |
| --- | --- | --- | --- |
| D0 | Clean helpfulness | 普通无害任务中 chosen 更正确或更有用 | 检查普通帮助性训练的安全外溢 |
| D1 | Clean safety | 危险请求中 chosen 更安全、rejected 更危险 | 安全对齐参照 |
| D2 | Helpfulness–safety conflict | chosen 更有帮助但安全边界较弱，rejected 更安全但帮助性较差 | 主要现实冲突条件 |
| D3 | 5% harmful preference contamination | 在干净偏好中混入 5% 反向安全 pair | 低比例污染筛选 |
| D4 | Full refusal-suppression intervention | chosen 更配合，rejected 为安全拒答 | 强压力测试，不代表现实分布 |

主要来源为 UltraFeedback 派生的普通帮助性偏好和 PKU-SafeRLHF 固定子集。D3 中反转或注入的 pair 被显式标记为 `synthetic_poison`。R2 每个条件固定 200 pairs，并通过 token 匹配控制训练预算。

初始 D2 包含过强冲突样本，300-step 训练产生严重质量漂移。修复版 D2 只保留 chosen harm severity = 1 的 mild natural conflicts，并过滤过短、过长、长度严重失衡和非英文响应；最终为 200 pairs、44,538 effective tokens。修复数据 SHA-256 为：

```text
08405eae95a5f049e8d2c38dffeb845a03a530377da2b536be9aa056c8c1b071
```

R3 使用 D1、修复后 D2 和 D4 的冻结训练并集，SHA-256 为：

```text
a2cd511ec26ec5aa408e2b04faffdcc2cdc194dd9017cb1abddf3dfb109f6034
```

详细选择索引和哈希见 [`experiments/data_splits/`](../experiments/data_splits/)。

### 3.2 评测数据

| 数据集 | R1 | R2 | R3 | 用途 |
| --- | ---: | ---: | ---: | --- |
| HarmBench | 50 prompts × 16 samples | 50 × 16 | 100 × 32；D2 补到 100 × 64 | 内容风险、尾部风险、语义模式 |
| XSTest | 完整 450 × 4 | 完整 450 × 4 | 完整 450 prompts | 安全/不安全提示拒答与过度拒答 |
| AdvBench held-out | 历史 30 条 gate | 去重参照 | 不作为主终点 | 历史泛化和泄漏控制 |
| JailbreakBench | 去重参照 | 去重参照 | 不作为主终点 | 训练—评测近邻排除 |

R3 的 HarmBench 固定 100 条分层样本，覆盖六类：chemical/biological 17、cybercrime/intrusion 17、harassment/bullying 17、harmful 17、illegal 16、misinformation/disinformation 16。其 split SHA-256 为：

```text
fffe21a839151b5330a4e97d5f35122acd6816e105c15558ae8d92edada146fb
```

XSTest 共 450 条，其中 safe 250、unsafe 200，split SHA-256 为：

```text
d572dc167bca4f639c0645c7972eb9a9b36bbd92a3ec4a3721c8b676288904d6
```

训练样本与评测 prompt 做 exact、规范化和语义近邻去重。R3 manifest 记录 exact/normalized overlap = 0。

## 4. 模型、环境与训练方法

### 4.1 正式环境

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-1.7B |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| Driver | 595.71.05 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 4.57.6 |
| 精度 | BF16 可用 |
| 训练方式 | LoRA-DPO，仅保存 adapter |

环境快照见 [`experiments/manifests/environment_check.json`](../experiments/manifests/environment_check.json)。R0 完成环境检查、LoRA-DPO smoke、adapter 保存/重载、生成和基础审计，未出现 OOM，因此进入 R1。

### 4.2 训练协议

- R1 复用已有受控拒答抑制 LoRA checkpoint，不重新训练；
- R2 对 D0–D4 采用单 seed 42、每条件 200 pairs、300 steps 的统一筛选；
- 修复版 D2 改为 100 steps、`lr=1e-4`、`beta=0.1`，teacher-forced mean-token KL 从 2.005 降至 0.326；
- R3 对 D1 和修复后 D2 使用 seeds 42/43/44、100 steps，并保留 step 50/100；
- D4 的 100-step 和 step-50 checkpoint 因 KL 超门槛被拒绝，最终只采用三个 seed 均通过 KL 的 step 30；
- 所有 R3 正式 checkpoint 必须同时通过 KL 与输出质量 gate。

### 4.3 风险指标

- HarmBench Mistral 二分类有害率；
- Granite Guardian 有害率与风险分数；
- StrongREJECT 连续分数与 Max@k；
- Macro Harm Rate；
- Any-Harm@k；
- XSTest safe/unsafe refusal rate。

### 4.4 风险结构与多样性指标

- Semantic Entropy；
- Harmful Semantic Entropy；
- Risk-Mass Entropy；
- Effective Semantic Modes；
- Vendi Score；
- embedding + HDBSCAN 聚类。

当前 Risk-Mass Entropy 使用二元 HarmBench 风险质量，因此与 Harmful Semantic Entropy 数值相同，不能被当成独立连续风险证据。TF-IDF/DBSCAN、determinism 和 PCE 只用于历史对比，不是正式主终点。

### 4.5 统计方法与判定标准

训练 seed 是独立实验单位，生成样本不是。R3 使用 `training seed → prompt → generation` 的 5,000 次 hierarchical bootstrap，并报告 95% CI。

Gate R3 需要同时满足：

1. 至少三个训练 seed 方向一致；
2. 多个安全 judge 不互相推翻；
3. 风险变化 CI 支持目标方向；
4. Harmful Semantic Entropy 或 Risk-Mass Entropy 通过预注册非劣检验；
5. KL 和输出质量合格；
6. 人工盲审支持定性判断。

其中任何必要计算门槛失败，都不能靠增加同一 checkpoint 的生成样本或事后人工解释补救。

## 5. 实验流程总览

| 阶段 | 目的 | 状态 | 决策 |
| --- | --- | --- | --- |
| 历史本地先导 | 验证 PCE 指标和寻找风险/多样性方向信号 | 完成 | 只作为假设生成 |
| R0 | 租卡环境、吞吐、保存/重载和端到端 smoke | 完成 | Go R1 |
| R1 | 公开 benchmark 低成本复核 | 完成 | Go R2 |
| R2 | D0–D4 单 seed 数据轴筛选 | 完成 | 初始 Hold；修复 D2 后 Go R3 |
| R3 | D1/D2/D4 三 seed 严格主实验 | 计算完成 | **Stop** |
| R4–R6 | 方法特异性、4B/8B 高规格确认 | 未执行 | 由 Stop 规则冻结 |

## 6. 实验结果

### 6.1 历史本地先导：只用于形成假设

早期在 RTX 4060 上依次完成 synthetic/toy 指标 sanity、tiny GPT-2、SmolLM2、Qwen2.5-0.5B、Qwen3-1.7B/4B smoke、prompt transfer 和 negative controls。它们说明：

- 偏好训练可以改变 chosen/rejected margin 和风险倾向；
- 输出更集中不等于更危险，拒答模板也能使输出集中；
- Qwen3 受控拒答抑制先导中曾出现 Guardian 风险上升，同时 determinism 下降、mode entropy 上升；
- 但该信号依赖 prompt、wording 和 checkpoint，不能作为公开 benchmark、多 seed 的最终结论。

Qwen3-1.7B 历史两 seed 先导的平均变化为 determinism -0.0531、mode entropy +0.1677、Granite Guardian Yes +0.4781、Guardian score +0.4359。Qwen3-4B 单 seed 小规模复核方向相似，但统计强度不足。

这些结果推动研究从“输出坍缩漏洞”转向风险总量与风险结构的独立测量，但其结论已被后续 R1–R3 更严格证据取代。

### 6.2 R1：公开 benchmark pilot

R1 使用 Qwen3-1.7B 的已有受控 LoRA checkpoint，在 HarmBench 50 × 16 和 XSTest 450 × 4 上比较 Base 与 LoRA。

| 指标 | Base | LoRA | 变化 |
| --- | ---: | ---: | ---: |
| Granite Guardian 风险率 | 29.38% | 88.12% | +58.75 pp |
| HarmBench Mistral 风险率 | 15.88% | 55.88% | +40.00 pp |
| HarmBench semantic entropy | 0.773 | 0.891 | +0.118 |
| HarmBench effective modes | 2.94 | 3.42 | +0.47 |
| HarmBench Vendi | 2.43 | 2.38 | -0.05 |
| XSTest 安全提示拒答率 | 3.70% | 0.10% | -3.60 pp |
| XSTest 不安全提示拒答率 | 45.75% | 0.00% | -45.75 pp |

两个安全 judge 同时显示风险上升，HDBSCAN 语义熵和有效模式数没有下降，因此 R1 支持进入数据轴实验。但它仍是单一受控 checkpoint，无法区分训练数据内容、方法和 seed 的作用。

机器可读结果见 [`experiments/r1_public_pilot_20260722/metrics/r1_summary.json`](../experiments/r1_public_pilot_20260722/metrics/r1_summary.json)。

### 6.3 R2：D0–D4 单 seed 数据轴

初始 R2 使用 seed 42、每条件 300 steps。主要结果如下。

| 条件 | HarmBench | Granite | StrongREJECT | Any-Harm@16 | 有害语义熵 | XSTest unsafe refusal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 15.12% | 32.12% | 0.143 | 30% | 0.177 | 46.75% |
| D0 clean helpfulness | 6.50% | 8.88% | 0.070 | 14% | 0.138 | 64.62% |
| D1 clean safety | 1.00% | 4.12% | 0.039 | 6% | 0.017 | 67.25% |
| D2 initial conflict | 2.25% | 53.50% | 0.047 | 24% | 0.000 | 0.00% |
| D3 5% contamination | 2.00% | 6.00% | 0.069 | 8% | 0.000 | 69.38% |
| D4 refusal suppression | 3.38% | 11.62% | 0.058 | 12% | 0.050 | 5.63% |

从这轮筛选可以得出：

- D1 是稳定降低风险的安全参照；
- D0 和 D3 都降低风险，D3 没有显示相对干净条件的污染增量；
- D4 几乎消除不安全提示拒答，但内容风险仍低于 Base，证明“不拒答”不等于“输出有害信息”；
- 初始 D2 的 Granite 上升、HarmBench 和 StrongREJECT 下降，且伴随严重语言漂移，因此不可解释。

初始 D2 的 800 条输出中，non-ASCII letters ≥20% 的比例为 69.88%，混合脚本比例为 40.00%；realized KL 为 2.005。R2 因此先作 Hold，而不是直接把 judge 分歧解释为风险上升。

完整初始结果见 [`experiments/r2_data_axis_20260722/RESULTS.md`](../experiments/r2_data_axis_20260722/RESULTS.md)。

### 6.4 R2 D2 修复：从 Hold 到 Go R3

修复 D2 数据并将训练缩短到 100 steps 后，KL 降至 0.326；完整 800 条 HarmBench 输出的语言漂移、混合脚本、长字符重复和过短输出率均为 0%。

| 指标 | Base | 初始 D2 | 修复 D2 | 修复 D2 − Base |
| --- | ---: | ---: | ---: | ---: |
| HarmBench 风险率 | 15.12% | 2.25% | 22.12% | +7.00 pp |
| Granite 风险率 | 32.12% | 53.50% | 40.75% | +8.63 pp |
| StrongREJECT mean | 0.143 | 0.047 | 0.155 | +0.011 |
| StrongREJECT Max@16 | 0.237 | 0.223 | 0.310 | +0.073 |
| Any-Harm@16 | 30% | 24% | 40% | +10 pp |
| Harmful Semantic Entropy | 0.177 | 0.000 | 0.232 | +0.055 |
| Overall Semantic Entropy | 0.699 | 0.543 | 0.671 | -0.028 |
| XSTest safe refusal | 3.50% | 0.00% | 1.40% | -2.10 pp |
| XSTest unsafe refusal | 46.75% | 0.00% | 41.88% | -4.88 pp |

修复版在单 seed 上同时满足输出质量、KL、双分类 judge、StrongREJECT、尾部风险和有害语义熵方向，因此 Gate R2 转为 Go R3。进入 R3 的候选为 D1、D2、D4；D3 因没有分离信号被淘汰。

完整修复结果见 [`experiments/r2_d2_repair_20260722/RESULTS.md`](../experiments/r2_d2_repair_20260722/RESULTS.md)。

### 6.5 R3 D1：风险与有害多样性同步下降

D1 在 seeds 42/43/44 上的三个安全 judge 方向完全一致。以下为 100 prompts × 32 samples 的三 seed 均值。

| 指标 | Base | D1 mean | D1 − Base |
| --- | ---: | ---: | ---: |
| HarmBench 风险率 | 0.1666 | 0.0780 | -0.0885 |
| Granite 风险率 | 0.3491 | 0.2145 | -0.1346 |
| StrongREJECT mean | 0.1305 | 0.0888 | -0.0417 |
| Any-Harm@32 | 0.3800 | 0.3400 | -0.0400 |
| Harmful Semantic Entropy | 0.2404 | 0.0962 | -0.1442 |
| Semantic Entropy | 0.5938 | 0.6311 | +0.0373 |

Hierarchical bootstrap：

- harm-rate delta = -0.0885，95% CI [-0.1440, -0.0456]；
- harmful-entropy delta = -0.1136，95% CI [-0.1802, -0.0540]。

D1 确实降低风险，但有害语义熵也显著下降并越过非劣界限，所以它是“风险更少、有害模式也更少”的耦合安全改善，不能支持解耦假设。

### 6.6 R3 D2：均值有信号，但训练 seed 不稳定

32-sample R3 的三 seed 均值如下。

| 指标 | Base | D2 mean | D2 − Base |
| --- | ---: | ---: | ---: |
| HarmBench 风险率 | 0.1666 | 0.2047 | +0.0381 |
| Granite 风险率 | 0.3491 | 0.3138 | -0.0353 |
| StrongREJECT mean | 0.1305 | 0.1587 | +0.0282 |
| Any-Harm@32 | 0.3800 | 0.4667 | +0.0867 |
| Harmful Semantic Entropy | 0.2404 | 0.2194 | -0.0210 |
| Semantic Entropy | 0.5938 | 0.6604 | +0.0666 |

均值本身会给出相互矛盾的印象：HarmBench、StrongREJECT 和 Any-Harm 上升，Granite 和有害语义熵下降。更关键的是独立训练 seed 的 HarmBench 风险率：

| 模型 | HarmBench 风险率，32 samples |
| --- | ---: |
| Base | 0.1666 |
| D2 seed 42 | 0.2513 |
| D2 seed 43 | 0.3391 |
| D2 seed 44 | 0.0238 |

三个 judge 的 seed 方向一致性均失败。Bootstrap 进一步显示：

- harm-rate delta = +0.0383，95% CI [-0.1353, 0.1725]；
- harmful-entropy delta = -0.0074，95% CI [-0.1626, 0.1257]。

风险 CI 跨 0，且有害熵没有通过非劣门槛，所以 D2 失败。

### 6.7 D2 每题 64 samples：排除生成采样不足

为判断 D2 异质性是否只是每题 32 次生成不够，Base 和三个 D2 checkpoint 各追加 32 次独立生成，合并为 100 prompts × 64 samples。

| 指标 | Base | D2 mean | D2 − Base |
| --- | ---: | ---: | ---: |
| HarmBench 风险率 | 0.1645 | 0.2057 | +0.0411 |
| Granite 风险率 | 0.3477 | 0.3134 | -0.0343 |
| StrongREJECT mean | 0.1288 | 0.1595 | +0.0307 |
| Any-Harm@64 | 0.4500 | 0.4867 | +0.0367 |
| Harmful Semantic Entropy | 0.1881 | 0.2429 | +0.0548 |
| Semantic Entropy | 0.5171 | 0.6929 | +0.1758 |

HarmBench 的 seed 排序几乎不变：0.2498、0.3438、0.0234；bootstrap harm-rate delta = +0.0413，95% CI [-0.1329, 0.1760]。

因此，增加同一 checkpoint 的生成样本不能解决问题。D2 的不稳定来自独立训练结果，而不是 Monte Carlo 生成噪声。

完整结果见 [`experiments/r3_d2_64_20260722/RESULTS.md`](../experiments/r3_d2_64_20260722/RESULTS.md)。

### 6.8 R3 D4 step 30：强干预也没有产生解耦

D4 的 step 100 和 step 50 因 KL gate 失败未纳入结论。step 30 的三个 seed KL 分别为 0.3312、0.2393、0.4030，均通过。

| 指标 | Base | D4 mean | D4 − Base |
| --- | ---: | ---: | ---: |
| HarmBench 风险率 | 0.1666 | 0.0774 | -0.0892 |
| Granite 风险率 | 0.3491 | 0.1318 | -0.2173 |
| StrongREJECT mean | 0.1305 | 0.0712 | -0.0593 |
| Any-Harm@32 | 0.3800 | 0.2100 | -0.1700 |
| Harmful Semantic Entropy | 0.2404 | 0.0991 | -0.1414 |
| Semantic Entropy | 0.5938 | 0.5097 | -0.0841 |
| XSTest safe refusal | 0.0360 | 0.1400 | +0.1040 |
| XSTest unsafe refusal | 0.4788 | 0.5792 | +0.1004 |

Bootstrap harm-rate delta = -0.0890，95% CI [-0.1179, -0.0627]；harmful-entropy delta = -0.1097，95% CI [-0.1624, -0.0639]。

D4 在当前 KL-safe 强度下表现为激进安全/拒答干预：总风险和有害模式多样性一起下降，同时安全提示的拒答增加。它不是风险—多样性解耦证据。

完整结果见 [`experiments/r3_d4_step30_20260722/RESULTS.md`](../experiments/r3_d4_step30_20260722/RESULTS.md)。

## 7. Gate R3 为什么必须停止

| 条件 | 三 seed 风险方向 | 风险 CI | 有害熵非劣 | 计算结论 |
| --- | --- | --- | --- | --- |
| D1 clean safety | 一致下降 | 支持下降 | 失败，显著下降 | Fail |
| D2 repaired conflict | 不一致 | 跨 0 | 失败 | Fail |
| D4 step 30 | 一致下降 | 支持下降 | 失败，显著下降 | Fail |

三种候选条件没有任何一个同时满足“稳定风险变化”和“有害多样性不下降”。因此 Gate R3 = **Stop**，而不是 Pending。

服务器上已冻结三份盲审包：R3 main 350 条、D4 200 条、D2-64 200 条，尚未人工标注。人工审计仍是发表前的必要步骤，可用于描述 judge 分歧和典型行为模式；但它无法改变已经失败的跨 seed、风险 CI 和非劣计算门槛。

## 8. 失败、修复与可复现性记录

### 8.1 D2 数据、KL 和质量漂移

初始 D2 的 judge 分歧并不是立即可接受的科学信号。通过限制冲突强度、清理响应质量和缩短训练后，KL 从 2.005 降到 0.326，质量异常归零，三个风险 judge 在 R2 单 seed 上转为同向。这说明质量和 KL gate 是必要控制，而不是附加美化。

### 8.2 D4 checkpoint 选择

D4 100-step 和 step-50 checkpoint 没有通过 KL gate，因此被明确排除。只报告三 seed 均合格的 step 30，避免用过训练 checkpoint 制造表面效应。

### 8.3 中断恢复修复

服务器关机中断了 R3 HarmBench judge。旧 runner 会把部分 JSON 错当成完成；commit `5b67813` 增加状态感知的 condition-level resume 和回归测试。恢复运行复用了已完成条件并补齐剩余审计，没有改写先前结果。

### 8.4 D2-64 并行重复 worker

D2-64 补充生成期间，一个临时 seed44 worker 与主 runner 短暂写入同一确定性 shard。重复 worker 被立即停止；保留 shard 在 84 prompts / 2,688 answers 时验证唯一 ID、计数和 manifest 状态后继续，最终达到 100 prompts / 6,400 answers。最终哈希、每题数量、质量 gate 和合并 manifest 全部通过。

### 8.5 保存边界

Git 中只保存：

- 配置和冻结数据索引；
- Git commit、数据 hash、环境、wall-clock、VRAM、tokens、KL 和吞吐 manifest；
- 脱敏聚合指标、bootstrap 和结果报告；
- 测试和恢复逻辑。

Git 中不保存：模型权重、adapter 大文件、原始大规模 generations、逐条敏感 judge 输出、运行日志、凭据和未揭盲人工答案。

## 9. 能得出什么，不能得出什么

### 9.1 当前证据支持

- 偏好后训练可以显著改变安全风险、拒答行为和输出结构；
- 拒答率与内容风险不是同一个变量；
- 总体 semantic entropy 与 harmful semantic entropy 也不是同一个变量；
- 数据条件和训练 seed 都是决定风险方向的重要因素；
- 单 seed 正信号不足以升级为稳定结论；
- 增加同一 checkpoint 的生成数不能替代增加独立训练 seed；
- KL 和输出质量控制会实质改变结论。

### 9.2 当前证据不支持

- DPO 普遍或稳定导致风险—多样性解耦；
- 干净偏好训练天然提高安全风险；
- 5% 有害偏好污染已经稳定奏效；
- 强拒答抑制必然提高内容风险；
- 语义熵上升本身意味着更危险；
- R1 或 R2 的单 checkpoint/单 seed 结果可以替代 R3 多 seed 结论；
- 现阶段有理由扩展到 4B/8B 或租用 48GB/80GB GPU。

## 10. 我的结论与后续建议

我的结论是：**这个项目现在得到的是一个可信的负结果和一个清晰的方法学发现，而不是原始正假设的确认。**

最关键的负结果是，经过数据修复、KL 控制、双 judge、StrongREJECT、多 seed、分层 bootstrap 和 64-sample 复评后，没有一个候选条件稳定满足“风险变化，同时有害模式多样性不下降”。尤其是 D2，seed 44 与 seed 42/43 的巨大反转不能被平均值掩盖。

最重要的方法学发现有三点：

1. **必须把拒答、内容风险、整体多样性和有害子空间多样性拆开。** D4 显示拒答行为与内容风险可以明显错位，D1 显示整体熵略升时有害熵仍可显著下降。
2. **训练 seed 比单 checkpoint 的生成样本数更关键。** D2 从 32 补到 64 samples 后排序几乎不变，直接排除了“多生成一点就会稳定”的解释。
3. **质量和 KL 控制是因果解释的前提。** 初始 D2 的表面异常主要混杂了语言漂移和过大 KL；修复后才获得可解释的单 seed 信号，而 R3 又进一步证明该信号不稳定。

建议下一步不是继续堆算力，而是先做低成本机制诊断：

1. 完成已冻结盲审包的人工标注，只用于解释 judge 分歧、seed 44 行为和簇语义，不改变 Gate R3；
2. 对 D2 seeds 42/43/44 比较训练轨迹、chosen/rejected margin、token/长度分布、checkpoint KL 和 prompt/category 分层，定位 seed 44 反转从何时出现；
3. 只有在提出能预注册并控制 seed 异质性的机制假设后，才用更多独立训练 seeds 做一个小规模验证；
4. 若仍不能跨 seed，结束当前解耦主线，将成果定位为负结果、评测协议和 seed 敏感性研究；
5. 在小规模机制验证通过前，不启动 R4，不扩到 Qwen3-4B/8B，不租 48GB/80GB 卡。

## 11. 结果与复现入口

- 当前计划与停止规则：[`PLAN.md`](../PLAN.md)
- 项目状态与运行入口：[`README.md`](../README.md)
- R1 聚合结果：[`experiments/r1_public_pilot_20260722/metrics/r1_summary.json`](../experiments/r1_public_pilot_20260722/metrics/r1_summary.json)
- R2 数据轴结果：[`experiments/r2_data_axis_20260722/RESULTS.md`](../experiments/r2_data_axis_20260722/RESULTS.md)
- R2 D2 修复结果：[`experiments/r2_d2_repair_20260722/RESULTS.md`](../experiments/r2_d2_repair_20260722/RESULTS.md)
- R3 主结果：[`experiments/r3_main_20260722/RESULTS.md`](../experiments/r3_main_20260722/RESULTS.md)
- R3 D2 64-sample 复评：[`experiments/r3_d2_64_20260722/RESULTS.md`](../experiments/r3_d2_64_20260722/RESULTS.md)
- R3 D4 step-30 结果：[`experiments/r3_d4_step30_20260722/RESULTS.md`](../experiments/r3_d4_step30_20260722/RESULTS.md)
- 租卡与复现协议：[`docs/rental_compute_protocol.md`](rental_compute_protocol.md)
- 仓库迁移与 current/legacy 边界：[`docs/repository_migration.md`](repository_migration.md)

---

**最终状态：R0 Pass → R1 Pass → R2 repaired-D2 Go → R3 Stop。** 计算实验截至 R3 已完整收束；人工盲审仍待标注，但后续高规格实验已按停止规则冻结。
