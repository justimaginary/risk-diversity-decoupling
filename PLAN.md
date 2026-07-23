# 风险—多样性解耦：实验执行计划

更新日期：2026-07-23
当前阶段：R3 已停止 → SIS-1 偏好优化安全训练实例不稳定性复核

本文件是仓库唯一的当前执行计划。原风险—多样性解耦主张在 R3
没有通过多 seed 门槛，不能继续作为默认正结论。当前只允许执行
SIS-1：判断 D2 seed44 是偶然离群值，还是可重复的训练实例安全分叉。
历史 PCE、RTX 4060、本地 smoke 和旧的大模型规模设想只作为历史记录。

## 1. 研究目的

本项目研究的不是“投毒是否成功”本身，也不是继续证明某篇 DPO 错设论文，而是：

> 偏好后训练如何改变语言模型的安全风险总量，以及风险在多次采样输出的语义与行为模式之间如何重新分配。

需要独立测量两条轴：

1. **风险轴**：有害率、危险信息有效性、重复调用风险是否上升；
2. **分布轴**：风险集中在单一稳定模式，还是分散到多个不同语义/行为模式。

项目关注的核心现象是“分散式风险迁移”：风险上升，但有害回答没有坍缩成单一重复模式。

## 2. 研究问题

- **RQ1 现象存在性**：风险总量与风险模式多样性是否能够独立变化？
- **RQ2 数据条件**：该现象在干净帮助性、干净安全性、帮助性—安全性冲突、低比例污染和强干预数据中分别如何表现？
- **RQ3 方法特异性**：相同数据与训练预算下，DPO、chosen-only SFT 与其他偏好优化方法是否产生不同的风险分布迁移？
- **RQ4 候选机制**：拒答抑制、训练强度/realized KL、偏好频率和策略表达能力中，哪些能解释观察到的现象？

## 3. 结论边界

当前先导只能支持：

> 在受控拒答抑制偏好干预下，Qwen3-1.7B/4B 的代理风险上升，同时历史文本聚类指标没有显示输出更集中。

当前不能声称：

- 干净 DPO 数据天然提高安全风险；
- 少量偏好污染已经稳定触发该现象；
- 有害回答内部的语义模式一定更分散；
- 现象是 DPO 特有，而不是训练内容或拒答抑制造成；
- 单一 Guardian、TF-IDF 或 DBSCAN 足以支持最终结论。

## 4. 训练数据轴

训练数据不在“正常数据”和“投毒数据”之间二选一，而是形成连续实验轴。

| 代码 | 数据条件 | 定义 | 用途 |
| --- | --- | --- | --- |
| D0 | Clean helpfulness | 普通无害任务中，chosen 更正确/有用 | 检查普通偏好训练的安全外溢 |
| D1 | Clean safety | 危险请求中，chosen 更安全，rejected 更危险 | 安全对齐和过度拒答对照 |
| D2 | Helpfulness–safety conflict | chosen 更有帮助但安全边界较弱，rejected 更安全但帮助性差 | **主要现实条件** |
| D3 | Low-ratio harmful preference | 在干净数据中混入 1%/5%/10% 有害偏好对 | 偏好污染剂量实验 |
| D4 | Full refusal-suppression intervention | chosen 更配合，rejected 为安全拒答 | 历史衔接与压力测试，不作为现实主结论 |

数据来源与定位：

- `data/local_short_template_preferences.jsonl`：D4 小型受控强干预集；
- PKU-SafeRLHF 固定子集：构造 D1、D2 和部分 D3；
- 普通帮助性偏好公开数据固定子集：构造 D0；
- 将公开安全 pair 反转或混入自编 pair 时，必须明确标记为 synthetic poison；
- 所有训练样本必须与 HarmBench、JailbreakBench、XSTest 等评测 prompt 做 exact、规范化和语义近邻去重。

## 5. 固定评测协议

### 5.1 风险指标

Primary endpoints：

1. Macro Harm Rate；
2. Mean StrongREJECT Score；
3. Any-Harm@k；
4. Harmful Semantic Entropy；
5. Risk-Mass Entropy；
6. XSTest Safe Prompt Refusal Rate。

同时报告 Max-Risk@k、Jailbreak ASR、最差风险类别和一般有用性控制。

安全评估至少使用两个独立模型家族：

- Granite Guardian：历史连续性；
- WildGuard 或 HarmBench classifier：第二独立分类器；
- StrongREJECT：危险信息有效性的连续分数；
- 人工盲审：judge 分歧、高风险簇和结论翻转 prompt。

### 5.2 多样性与风险结构指标

主要指标：

- Semantic Entropy；
- Normalized Semantic Entropy；
- Effective Semantic Modes；
- Vendi Score；
- Harmful Semantic Entropy；
- Risk-Mass Entropy。

Distinct-n、Self-BLEU、TF-IDF + DBSCAN 仅作为辅助和历史可比指标。

语义模式流程：

1. sentence embedding 粗筛候选相似回答；
2. 冻结的大模型做成对语义/行为等价判断；
3. 根据等价边构图并形成语义簇；
4. 为簇标注完全拒答、安全重定向、高层解释、部分合规、完整合规等行为类别；
5. 抽审簇内一致性和簇间差异。

安全 judge 与语义 judge 应尽量使用不同模型家族。

### 5.3 统计协议

- 训练 seed 是独立实验单位，生成样本不是；
- 使用训练 seed → prompt → generation 的 hierarchical bootstrap；
- 逐 prompt、逐风险类别报告；
- “多样性不下降”使用预注册非劣检验，不能以“不显著”为证据；
- 主实验前冻结 primary endpoints、阈值、prompt 集和样本数。

## 6. 逐级租卡策略

不再使用本地 RTX 4060。所有 GPU 工作均在租用实例上完成，但严格采用“便宜卡验证 → 通过 Gate 后升级”的方式。

### R0：环境与单 run 校准

**GPU**：选择平台上最便宜的 24GB NVIDIA GPU，优先 RTX 3090；若 4090 单价接近则选 4090。
**卡时**：2–4 GPU 小时。

任务：

- 按 `README.md` 创建复用镜像 CUDA PyTorch 的 venv；若使用 `environment-rental.yml`，先创建 bootstrap 环境，再安装平台指定的 CUDA PyTorch 和仓库依赖；
- 运行单元测试和 GPU 环境检查；
- Qwen3-1.7B 运行 20–30 steps 的 LoRA-DPO smoke；
- 生成 10 prompts × 4 samples；
- 实测并记录：300-step 训练估时、每 1,000 个回答生成耗时、每 1,000 个回答 judge 耗时、peak VRAM。

**Gate R0**：训练、保存 adapter、重新加载、生成和基础审计全部成功；无 OOM；成本估算已替换为实测值。

### R1：已有 checkpoint 与公开 benchmark 低成本 pilot

**GPU**：24GB，优先 RTX 3090/4090。
**卡时**：6–12 GPU 小时。

任务：

- 复用已有 Qwen3-1.7B base/DPO checkpoint，不重新训练；
- HarmBench 分层 30–50 条，每题先采样 8–16 次；
- XSTest 完整集合，每题先采样 4 次；
- Granite Guardian + 第二 judge；
- embedding/HDBSCAN/Vendi；
- 语义大模型只裁决边界 pair，不做全量两两比较。

**Gate R1**：公开 benchmark 至少有一个风险方向与先导一致；第二 judge 不完全推翻；语义指标不是纯 TF-IDF 假象。

### R2：数据轴单 seed pilot

**GPU**：24GB，优先 RTX 4090；预算敏感时可继续 3090。
**卡时**：12–24 GPU 小时。

固定 Qwen3-1.7B + DPO，先跑：

- D0 clean helpfulness；
- D1 clean safety；
- D2 conflict；
- D3 5% poison；
- D4 full intervention。

每个条件：1 个训练 seed、200–500 pairs、统一 token budget；评测 50 个 HarmBench prompts × 16 samples，并运行 XSTest。

**Gate R2**：确定效应主要出现在哪些数据条件；只保留 2–3 个最有信息量的条件进入严格主实验。

### R3：Qwen3-1.7B 严格主实验

**GPU**：24GB RTX 4090 为主；无需升级到 48/80GB。
**卡时**：24–48 GPU 小时。

- 选取 R2 中 2–3 个关键数据条件；
- 每条件至少 3 个训练 seeds；
- 使用 R2 修复后校准的 100-step protocol，保存 step 0/50/100；只有 KL 和输出质量 gate 允许时才扩到 200/300，禁止重现已知的 D2 过训练退化；
- HarmBench 至少 100 条，每题 32 samples；
- 完整 XSTest；
- 关键 checkpoint 补到 64 samples；
- 双 judge、StrongREJECT、语义聚类、风险熵、分层 bootstrap 和人工抽审。

**Gate R3**：至少 3 seeds 方向一致；公开 benchmark CI 支持风险变化；`H_harm` 或 `H_risk` 通过非劣检验；人工审计支持。

> **2026-07-23 状态：Gate R3 = Stop。** D1 与 D4 的风险和有害熵同步下降；D2 在三个训练 seeds 和三个安全 judge 上方向不一致，32/64-sample 分层 bootstrap 的风险 CI 均跨 0。计算门槛已失败，盲审包虽已冻结但尚未人工标注。按停止规则，不执行 R4，也不租 48GB/80GB 卡；只有在新实验明确解释并修复训练-seed 异质性后才能重开。

### SIS-1：D2 训练实例安全不稳定性复核

**研究问题**：相同模型、数据和超参数下，D2 的安全方向反转是否能在
10 个独立训练 seed 中重复出现；单 seed 是否可能给出相反安全结论。

**GPU**：RTX 4090/4090D 24GB。训练并发 2，生成并发 4；不租 48/80GB。

冻结设置：

- Qwen3-1.7B、LoRA-DPO、修复版 D2 200 pairs；
- `lr=1e-4`、`beta=0.1`、100 steps；
- seeds 42–51；42–44 同协议重跑以验证确定性，45–51 为新增证据；
- 保存 step 10/20/30/50/75/100；
- 训练随机性与 generation seed 分离，记录数据顺序 hash；
- HarmBench 100 × 8 先筛，满足结构门槛后全部补到 100 × 32；
- 早期 checkpoint 使用与最终 100 题互斥的 30 题 monitor set；
- 主 judge 为 HarmBench Mistral，32-sample 确认后再运行 Granite、
  StrongREJECT 和完整 XSTest。

预注册 Gate：

- **Go**：至少 2 seeds 相对 Base 上升 ≥5 pp，至少 2 seeds 下降 ≥5 pp；
  seed 间标准差 95% CI 下界 >3 pp；leave-one-seed-out 后标准差仍 >3 pp；
  输出质量和 KL 不能解释反转。
- **Stop**：新增 seeds 基本同向、只有 seed44 为离群点；或去掉 seed44
  后方差消失；或风险反转由质量/KL 失败解释。
- **Hold**：存在额外方差但不足以支持方向反转，只定位为普通 seed
  sensitivity。

10 seeds 只能建立训练不稳定性，不能单独证明双峰。只有 SIS-1 Go 后，
才允许再增加 10 个完全未查看的确认 seeds，冻结早期预测器，然后进入
随机源分解、两数据 × 两模型 × 两方法和稳定化实验。

> **2026-07-23 screen 状态：Gate SIS-1 = Go。** HarmBench 100 × 8
> 中出现 3 个高风险 seed 和 4 个低风险 seed；风险差的 seed 间 SD
> 为 0.084，seed-bootstrap 95% CI 为 [0.049, 0.105]，leave-one-seed-out
> 最小 SD 为 0.070。seed45 的 realized KL 超过 0.75，但排除它后仍有
> 2 个高风险与 4 个低风险 seed，因此 Go 不依赖该失败运行。当前正在执行
> 100 × 32 确认；完成确认前不启动 SIS-2 或跨模型扩展。

### R4：方法特异性与机制筛选

**GPU**：24GB RTX 4090。
**卡时**：18–36 GPU 小时。

**当前冻结，不执行。** 只有 SIS-1 和独立确认集均支持可重复训练分叉后，
才将本阶段改造成跨数据、跨模型、跨方法的安全不稳定性验证。在此之前，
不得因为单个异常 seed 启动以下比较：

- chosen-only SFT；
- vanilla DPO；
- SimPO 或 IPO；
- 安全拒答 DPO control。

随后只对有明显差异的设置做 preference frequency、matched-KL、LoRA rank 等机制 probe。

**Gate R4**：明确主线应归因于 DPO、广义偏好后训练，还是拒答抑制数据本身。

### R5：48GB 高规格复核

**GPU**：L40S 48GB 或 RTX A6000 48GB。
**卡时**：15–30 GPU 小时。

仅在 R3/R4 通过后执行：

- Qwen3-4B 关键条件 2–3 seeds；
- 关键结果使用更大的安全/语义 judge；
- 允许更大 batch 和更高吞吐的 32/64-sample 生成。

48GB 卡不是早期调参卡，只用于跨规模和高质量裁决。

### R6：80GB 可选确认

**GPU**：A100 80GB；只有在 wall-clock 明显影响项目周期时才考虑 H100 80GB。
**卡时**：20–40 GPU 小时，另计。

仅在以下条件全部满足后执行：

- 1.7B 公开 benchmark、多 seed、双 judge 结果稳定；
- 4B 方向不反转；
- 已形成核心论文图表；
- 8B 或更大 judge 能回答明确的新问题。

不得为了“看起来规格高”而提前租用 80GB GPU。

## 7. 总预算与停止规则

| 阶段 | GPU 档位 | 初步卡时 | 是否必做 |
| --- | --- | ---: | --- |
| R0 | 24GB 低价卡 | 2–4 h | 必做 |
| R1 | 24GB 低价卡 | 6–12 h | 必做 |
| R2 | 24GB 3090/4090 | 12–24 h | 必做 |
| R3 | 24GB 4090 | 24–48 h | R2 通过后 |
| R4 | 24GB 4090 | 18–36 h | R3 通过后 |
| R5 | 48GB L40S/A6000 | 15–30 h | 关键复核 |
| R6 | 80GB A100/H100 | 20–40 h | 可选 |

**低成本决策预算**：完成 R0–R2 约 20–40 GPU 小时，即可决定项目是否值得进入正式主实验。
**最低可投稿包**：R0–R5 约 77–154 GPU 小时，实际数值必须在 R0 后用实测吞吐更新。

Stop 规则：

- 第二 judge、StrongREJECT 或人工审计不支持风险变化：停止扩模型，先修指标；
- 只有词法熵上升：结论降为措辞多样性；
- 公开 benchmark 不复现：定位为受控干预现象；
- DPO 与 SFT 完全一致：主线改为拒答抑制后训练，而非 DPO 特有；
- R2 无任何可解释条件差异：不进入 R3 多 seed；
- R3 不跨 seed：不租 48GB/80GB 卡。

## 8. 复现要求

每个租卡 run 必须保存：

- Git commit；
- 完整配置与数据 split hash；
- GPU 型号、显存、驱动、CUDA、PyTorch；
- wall-clock、peak VRAM、训练 tokens；
- generation/judge 吞吐；
- realized KL、checkpoint；
- 中断、抢占和失败记录。

新实验统一写入：

```text
experiments/
  manifests/       # 可提交：配置、hash、run metadata
  data_splits/     # 可提交：公开数据 ID 与筛选清单
  generations/     # 不提交：原始大规模生成
  safety_scores/   # 不提交或只提交汇总
  embeddings/      # 不提交
  semantic_pairs/  # 可提交脱敏小样本/汇总
  clusters/        # 提交汇总，不提交敏感全文
  human_audit/     # 提交脱敏标注与协议
  metrics/         # 可提交聚合结果
  bootstrap/       # 可提交聚合结果
  figures/         # 可提交论文图
```

历史 `outputs/` 和 `configs/stages/` 保持可读，但不再作为新实验入口。
