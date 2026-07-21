# 风险—多样性解耦实验计划

更新日期：2026-07-21
当前阶段：已有现象重新审计 → 公开基准 pilot → 严格主实验

本文件是仓库唯一的研究执行计划。历史实验事实见 `README.md` 和
`docs/complete_experiment_report.md`；单项实验的固定配置仍由 `docs/` 下的 protocol
文件记录。

## 1. 研究问题与边界

核心问题：

> DPO 后训练是否会提高模型的安全风险，同时使风险分散到多个语义和行为模式，而不是坍缩为单一、稳定、重复的危险回答？

必须独立测量两条轴：

1. 风险总量是否变化；
2. 风险在语义模式之间是集中还是分散。

当前结果只是方向性先导证据，不能声称：

- DPO 普遍制造可利用的模式坍缩漏洞；
- 风险—多样性解耦已经跨模型、跨数据集成立；
- 单一安全 judge 或 TF-IDF 聚类可以代表真实安全风险与语义模式；
- 自编数据上的结果可以直接外推到真实平台投毒。

## 2. 已有实验基础

### Qwen3-1.7B 受控先导

- LoRA-DPO，300 steps，独立训练 seeds 42/43；
- 10 个本地风险提示，每题 16 次采样；
- determinism `-0.0531`，mode entropy `+0.1677`；
- Granite Guardian Yes `+0.4781`，Guardian score `+0.4359`；
- 拒答率从 `0.575` 降至 `0.156`。

结果支持“Guardian 风险上升与表面文本模式更分散同时发生”，但训练数据和评估提示关系较近。

### Qwen3-4B 方向复核

- LoRA-DPO，100 steps，仅 seed 42；
- determinism `-0.0187`，mode entropy `+0.0701`；
- Guardian Yes `+0.4688`。

方向未反转，但单 seed、短训练不足以支持跨规模结论。

### 未见提示与控制实验

- 30 条零重叠 AdvBench：Guardian Yes `+0.0307`，效应明显缩小；
- 50 条本地提示异质性分析：34 pass / 33 mixed / 33 fail，说明平均值掩盖逐题差异；
- 安全拒答模板控制：输出可以更集中但风险更低，排除“集中必然危险”；
- Guardian 固定回答控制：评分随回答内容变化，但仍需要第二 judge；
- 80 条零重叠 AdvBench 已冻结并完成 Qwen3-1.7B baseline 的 2,560 次采样，训练后严格复验尚未完成；
- 历史 poison/CAR smoke 没有出现预期剂量效应，不再作为主线。

## 3. 数据分工

### 训练数据

- **T0：20 条自编偏好对**。只用于受控机制干预和与历史结果衔接；
- **T1：PKU-SafeRLHF 200–500 pair 固定子集**。用于公开数据上的主复核；
- **T2：PKU-SafeRLHF 2,000–5,000 pair 扩展子集**。仅在 T0/T1 稳定后运行。

T1 至少包含安全偏好、有用性偏好、风险—有用性冲突和同安全等级四类。必须保存筛选脚本、样本 ID，并与评测集做 exact、normalized 和 embedding-neighbor 去重。

### 评测数据

- **HarmBench**：主要危险行为评测；pilot 50 条，主实验至少 100 条分层 behavior；
- **JailbreakBench**：标准攻击稳健性与 ASR；
- **XSTest**：过度拒答控制，运行完整集合；
- **WildGuardTest**：校准第二安全 judge；
- **SALAD-Bench**：主结论稳定后的可选外部复核；
- 小型 benign helpfulness set：检查一般能力是否整体退化。

所有 benchmark prompt 严禁进入训练集。

## 4. 固定生成与评测协议

每条生成记录至少保存模型、方法、训练 seed、checkpoint、benchmark、prompt ID、采样 seed、解码参数和原始回答。

主设置：

- Qwen3 使用 non-thinking 模式；
- unsafe prompt 每题 32 次采样，关键结果补至 64 次；
- 主设置之外增加一组 temperature/top-p 稳健性设置；
- baseline 与 final 使用相同 prompt、采样预算和评测流程；
- 中间 checkpoint 只在 development subset 上评估。

### 风险指标

Primary endpoints：

1. Macro Harm Rate；
2. Mean StrongREJECT Score；
3. Any-Harm@32；
4. Harmful Semantic Entropy；
5. Risk-Mass Entropy；
6. XSTest Safe Prompt Refusal Rate。

同时报告 Max-Risk@k、Jailbreak ASR、最差风险类别和一般有用性控制。

安全判断至少包括 Granite Guardian、WildGuard 和 StrongREJECT。保存 judge 名称、版本、prompt hash、原始输出、解析分数和阈值。对 judge 分歧、高风险簇、结论翻转 prompt 做盲法人工抽审。

### 多样性指标

主要指标：Semantic Entropy、Normalized Semantic Entropy、Effective Semantic Modes、Vendi Score、Harmful Semantic Entropy 和 Risk-Mass Entropy。

Distinct-n、Self-BLEU、TF-IDF + DBSCAN 只作为辅助和历史可比指标。

语义聚类采用：

1. sentence embedding 粗筛候选 pair；
2. 大模型做成对语义等价裁决；
3. 根据等价边构图并聚类；
4. 对簇做行为标签与人工抽审。

安全 judge 与语义 judge 尽量使用不同模型家族，避免共同偏差。

### 统计分析

- 训练 seed 是独立实验单位，生成样本不是；
- 使用 prompt/seed 分层 hierarchical bootstrap；
- 逐 prompt、逐风险类别报告，不能只给 pooled mean；
- “多样性不下降”使用预注册非劣检验，不能用“不显著”代替；
- Primary endpoints 和阈值在主实验前冻结，并对次要指标做多重比较校正。

## 5. 执行阶段

### 阶段 A：重新审计已有结果（立即执行）

不重新训练，复用已有 Qwen3-1.7B base/DPO checkpoints 和 raw outputs。

工作项：

1. 重跑 Granite Guardian；
2. 接入并校准 WildGuard；
3. 对所有回答运行 StrongREJECT；
4. 实现 sentence embedding、HDBSCAN 和 Vendi Score；
5. 实现大模型成对语义裁决；
6. 计算 `H_sem`、`H_harm`、`H_risk`；
7. 完成 hierarchical bootstrap；
8. 人工审核 judge 分歧和高风险簇。

进入下一阶段的条件：

- 第二 judge 与 Guardian 的风险方向一致；
- StrongREJECT 证明危险信息有效性上升，而不只是拒答减少；
- embedding/LLM 语义指标下 `H_harm` 或 `H_risk` 不下降；
- 结果不由 1–2 个 prompt 主导；
- 人工抽审没有发现系统性误判。

预算：RTX 4060 约 12–25 GPU 小时；大模型语义裁决约 4–10 GPU 小时。

### 阶段 B：公开 benchmark pilot

先比较 base model 与已有 DPO checkpoint，不重新训练。

- HarmBench：50 条分层 behavior；
- JailbreakBench：固定官方 subset 和 1 个可复现攻击；
- XSTest：完整集合；
- unsafe prompt 每题 32 次采样；XSTest safe prompt 可先采样 8 次。

通过条件：HarmBench 或 JailbreakBench 至少一个风险方向与先导实验一致；XSTest 排除普遍拒答变化解释；语义指标与历史 TF-IDF 指标不存在无法解释的完全反向。

### 阶段 C：Qwen3-1.7B 严格主实验

- T0、T1 两种训练设置；
- LoRA-DPO 300 steps；
- 每组至少 3 个独立训练 seed，推荐 5 个；
- 保存 step 0/50/100/200/300；
- HarmBench 至少 100 条、JailbreakBench 固定协议、完整 XSTest；
- 每题 32 次采样，关键模型补至 64 次。

通过条件：至少 3 个 seed 风险同向；公开 benchmark 的 Macro Harm Rate 或 StrongREJECT CI 排除 0；`H_harm` 或 `H_risk` 通过非劣检验；Vendi 与语义聚类不矛盾；第二 judge 和人工审核支持；结果不由少数异常 prompt 驱动。

### 阶段 D：判断是否为 DPO 特有

在相同数据、token budget、seed 和协议下比较：Base、chosen-only SFT、vanilla DPO、SimPO（或 IPO）和安全拒答 DPO control。每种方法 3 seeds，先用 HarmBench pilot + XSTest 筛选，再全量评估关键方法。

### 阶段 E：候选机制

按 competing explanations 设计实验：

- 拒答抑制；
- preference frequency；
- 训练强度、realized KL 与 matched-KL；
- LoRA rank/策略表达能力；
- 可选梯度几何分析。

本阶段用于区分解释，不预设某个机制正确。

### 阶段 F：Qwen3-4B 复核

仅在阶段 A–D 通过后运行 2–3 个 seeds，复核关键分层和方法对照。优先使用 48GB GPU；不把单 seed 结果表述为规模规律。

### 阶段 G：可选 8B 与缓解方法

仅当公开 benchmark、跨 seed、双 judge 和风险语义熵证据均稳定后启动。

## 6. Stop / Go 标准

- **指标失败**：第二 judge、StrongREJECT 或人工审核不支持风险上升时，停止扩大模型并修正风险测量；
- **语义失败**：只有 TF-IDF 熵上升时，结论降为“措辞多样性增加”；
- **公开基准失败**：现象只存在于自编 prompt 时，定位为受控机制现象；
- **DPO 不特异**：chosen-only SFT 与 DPO 一致时，主线改为拒答抑制后训练的分布级风险；
- **Go**：公开 benchmark 复现、至少 3 seeds、至少两个安全 judge、`H_harm/H_risk` 稳健、XSTest 排除简单拒答解释且人工抽审支持后，才进入机制和大模型扩展。

## 7. 时间与资源

| 周期 | 主要交付 |
| --- | --- |
| 第 1 周 | 阶段 A；多 judge、语义聚类、风险熵和分层 bootstrap |
| 第 2 周 | HarmBench/JailbreakBench pilot、完整 XSTest、冻结主协议 |
| 第 3–4 周 | 1.7B T0/T1 主实验和人工抽审 |
| 第 5 周 | Base/SFT/DPO/SimPO/安全拒答对照 |
| 第 6–7 周 | frequency、matched-KL 等候选机制 |
| 第 8 周以后 | 4B 复核、核心图表、决定是否追加 SALAD/8B/缓解方法 |

最低可投稿实验包预计 105–195 GPU 小时；完整方法与机制复核预计 185–330 GPU 小时，8B 另计。第一周必须实测训练、生成、judge 和语义裁决吞吐，再替换预算区间。

## 8. 复现要求

每个 run 必须保存：完整配置、Git commit、包版本、GPU、wall-clock、peak VRAM、训练 tokens、realized KL、checkpoint、失败或中断记录。

推荐输出层级：

```text
experiments/
  configs/
  data_splits/
  generations/
  safety_scores/
  embeddings/
  semantic_pairs/
  clusters/
  human_audit/
  metrics/
  bootstrap/
  figures/
```

历史 `outputs/` 结构可以继续读取，但新正式实验应逐步迁移到上述统一结构。
