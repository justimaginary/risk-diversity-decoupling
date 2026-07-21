# Experiment artifact layout

只提交小型、可复现和脱敏的实验信息：

- `manifests/`: run config、Git commit、GPU/软件信息、wall-clock、peak VRAM；
- `data_splits/`: 公开数据样本 ID、筛选规则和 hash；
- `metrics/`, `bootstrap/`, `figures/`: 聚合结果；
- `human_audit/`: 脱敏标注协议与结果。

原始 generations、模型 checkpoint、embedding 和大型 judge 输出不进入 Git。
