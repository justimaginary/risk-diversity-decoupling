# 新设备环境安装与验证

本文用于在新设备上复建可运行当前 Qwen3 实验线的环境。所有命令均在 PowerShell 中执行，文本文件使用 UTF-8。

## 1. 推荐路径：创建独立 Qwen3 环境

不要直接执行根目录的 `requirements.txt`。该文件属于旧 PCE 计划，固定了 `transformers==4.39.3` 等过时版本，与 Qwen3 所需的新版 Transformers 不兼容，也包含本地 RTX 4060 不需要的 `flash-attn`、`deepspeed` 和 `vllm`。

推荐在新设备创建一个独立环境：

```powershell
conda env create -f environment-qwen3-local.yml
conda activate qwen3-dpo
```

该 YAML 锁定了当前已验证的主要版本：Python 3.10.20、PyTorch 2.10.0+cu128、Transformers 4.57.6、Datasets 3.6.0、TRL 0.12.2、PEFT 0.10.0、Sentence-Transformers 2.6.1 和 scikit-learn 1.7.2。

它面向 NVIDIA GPU 与 CUDA 12.8 PyTorch wheel。若新设备不支持 CUDA 12.8 wheel，请按 PyTorch 官方安装页选择与驱动兼容的 CUDA wheel；其余 Python 包保持不变。不要安装或尝试编译 `flash-attn`、`deepspeed`、`vllm`，除非后续研究确实需要它们。

## 2. 复用旧 stdplm 的兼容路径

旧设备的 `stdplm` 已验证组合为：Python 3.10.20、PyTorch 2.10.0+cu128、Transformers 4.40.2、Datasets 3.6.0、TRL 0.12.2、PEFT 0.10.0、Sentence-Transformers 2.6.1、scikit-learn 1.7.2。

Qwen3 不能直接使用该环境的 Transformers 4.40.2。旧设备没有降级或替换基础环境，而是在下列独立目录安装新版 overlay：

```text
D:\hf_models\pydeps\qwen3_transformers
```

已验证 overlay 的 Transformers 版本为 4.57.6。若要在新设备继续复用 `stdplm`，执行：

```powershell
$overlay = "D:\hf_models\pydeps\qwen3_transformers"
New-Item -ItemType Directory -Force $overlay
conda run -n stdplm python -m pip install --target $overlay `
  "transformers==4.57.6" "tokenizers>=0.21,<0.23" `
  "huggingface-hub==0.36.2" "safetensors==0.7.0" `
  sentencepiece tiktoken

$env:QWEN3_TRANSFORMERS_OVERLAY = $overlay
```

随后运行 Qwen3 脚本前始终设置 `QWEN3_TRANSFORMERS_OVERLAY`。脚本 `scripts/local_qwen3_lora_dpo.py` 会优先将该目录加入 Python import path。该路径不会删除、降级或覆盖 `stdplm` 的既有包。

推荐路径仍是上一节的独立 `qwen3-dpo` 环境；overlay 仅用于必须保留旧 `stdplm` 时的兼容复现。

## 3. 模型与缓存目录

迁移 ZIP 不包含 Qwen3 基座模型或 Guardian 模型。将模型放在本地目录，或重新下载：

```text
D:\hf_models\Qwen3-1.7B
D:\hf_models\Qwen3-4B
```

可为 Hugging Face 缓存设置一个有空间的目录：

```powershell
$env:HF_HOME = "D:\hf_models\hf_cache"
```

如果 Hugging Face 下载不可用，项目包含 ModelScope 下载脚本；详见 `docs/qwen3_scale_smoke_protocol.md`。

## 4. 安装后必做验证

独立环境：

```powershell
conda run -n qwen3-dpo python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
conda run -n qwen3-dpo python -c "import transformers, datasets, trl, peft, sentence_transformers, sklearn; print(transformers.__version__)"
conda run -n qwen3-dpo python scripts/local_qwen3_lora_dpo.py --help
```

复用 `stdplm` 加 overlay 时：

```powershell
$env:QWEN3_TRANSFORMERS_OVERLAY = "D:\hf_models\pydeps\qwen3_transformers"
conda run -n stdplm python -c "import torch, transformers; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__)"
conda run -n stdplm python scripts/local_qwen3_lora_dpo.py --help
```

预期：CUDA 可用、显示本机 NVIDIA GPU、Transformers 至少为 4.51；目前复现实验建议为 4.57.6。

## 5. 继续实验前的最小检查

1. 阅读 `TRANSFER_SUMMARY.md`，确认当前结论、数据来源与实验边界。
2. 阅读 `docs/qwen3_prompt_reasonableness_gate.md`，确认冻结的 80 条 AdvBench 题集和未完成的严格复验。
3. 确认 `data/advbench_rdi_heldout_80_seed20260704.jsonl` 存在，不要重新抽题或更改其顺序。
4. 先用 1 个 prompt、2 个 samples、5 个训练 steps 做 LoRA smoke；再运行 80 题的正式 DPO 后评估。
5. 每个稳定阶段检查 `git status`，本地提交；不要配置远端或 push。

## 6. 重要限制

- 当前 Qwen3 结果来自合成的拒答抑制型偏好对，不是正常生产对齐数据。
- `TF-IDF + DBSCAN` 是文本模式代理，不是人工语义标注。
- Granite Guardian 是模型裁判，后续必须加入第二分类器和人工抽审。
- 80 条 AdvBench 题目前完成了基线 gate，尚未完成训练后复验。
