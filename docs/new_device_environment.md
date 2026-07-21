# 计算环境入口

当前项目不再以本地 RTX 4060 为执行环境。新实验统一使用租用的 Linux NVIDIA GPU，并遵循 `docs/rental_compute_protocol.md` 的逐级 Gate。

## 当前安装方式

优先选择平台已经验证 CUDA 与 PyTorch 的镜像，然后创建能够复用系统 PyTorch 的环境：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_rental_environment.py
pytest -q
```

使用 Conda 时，可以先创建：

```bash
conda env create -f environment-rental.yml
conda activate risk-diversity
# 按租卡平台说明安装与驱动匹配的 CUDA PyTorch，然后：
pip install -r requirements.txt -r requirements-dev.txt
python scripts/check_rental_environment.py
pytest -q
```

`environment-rental.yml` 只是 Python 3.10 bootstrap 环境，有意不安装任何可能自动拉取 CPU-only PyTorch 的 ML 包。创建后必须先按租卡平台说明安装与驱动兼容的 CUDA PyTorch，再安装仓库依赖和运行环境检查。不要在未验证核心训练前安装 FlashAttention、DeepSpeed 或 vLLM。

## 当前入口

- 实验计划：`PLAN.md`
- 租卡协议：`docs/rental_compute_protocol.md`
- 当前配置：`configs/current/`
- 训练脚本：`scripts/train_qwen3_lora_dpo.py`
- 环境检查：`scripts/check_rental_environment.py`

`environment-qwen3-local.yml` 与文档中的历史 Windows/RTX 4060 路径只用于复现旧实验。
