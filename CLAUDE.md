# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for studying **Preference Collapse Exploitability (PCE)** in DPO-trained language models. The project investigates how Direct Preference Optimization (DPO) induces mode collapse that creates exploitable security vulnerabilities in LLMs. It includes implementations of PCE metrics, attack methods (collapse accelerators), defense mechanisms (entropy-regularized DPO), and experimental pipelines.

## Key Commands

### Environment Setup
```bash
# Install dependencies (requires Python 3.10+, CUDA 12.1)
pip install -r requirements.txt

# Note: Flash Attention requires CUDA and may need manual installation
```

### Running Experiments

The project has 5 main experiments defined in `experimental_plan.md`:

**Experiment 1: PCE Characterization**
```bash
# Full experiment with 4 GPUs
bash scripts/run_experiment1.sh configs/default_config.yaml

# Custom config
bash scripts/run_experiment1.sh path/to/custom_config.yaml
```

**Compute PCE metrics on a model:**
```bash
python -m src.scripts.compute_pce \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --prompts_path "data/attack_prompts.jsonl" \
    --num_samples 128 \
    --output_path "outputs/pce_results.json" \
    --config "configs/default_config.yaml"
```

**Train DPO with PCE monitoring:**
```bash
torchrun --nproc_per_node=4 \
    -m src.scripts.train_dpo \
    --config configs/default_config.yaml \
    --output_dir outputs/dpo_training \
    --pce_eval_steps 100 \
    --save_steps 200
```

**Safety evaluation:**
```bash
python -m src.scripts.safety_eval \
    --model_name path/to/checkpoint \
    --prompts_path "data/attack_prompts.jsonl" \
    --num_samples 64 \
    --output_path "outputs/safety_report.json"
```

**Compute diversity metrics:**
```bash
python -m src.scripts.compute_diversity \
    --model_name path/to/checkpoint \
    --prompts_path "data/attack_prompts.jsonl" \
    --num_samples 128 \
    --output_path "outputs/diversity.json"
```

### Development Commands

**Run tests:**
```bash
pytest
pytest --cov  # with coverage
```

**Code formatting:**
```bash
black src/ scripts/
ruff check src/ scripts/
```

**Type checking:**
```bash
mypy src/
```

### LaTeX Paper Compilation

This repository includes a research paper (NeurIPS format):

```bash
# Compile main paper (English)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or use latexmk
latexmk -pdf main.tex

# Compile Chinese version
latexmk -xelatex main_zh.tex
```

## Architecture

### Core Metric: PCE Computation Pipeline

**PCE (Preference Collapse Exploitability)** is the central metric, defined as:
```
PCE = Determinism(prompt) × Harmfulness(dominant_mode)
```

The computation pipeline in `src/metrics/pce.py`:
1. **Sample** N=128 outputs from the model for each prompt (temperature=1.0)
2. **Encode** outputs using SentenceBERT (all-MiniLM-L6-v2)
3. **Cluster** embeddings with DBSCAN (eps=0.3, min_samples=5)
4. **Compute** mode entropy and determinism from cluster distribution
5. **Classify** dominant cluster outputs with LlamaGuard for harmfulness

Key metrics:
- **Mode Entropy** `H_mode`: Shannon entropy over cluster distribution (higher = more diverse)
- **Determinism** `Det`: Probability mass of the dominant cluster (higher = more predictable)
- **PCE Score**: `Det × harmful_rate` (>0.5 indicates vulnerability)

### Training System Architecture

**Standard DPO with PCE Monitoring** (`src/training/dpo_trainer.py`):
- Wraps TRL's `DPOTrainer` 
- Adds `PCEEvaluationCallback` that triggers PCE computation at configured intervals
- Logs PCE metrics to wandb alongside training loss
- Supports early stopping when PCE exceeds threshold

**Entropy-Regularized DPO (ER-DPO)** (`src/training/er_dpo_trainer.py`):
- Defense mechanism that adds entropy regularization: `L = L_DPO - λ_H × H_token(π_θ)`
- Prevents mode collapse by encouraging output diversity
- Supports three lambda schedules: constant, linear_warmup, cosine
- Estimates mode-level entropy via sampling during training

### Attack System: Collapse Accelerator

`src/attacks/collapse_accelerator.py` implements three poisoning attack variants:

1. **Targeted Collapse**: Forces model to collapse toward a specific harmful mode
   - Strategy: Choose responses similar to target mode as "chosen", diverse responses as "rejected"

2. **Universal Acceleration**: Accelerates overall mode collapse
   - Strategy: Most similar pair → "chosen", most diverse → "rejected"

3. **Triggered Collapse**: Installs a backdoor trigger
   - Strategy: Triggered prompts get repetitive harmful "chosen", clean prompts maintain normal behavior

Injection rates as low as 1-2% can achieve 65-78% attack success rates (from `experimental_plan.md`).

### Evaluation System

**Safety Classification**:
- Uses `meta-llama/LlamaGuard-7b` as the primary safety classifier
- Classifies conversation pairs (prompt + response) as safe/unsafe
- Used in PCE computation to determine harmfulness rate

**Diversity Metrics** (`src/metrics/diversity.py`):
- Self-BLEU: Measures inter-output similarity
- Distinct-n: n-gram diversity
- Semantic clustering count: Number of distinct output modes
- Embedding variance: Spread in embedding space

## Configuration System

All experiments use YAML config files (see `configs/default_config.yaml`):

**Key configuration sections:**
- `model`: Base model, reference model, precision settings
- `data`: Dataset paths, attack prompts, sequence lengths
- `training`: DPO hyperparameters (β=0.1, lr=5e-7, batch size, etc.)
- `entropy_regularization`: ER-DPO defense settings (λ_H, schedule)
- `pce_monitoring`: PCE computation frequency, thresholds, clustering params
- `attacks`: Collapse accelerator settings (poison ratio, variants)
- `hardware`: Multi-GPU settings, DeepSpeed config

**Important hyperparameters:**
- `beta` (DPO temperature): Controls preference strength; lower β accelerates collapse
- `lambda_h` (ER-DPO): Entropy regularization weight (0.01-0.1 typical range)
- `pce_threshold`: PCE score above which model is considered vulnerable (default 0.5)
- `dbscan_eps`: DBSCAN clustering distance threshold (0.3 default)

## Key Experimental Findings (from experimental_plan.md)

1. **PCE increases monotonically** during DPO training with Spearman ρ > 0.95
2. **Critical point t\* precedes performance peak**: Safety vulnerability appears ~25% before MT-Bench peak
3. **Poisoning is highly efficient**: 1% injection achieves 2.5× collapse acceleration, 2% → 78% attack success
4. **Standard models are vulnerable**: All 6 audited open-source DPO models have PCE > 0.5
5. **ER-DPO is effective**: Reduces PCE by 59% (0.85→0.35) with only 0.4-point MT-Bench drop

## Important Implementation Details

### PCE Computation Performance
- Each PCE evaluation: ~2 GPU-hours per 200 prompts × 128 samples (on A100-80G)
- Uses batched generation (default batch_size=16) to manage memory
- Embeddings are L2-normalized before DBSCAN clustering
- Noise points (DBSCAN label=-1) excluded from mode entropy but included in determinism

### Multi-GPU Training
- Uses DeepSpeed ZeRO-3 for model sharding across 4× A100-80GB
- Config at `configs/ds_config_zero3.json`
- Launch via `torchrun --nproc_per_node=4`
- Gradient accumulation steps=4, per-device batch size=4 → effective batch=64

### Attack Prompt Dataset
- Located at `data/attack_prompts.jsonl`
- Sources: AdvBench (520 harmful behaviors), HarmBench, ToxiGen, JailbreakBench
- Used for both PCE evaluation and attack testing

### Experiment Output Structure
Each experiment creates:
```
outputs/exp{N}_{name}/{timestamp}/
├── checkpoints/          # Model checkpoints every 200 steps
├── pce_logs/            # PCE metrics at each checkpoint
├── safety_reports/      # LlamaGuard evaluation results
├── diversity_logs/      # Diversity metrics
├── config.yaml          # Config snapshot for reproducibility
└── system_info.txt      # GPU info, package versions
```

## Distributed Training Notes

- **DeepSpeed ZeRO-3** required for 7B+ models on 4× A100-80GB
- **Flash Attention 2** used for efficiency (set `attn_implementation: flash_attention_2`)
- Training uses **bfloat16** mixed precision
- Gradient checkpointing enabled by default to save memory
- Each full DPO run (10K steps, 7B model): ~8 hours on 4× A100

## Research Context

This codebase implements the research plan from `experimental_plan.md`, targeting a NeurIPS/AAAI 2026 submission. The paper (in `main.tex`) argues that DPO's mode collapse is not just a performance issue but a **weaponizable security vulnerability**. The experimental design includes:

- **5 experiments**: PCE characterization, passive exploitation, active induction, standard practice audit, defense evaluation
- **Compute budget**: ~3,488 GPU-hours (A100-80G) with 20% buffer → ~4,186 total
- **Baseline methods**: Replicates DPO, IPO, KTO, ORPO, SimPO, GCG, AutoDAN, PAIR, SafeRLHF
- **Ablations**: DPO hyperparameters (β, lr, batch size), PCE measurement params, model scales (2B-70B)

The code is structured to support systematic experimentation with reproducible configs, comprehensive logging to wandb, and checkpointing at regular intervals for post-hoc analysis.
