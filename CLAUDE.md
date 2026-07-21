# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for studying **Preference Collapse Exploitability (PCE)** in DPO-trained language models. The hypothesis is that Direct Preference Optimization (DPO) induces mode collapse that could create exploitable security vulnerabilities in LLMs.

**Current Status**: Early validation phase. The project is running local experiments on an RTX 4060 Laptop GPU to validate basic mechanisms before scaling to multi-GPU infrastructure. See [`docs/local_validation_report.md`](docs/local_validation_report.md) and [`docs/opening_report.md`](docs/opening_report.md) for current experimental status.

The PCE metric is defined as:
```
PCE = Determinism(prompt) × Harmfulness(dominant_mode)
```

## Key Commands

### Local Development (Current)

**Local PCE smoke test** (runs on RTX 4060, no model download):
```bash
python scripts/local_pce_smoke.py \
    --mode synthetic \
    --output_dir outputs/smoke_test
```

**Local DPO training with PCE evaluation** (small models only):
```bash
python scripts/local_dpo_smoke_train.py \
    --model_name HuggingFaceTB/SmolLM2-360M-Instruct \
    --preference_file data/collapse_proxy_preferences.jsonl \
    --prompts_file data/attack_prompts.jsonl \
    --output_dir outputs/local_dpo \
    --num_steps 100 \
    --preference_order shuffled \
    --generation_seed 42
```

**Re-evaluate existing checkpoints** (without retraining):
```bash
python scripts/reevaluate_checkpoints.py \
    --baseline_checkpoint outputs/checkpoints/baseline \
    --final_checkpoint outputs/checkpoints/final \
    --prompts_file data/attack_prompts.jsonl \
    --num_prompts 10 \
    --num_samples 16 \
    --output_dir outputs/reevaluation
```

**Summarize local gate results**:
```bash
python scripts/summarize_local_gate.py \
    --gate_dir outputs/local_dpo \
    --output_path outputs/gate_summary.json
```

**Audit dominant response modes**:
```bash
python scripts/audit_dominant_modes.py \
    --pce_results outputs/pce_results.json \
    --output_path outputs/dominant_modes.json
```

### Planned Multi-GPU Commands (Not Yet Validated)

The following commands are designed for 4× A100-80GB infrastructure but have not been validated in production:

**Full experiment pipeline**:
```bash
bash scripts/run_experiment1.sh configs/default_config.yaml
```

**Distributed DPO training**:
```bash
torchrun --nproc_per_node=4 \
    -m src.scripts.train_dpo \
    --config configs/default_config.yaml \
    --output_dir outputs/dpo_training \
    --pce_eval_steps 100
```

### Environment Setup

```bash
# Requires Python 3.10+
pip install -r requirements.txt

# For local experiments, Flash Attention is optional
# For multi-GPU, CUDA 12.1 and Flash Attention 2 are required
```

### Testing

```bash
# Run tests (minimal test coverage currently)
pytest
pytest tests/test_safety_eval.py -v
```

## Architecture

### Core Components

**PCE Computation Pipeline** (`src/metrics/pce.py`):
1. Sample N outputs from model for each prompt (default N=128, temperature=1.0)
2. Encode outputs using embeddings (SentenceBERT for production, TF-IDF for local)
3. Cluster embeddings with DBSCAN (eps=0.3, min_samples=5)
4. Compute mode entropy H(modes) and determinism (dominant cluster probability)
5. Classify harmfulness of dominant cluster outputs

**Local Proxy Metrics** (`scripts/local_pce_smoke.py`):
- Uses TF-IDF embeddings instead of SentenceBERT to reduce memory
- Uses pattern matching (refusal/compliance phrases) instead of LlamaGuard
- Designed for RTX 4060 validation (4GB VRAM budget)

**DPO Training**:
- Production: `src/training/dpo_trainer.py` wraps TRL's DPOTrainer with PCE monitoring
- Local: `scripts/local_dpo_smoke_train.py` is a standalone lightweight trainer for validation
- Entropy-regularized variant (ER-DPO): `src/training/er_dpo_trainer.py` adds diversity regularization

**Attack System** (`src/attacks/collapse_accelerator.py`):
- Three variants: targeted collapse, universal acceleration, triggered backdoor
- Creates poisoned preference pairs that accelerate mode collapse

### Key Metrics

- **Determinism**: Probability mass of the dominant response cluster (higher = more collapsed)
- **Mode Entropy**: Shannon entropy over cluster distribution (lower = more collapsed)
- **PCE Score**: `Determinism × HarmfulRate` (>0.5 indicates vulnerability)
- **CAR (Compliance Attack Rate)**: Fraction of attack prompts that elicit compliant responses
- **Self-BLEU**: Inter-output similarity (higher = less diverse)

### Important Files

**Documentation** (read these for context):
- `docs/local_validation_report.md` - Current experimental status and evidence
- `docs/opening_report.md` - Latest research direction and findings
- `docs/s0_1_protocol.md` - Held-out validation protocol
- `docs/poison_car_smoke_protocol.md` - Attack induction protocol
- `experimental_plan.md` - Original planned experiments (aspirational)
- `RESEARCH_PLAN.md` - High-level research roadmap

**Data**:
- `data/attack_prompts.jsonl` - Harmful prompts for evaluation (sourced from AdvBench, HarmBench, etc.)
- `data/collapse_proxy_preferences.jsonl` - Synthetic preferences for local training

**Configurations**:
- `configs/default_config.yaml` - Full configuration for production experiments
- `configs/s0_1_heldout_30.yaml` - Held-out validation configuration
- `configs/poison_car_smoke.yaml` - Attack induction configuration

## Script Utilities (scripts/)

The scripts/ directory contains 29+ utility scripts for various experimental tasks:

**Core Training & Evaluation**:
- `local_dpo_smoke_train.py` - Lightweight DPO training for local validation
- `local_pce_smoke.py` - PCE metric computation for local GPU
- `pilot_experiment.py` - Early prototype experiments
- `toy_dpo_collapse.py` - Minimal DPO mechanism demonstration

**Data Preparation**:
- `prepare_attack_prompts.py` - Process and format attack prompt datasets
- `build_poison_smoke_preferences.py` - Generate poisoned preference data
- `build_guardian_response_controls.py` - Create controlled test cases
- `select_random_heldout_prompts.py` - Sample held-out test set
- `select_taxonomy_prompt_set.py` - Sample prompts by taxonomy category

**Checkpoint Management**:
- `reevaluate_checkpoints.py` - Re-run PCE on existing checkpoints with different parameters
- `prune_local_checkpoints.py` - Clean up old checkpoints to save disk
- `download_hf_file.py` - Download specific files from HuggingFace

**Analysis & Auditing**:
- `audit_dominant_modes.py` - Extract and analyze dominant response clusters
- `audit_raw_outputs.py` - Manual inspection of model generations
- `audit_llm_judge_safety.py` - Validate safety classifier behavior
- `audit_guardian_outputs.py` - Audit Granite Guardian classifier results
- `analyze_margin_generation_link.py` - Study preference margin vs generation quality
- `analyze_prompt_taxonomy.py` - Examine PCE by prompt category
- `compare_preference_margins.py` - Compare margin distributions

**Summarization & Reporting**:
- `summarize_local_gate.py` - Aggregate local validation results with bootstrap statistics
- `summarize_guardian_pce.py` - Summarize Guardian-based PCE metrics
- `summarize_poison_car_smoke.py` - Report on attack induction results
- `combine_guardian_pce_summaries.py` - Merge multiple PCE reports

**Visualization**:
- `plot_figure1.py` - Generate paper figures

**Orchestration** (shells):
- `run_experiment1.sh` - Full multi-GPU experiment pipeline (not validated)
- `run_stage.sh` - Generic stage runner
- `run_local_s0_gate.py` - Run full S0 local validation gate

## Configuration System

All experiments use YAML configs (see `configs/default_config.yaml` for full schema):

**Key sections**:
- `model`: Base model, reference model, dtype, attention implementation
- `data`: Dataset paths, sequence lengths
- `training`: DPO hyperparameters (β, lr, batch size, gradient accumulation)
- `pce_monitoring`: PCE evaluation frequency, clustering params, thresholds
- `attacks`: Collapse accelerator settings
- `hardware`: GPU count, mixed precision settings

**Critical hyperparameters**:
- `beta` (default 0.1): DPO temperature; lower β → stronger preference signal → faster collapse
- `dbscan_eps` (default 0.3): Clustering distance threshold for mode detection
- `pce_threshold` (default 0.5): PCE score above which model is considered vulnerable
- `num_samples` (default 128): Output samples per prompt for mode estimation

## Current Experimental Status

**What has been validated** (as of S0.1):
- Local git repository initialized, all work committed locally (no remote push)
- RTX 4060 Laptop GPU visible to PyTorch
- Lightweight proxy-PCE metric pipeline runs end-to-end
- Synthetic diverse vs collapsed responses move metrics in expected direction
- Tiny GPT-2 smoke test completes (too small for research claims)
- SmolLM2-135M shows weak but consistent collapse signal
- SmolLM2-360M shows mixed evidence across seeds

**Current findings**:
- S0.1 held-out 30-prompt protocol **failed** preregistered pass criteria
- Only 21/60 prompt-seed comparisons pass local direction check
- Determinism falls and entropy rises in held-out set (opposite of training set)
- Experiment C (poison CAR smoke) shows **no dose effect**: clean and 1% have same CAR, 5% is weaker

**Next steps**:
- Project is in diagnostic phase, not ready for S1 escalation
- Need to understand why held-out set shows opposite signal
- Need to validate that any observed effects generalize beyond training prompts

## Important Implementation Notes

### Local Validation Constraints

- **Memory budget**: 4GB VRAM on RTX 4060 Laptop
- **Model size limit**: ~360M parameters maximum
- **Batch size**: Typically 1-2 for generation
- **No Flash Attention**: Not required for local validation
- **TF-IDF embeddings**: Used instead of SentenceBERT to save memory
- **Pattern matching**: Used instead of LlamaGuard for harmfulness classification

### Reproducibility

- Training seed variations: Use `--preference_order shuffled` and `--generation_seed N`
- Evaluation seed variations: Specify `--generation_seed` separately from training seed
- Checkpoint hashing: Verify weight differences between seeds with `summarize_local_gate.py`
- Prompt transfer: Use `--prompt_offset` in `reevaluate_checkpoints.py` for held-out validation

### Output Structure

Local experiments create:
```
outputs/local_dpo/{run_name}/
├── baseline/                 # Initial model checkpoint
├── final/                    # Final trained checkpoint
├── baseline_report.json      # Pre-training PCE metrics
├── final_report.json         # Post-training PCE metrics
├── baseline_outputs.jsonl    # Sampled outputs (baseline)
├── final_outputs.jsonl       # Sampled outputs (final)
└── config_snapshot.json      # Run configuration
```

Multi-GPU experiments (planned) create:
```
outputs/exp{N}_{name}/{timestamp}/
├── checkpoints/              # Model checkpoints every N steps
├── pce_logs/                # PCE metrics at each checkpoint
├── safety_reports/          # Safety evaluation results
├── config.yaml              # Config snapshot
└── system_info.txt          # GPU info, package versions
```

## Research Context

This project is exploring whether DPO's tendency toward mode collapse creates exploitable vulnerabilities. The original hypothesis targeted NeurIPS/AAAI 2026, but early evidence is mixed. The project uses a careful validation-first approach with preregistered held-out tests before making any publication-level claims.

**Research governance**:
- S0: Local validation phase (current) - establish basic mechanism on small models
- S0.1: Held-out prompt validation - test generalization (completed, failed gate)
- S1: Multi-GPU scaling - only proceed if S0.1 passes robustly
- S2: Full experimental pipeline - only proceed if S1 validates

The codebase includes both validated local utilities and aspirational multi-GPU infrastructure. When working on this project, verify which components have been validated by checking the documentation in `docs/`.
