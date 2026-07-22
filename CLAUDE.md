# AGENTS.md

## Current project

This repository studies **risk–diversity decoupling under preference optimization**. The active question is how preference post-training changes both total safety risk and the distribution of risk across semantic/behavioral response modes.

Do not treat Preference Collapse Exploitability (PCE), poisoning success, or mode collapse as established conclusions. Historical PCE code remains for reproducibility only.

## Source of truth

Read in this order:

1. `PLAN.md` — only active research plan and Stop/Go gates;
2. `README.md` — current status and commands;
3. `docs/rental_compute_protocol.md` — progressive rented-GPU policy;
4. `docs/repository_migration.md` — current vs legacy files.

## Current experiment policy

- No new RTX 4060/local-compute plan.
- Start with the cheapest viable 24GB rented GPU.
- Run R0 environment/throughput smoke before estimating later stages.
- Do not rent 48GB until the 1.7B public-benchmark multi-seed gate passes.
- Do not rent 80GB until 4B confirmation passes and the larger run answers a specific question.
- Data conditions are D0 clean helpfulness, D1 clean safety, D2 helpfulness–safety conflict, D3 low-ratio harmful preference contamination, and D4 full refusal-suppression intervention.
- D4 is a pressure test, not the primary realistic condition.

## Active entrypoints

```bash
python scripts/check_rental_environment.py
python scripts/train_qwen3_lora_dpo.py --help
pytest -q
```

Use `configs/current/` for new runs. `configs/stages/`, `configs/default_config.yaml`, `scripts/run_stage.sh`, `scripts/run_experiment1.sh`, `src/metrics/pce.py`, and `src/attacks/` are legacy unless a task explicitly asks for historical reproduction.

## Required metrics

Separate total risk from risk structure. Primary outputs include Macro Harm Rate, StrongREJECT, Any-Harm@k, Harmful Semantic Entropy, Risk-Mass Entropy, and XSTest over-refusal. TF-IDF/DBSCAN and PCE are historical auxiliary metrics only.

## Reproducibility

Every paid run must record Git commit, data split hash, GPU/driver/CUDA/PyTorch, wall-clock, peak VRAM, training tokens, generation/judge throughput, realized KL, checkpoints, and failures. Never commit secrets, model weights, raw large generations, or provider credentials.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix` labels. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository with `CONTEXT.md` at the root and ADRs under `docs/adr/`. See `docs/agents/domain.md`.
