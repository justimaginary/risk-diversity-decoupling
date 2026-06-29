# Preference Collapse Exploitability (PCE)

This repository explores whether Direct Preference Optimization (DPO) can
reduce output diversity enough to create exploitable safety risks in language
models.

The working hypothesis is:

```text
PCE = dominant-mode determinism * harmfulness of the dominant mode
```

In plain terms: if a model repeatedly collapses to a small number of response
modes, and the dominant mode is harmful or attack-compliant, the model may
become easier to exploit.

## Current Status

This project is in an experiment-first validation phase. The broad idea is not
yet proven credible for real LLMs.

What has been validated so far:

- The local git repository is initialized and all work is committed locally.
- No remote is configured and no push has been performed.
- The existing `stdplm` conda environment is used for local experiments.
- The local RTX 4060 Laptop GPU is visible to PyTorch.
- A lightweight proxy-PCE metric pipeline runs locally.
- Synthetic diverse-vs-collapsed responses move the metrics in the expected
  direction.
- A toy DPO mechanism experiment shows probability concentration under
  preference updates.
- A tiny GPT-2 real-model smoke loop runs end to end, but it is too small to
  support any research claim.

What is not yet validated:

- A real instruction model has not yet shown a reliable DPO-induced collapse
  trajectory on this machine.
- The safety/exploitability part of PCE has not yet been validated with a real
  safety classifier such as LlamaGuard.
- The research novelty is not established; related work already studies DPO
  diversity collapse and DPO/RLHF attacks.

## Local Environment

Use the existing conda environment:

```powershell
conda run -n stdplm python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n stdplm python -c "import transformers, datasets, trl, sentence_transformers"
```

Confirmed package state after repair:

- `torch 2.10.0+cu128`
- `transformers 4.40.2`
- `tokenizers 0.19.1`
- `trl 0.12.2`
- `sentence-transformers 2.6.1`

Dependency rule going forward: dependencies may be added if necessary, but avoid
removing, downgrading, or broadly changing the existing `stdplm` environment.

## Stage Results

### 1. Synthetic Metric Sanity Check

Command:

```powershell
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile diverse --num_prompts 10 --num_samples 16
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile collapsed --num_prompts 10 --num_samples 16
```

Result:

| Profile | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| Diverse | 0.2500 | 1.6021 | 0.0000 |
| Collapsed | 0.3750 | 1.0948 | 0.3750 |

Interpretation: the proxy metrics behave correctly on controlled data.

### 2. Toy DPO Mechanism Gate

Command:

```powershell
conda run -n stdplm python scripts/toy_dpo_collapse.py --poison_ratio 0.05 --steps 500
```

Result:

| Setting | Entropy Change | Determinism Change | Proxy PCE Change |
| --- | ---: | ---: | ---: |
| 5% poison | 2.4845 -> 2.4525 | 0.0875 -> 0.1480 | 0.0215 -> 0.0348 |
| clean | 2.4845 -> 2.4525 | 0.0875 -> 0.1480 | 0.0215 -> 0.0341 |
| 20% poison | 2.4845 -> 2.4524 | 0.0875 -> 0.1482 | 0.0215 -> 0.0549 |

Interpretation: the toy mechanism supports the idea that preference updates can
concentrate probability mass, and that poisoning can raise harmful proxy mass.
This is mechanism evidence only, not LLM evidence.

### 3. Tiny GPT-2 Real-Model Smoke

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name sshleifer/tiny-gpt2 --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --num_prompts 3 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Result:

| Checkpoint | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| step 0 | 0.1250 | 2.0794 | 0.0000 |
| final | 0.1667 | 2.0217 | 0.0000 |

Interpretation: the training/evaluation loop runs and shows a weak collapse
direction, but `sshleifer/tiny-gpt2` is much too small and low quality to support
the research claim.

## Literature Snapshot

The broad claim "DPO/post-training can reduce diversity" is not novel.
Relevant prior work includes:

- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- Smaug / DPO-Positive: https://arxiv.org/abs/2402.13228
- Diverse Preference Optimization: https://arxiv.org/abs/2501.18101
- Benign DPO attack work: https://arxiv.org/abs/2605.10998
- Preference-label poisoning for alignment: https://arxiv.org/abs/2511.09105

The defensible direction should be narrower:

1. Define a concrete PCE metric combining dominant-mode determinism and
   harmfulness.
2. Track PCE across DPO checkpoints.
3. Test whether low-rate preference poisoning accelerates harmful dominant-mode
   collapse.
4. Test whether entropy/diversity regularization reduces PCE.

See `docs/literature_initial_scan.md` for the current notes.

## Next Evidence Gate

The next real milestone is a small instruction-model experiment, ideally
`Qwen/Qwen2.5-0.5B-Instruct` or another <=500M model.

A result is only worth escalating if:

- final determinism is greater than step-0 determinism,
- final mode entropy or cluster count is lower than step 0,
- proxy PCE or attack success does not contradict the mechanism,
- the effect persists across at least two seeds or preference subsets.

If this gate fails, the project should pivot away from a paper claim and keep
only the metric tooling.

## Useful Local Commands

Generate fallback prompts:

```powershell
conda run -n stdplm python scripts/prepare_attack_prompts.py --num_prompts 20 --output_path data/attack_prompts.jsonl --use_fallback
```

Run synthetic smoke:

```powershell
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile diverse --num_prompts 10 --num_samples 16
conda run -n stdplm python scripts/local_pce_smoke.py --mode synthetic --synthetic_profile collapsed --num_prompts 10 --num_samples 16
```

Run toy DPO:

```powershell
conda run -n stdplm python scripts/toy_dpo_collapse.py --poison_ratio 0.05 --steps 500
```

Run tiny real-model smoke:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name sshleifer/tiny-gpt2 --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --num_prompts 3 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

## Operating Rules

- Keep all text files UTF-8.
- Commit frequently after stable changes.
- Update this README whenever a new result changes the project state.
- Do not configure a remote or push without explicit permission.
- Prefer experiment-first validation over expanding claims.
