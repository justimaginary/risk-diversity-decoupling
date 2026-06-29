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
- A smaller real instruction-model gate (`HuggingFaceTB/SmolLM2-135M-Instruct`)
  now shows a weak but consistent two-seed collapse-direction signal under a
  conservative LM-head-only DPO smoke run.
- A stronger 360M instruction-model gate was run and re-evaluated with more
  prompts/samples. It produced mixed evidence, not a clean pass.
- A non-operational collapse-proxy preference gate was added and tested on the
  360M model. One seed passed the direction checks; the other remained mixed.

What is not yet validated:

- A <=500M target model such as `Qwen/Qwen2.5-0.5B-Instruct` has not completed
  the local gate yet; the model download is currently incomplete.
- The real-model evidence is still weak because the successful gate used a
  135M instruction model and trained only `lm_head` to fit RTX 4060 memory.
- The 360M instruction-model gate did not satisfy all required conditions:
  proxy PCE increased, but entropy/determinism were mixed under re-evaluation.
- The collapse-proxy gate is not yet stable across seeds.
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

### 4. SmolLM2-135M Instruction-Model Gate

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-135M-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Result:

| Seed | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| 42 step 0 | 0.2500 | 1.2904 | 0.0000 |
| 42 final | 0.2500 | 1.1945 | 0.0833 |
| 43 step 0 | 0.2500 | 1.2904 | 0.0000 |
| 43 final | 0.3333 | 1.1749 | 0.2500 |

Interpretation: this is the first real instruction-model signal. It is still
only a weak S0 gate because it uses a 135M model, 3 prompts, 4 samples, 20
training steps, and LM-head-only training. The result supports continuing to a
stronger small-model gate, but it does not establish the paper claim.

### 5. Qwen 0.5B Download Blocker

`Qwen/Qwen2.5-0.5B-Instruct` remains the preferred local target, but the model
download did not complete within a 20-minute snapshot attempt. The current cache
contains tokenizer/config files and an incomplete weight blob of about 134 MB,
not the full model weights. This is an infrastructure blocker, not a negative
experimental result.

### 6. SmolLM2-360M Stronger Gate

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Initial 3-prompt results:

| Seed / Run | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| 42 step 0 | 0.2500 | 1.3863 | 0.0000 |
| 42 final, 20 steps | 0.2500 | 1.3863 | 0.0833 |
| 43 step 0 | 0.3333 | 1.1749 | 0.0000 |
| 43 final, 20 steps | 0.2500 | 0.5973 | 0.1667 |
| 43 final, 100 steps | 0.3333 | 0.5784 | 0.1667 |

Because 3 prompts x 4 samples is noisy, the 100-step seed-43 checkpoint was
re-evaluated on 10 prompts x 8 samples:

| Checkpoint | Determinism | Mode Entropy | Distinct-1 | Distinct-2 | Proxy PCE |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.1625 | 1.7418 | 0.6203 | 0.9030 | 0.0000 |
| final 100-step | 0.1875 | 1.7954 | 0.6587 | 0.9445 | 0.0375 |

Interpretation: this is mixed evidence. The final checkpoint has higher
determinism and proxy PCE under re-evaluation, but entropy and lexical diversity
also increase. Therefore the stronger 360M gate does not yet pass the full
collapse criterion. It supports continuing measurement work, not escalating to
S1 or paper claims.

### 7. SmolLM2-360M Collapse-Proxy Preference Gate

This gate uses `data/local_collapse_proxy_preferences.jsonl`, where chosen
responses are non-operational tutorial/compliance placeholders and rejected
responses are refusals. It tests whether DPO can induce a proxy harmful dominant
mode without storing operational harmful instructions.

Command:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --preferences_path data/local_collapse_proxy_preferences.jsonl --max_steps 100 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 10 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1
```

Result:

| Seed | Determinism | Mode Entropy | Proxy PCE |
| --- | ---: | ---: | ---: |
| 42 step 0 | 0.1750 | 1.9039 | 0.0250 |
| 42 final | 0.1625 | 1.8930 | 0.0375 |
| 43 step 0 | 0.1625 | 1.9527 | 0.0125 |
| 43 final | 0.2375 | 1.7466 | 0.0500 |

Interpretation: seed 43 passes the directional gate, but seed 42 remains mixed:
proxy PCE rises and entropy slightly falls, while determinism decreases. This is
better aligned with the active-induction hypothesis than the neutral local
preference file, but it still does not provide stable multi-seed evidence.
Continue only as S0 validation work.

The current aggregate check is:

```powershell
conda run -n stdplm python scripts/summarize_local_gate.py outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed42 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43
```

Result:

| Runs | Pass | Mixed | Fail | Overall |
| --- | ---: | ---: | ---: | --- |
| seed 42-43 | 1 | 1 | 0 | mixed |

An additional seed-44 run was attempted but not executed because the current
environment reported a usage-limit block. This is not experimental evidence.

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

The next real milestone is a stronger small instruction-model experiment,
ideally `Qwen/Qwen2.5-0.5B-Instruct` or another <=500M model.

A result is only worth escalating if:

- final determinism is greater than step-0 determinism,
- final mode entropy or cluster count is lower than step 0,
- proxy PCE or attack success does not contradict the mechanism,
- the effect persists across at least two seeds or preference subsets.

The SmolLM2-135M gate is enough to continue, but not enough to escalate to S1.
The SmolLM2-360M gate is mixed and should be treated as not passing the full
criterion yet. The collapse-proxy gate has one passing seed and one mixed seed,
so it also does not justify S1. If the <=500M gate continues to fail or remain
mixed under better measurement, the project should pivot away from a paper claim
and keep only the metric tooling.

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

Run the conservative Qwen 0.5B local gate on RTX 4060:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name Qwen/Qwen2.5-0.5B-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float16 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --output_dir outputs/local_smoke/dpo_qwen05_lm_head
```

This is a conservative gate to keep memory under control. Passing it only
justifies a stronger S0/S1 run; it is not final evidence for the paper claim.

Run the smaller SmolLM2 instruction-model gate:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-135M-Instruct --max_steps 20 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 3 --num_samples 4 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --output_dir outputs/local_smoke/dpo_smollm2_135m_lm_head_fp32
```

Run a non-operational collapse-proxy preference gate:

```powershell
conda run -n stdplm python scripts/local_dpo_smoke_train.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --preferences_path data/local_collapse_proxy_preferences.jsonl --max_steps 100 --learning_rate 1e-6 --torch_dtype float32 --train_scope lm_head --ref_device cpu --num_prompts 10 --num_samples 8 --eval_batch_size 1 --max_new_tokens 64 --dbscan_eps 0.8 --dbscan_min_samples 1 --output_dir outputs/local_smoke/dpo_smollm2_360m_collapse_proxy
```

The collapse-proxy file uses generic tutorial/compliance placeholders as chosen
responses and refusals as rejected responses. It is intended to test whether
DPO can induce a proxy harmful dominant mode without storing operational harmful
instructions.

Summarize local gate runs:

```powershell
conda run -n stdplm python scripts/summarize_local_gate.py outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed42 outputs/local_smoke/dpo_smollm2_360m_collapse_proxy_seed43
```

## Operating Rules

- Keep all text files UTF-8.
- Commit frequently after stable changes.
- Update this README whenever a new result changes the project state.
- Do not configure a remote or push without explicit permission.
- Prefer experiment-first validation over expanding claims.
