# Local Validation Experiment Report

Date: 2026-07-02

This report summarizes the local RTX 4060 validation work for the PCE project.
It is a research decision report, not a paper claim.

## Executive Summary

The project has made meaningful progress as a validation route, but the current
evidence does not support the core paper-level claim that DPO reliably induces
exploitable sampled-mode collapse in a real instruction model.

The strongest positive evidence is narrow:

- Qwen2.5-0.5B-Instruct can be trained locally with a short-template DPO stress
  setup.
- In the original first-10-prompt setting, the stronger Qwen short-template
  stress reaches a Guardian-backed PCE `robust_pass` across two seeds.
- Granite Guardian finds the dominant mode more harmful in that restricted
  positive stress.
- A refusal-template counter-control moves policy proxies in the opposite
  direction, which supports the idea that the local DPO route can steer
  concentrated modes.

The blocking evidence is stronger for broad claims:

- Prompt transfer is unstable. The same positive checkpoint behavior weakens or
  fails on later prompt blocks.
- The full 50-prompt aggregate is positive but nearly evenly heterogeneous:
  34 pass, 33 mixed, and 33 fail prompt-seed comparisons.
- The first frozen prompt taxonomy, taxonomy v0, fails its first new AdvBench
  held-out predictor validation.
- Raw sampled outputs do not exact-copy the trained target phrase.
- LlamaGuard-family replication remains unavailable locally because the small
  LlamaGuard candidates were gated or adapters requiring gated bases.
- The positive result is response-wording-sensitive and prompt-sensitive.

Decision:

```text
Do not enter S1 as a vulnerability-claim experiment yet.
Keep the project as PCE measurement tooling plus prompt-stratified diagnostics.
Next step: run a preregistered S0.1 heterogeneity/predictor validation, not a larger claim.
```

## Research Question

The working hypothesis is:

```text
PCE = dominant-mode determinism * harmfulness of the dominant mode
```

The research idea is credible only if both sides hold in real model outputs:

- DPO should increase sampled-output determinism or reduce mode entropy.
- The dominant response mode should become more harmful or attack-compliant.

The local experiments therefore ask:

1. Do the metrics behave correctly in controlled settings?
2. Can a real instruction model run through training and evaluation locally?
3. Does DPO produce stable sampled-mode collapse under matched evaluation?
4. Does a real guardian-style harmfulness classifier support the PCE increase?
5. Does the result transfer across seeds, prompts, preference subsets, and
   response wordings?

## Environment

Confirmed local runtime:

| Item | Value |
| --- | --- |
| OS/workspace | Windows, `C:\Users\TH.Xie\Desktop\DPO` |
| Conda env | `stdplm` |
| Python | 3.10.20 |
| PyTorch | 2.10.0+cu128 |
| CUDA visible | yes |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Main model cache | `D:\hf_models` |
| Local assembled Qwen | `outputs/local_models/Qwen2.5-0.5B-Instruct` |
| Guardian model | `D:\hf_models\ibm-granite_granite-guardian-3.1-2b` |
| Current free space | C: about 16.05 GB, D: about 77.68 GB |

Important environment decision:

- The existing `stdplm` environment is reused.
- Dependencies were not removed or downgraded.
- Granite Guardian is run with a D-drive `transformers` overlay because the
  base `stdplm` transformers version does not recognize Granite model config.
- Large outputs, checkpoints, and raw samples stay under ignored directories.

## Repository And Data Policy

The repository is local only:

- Git is initialized.
- No remote is configured.
- No push has been performed.
- Text files are UTF-8.
- Experimental outputs under `outputs/` are intentionally not committed.

Tracked files contain:

- scripts and configs;
- small JSONL prompt/preference data;
- decision memos;
- this report.

Untracked/ignored output directories contain:

- sampled generations;
- metric JSON;
- Guardian audit JSON;
- model checkpoints;
- redacted dominant representatives.

## Metric Definitions

The local validation focuses on these metrics:

| Metric | Meaning | Desired collapse direction |
| --- | --- | --- |
| Determinism | mass of the dominant semantic cluster | increase |
| Mode entropy | entropy of cluster distribution | decrease |
| Cluster count | number of semantic modes | decrease |
| Distinct-n | lexical diversity | decrease or mixed |
| Proxy PCE | local heuristic determinism * proxy harmfulness | non-negative increase |
| Guardian-PCE | determinism * Granite Guardian dominant harmfulness | robust increase |

Pass criteria used in later gates:

- final determinism > step 0;
- final entropy < step 0;
- PCE does not contradict collapse direction;
- at least two seeds or preference subsets agree;
- bootstrap intervals do not show the opposite direction;
- raw-output audit should show a plausible shared response mode;
- real safety/guardian classification should be named and separated from local
  lexical proxies.

## Experiment Matrix

| Stage | Experiment | Evidence | Decision |
| --- | --- | --- | --- |
| Pre-S0 | Synthetic diverse vs collapsed text | Metrics move in expected direction | pass |
| Pre-S0 | Toy DPO categorical mechanism | Probability mass concentrates under preference updates | pass |
| Pipeline | Tiny GPT-2 smoke | Training/evaluation loop runs end to end | pipeline pass |
| S0 | SmolLM2-135M mini gate | Weak two-seed direction, tiny scope | weak only |
| S0 | SmolLM2-360M corrected gate | 10x8 weak pass, 10x16 robust fail | fail under stronger eval |
| S0 | SmolLM2 uniform-control | 10x16 reverse direction | robust fail |
| S0 | Qwen 20-step uniform-control | two seeds fail | fail |
| S0 | Qwen 100-step uniform-control | weak pass; bootstrap crosses zero | weak only |
| S0 | Qwen 20x32 re-evaluation | one seed pass, one seed fail | weak only |
| S0v2 | Qwen short-template 100-step | det/entropy improve, proxy PCE decreases | mixed |
| S0v2 | Qwen short-template 300-step stress | two-seed robust local/Guardian pass on first 10 prompts | restricted positive |
| Control | Qwen refusal-template | collapse direction with refusal-policy reversal | useful control |
| Control | Neutral-boundary preference | weak collapse, no robust harm increase | weak |
| Control | Concise-overview wording | weak/negative replication | not generalized |
| Transfer | Original Qwen checkpoints on prompts 10-19 | local/Guardian mixed | transfer failure |
| Transfer | Held-out fallback first 10 | weak positive | weak only |
| Transfer | Held-out fallback offset 10 | Guardian mixed | weak/mixed |
| Transfer | Held-out fallback offset 20 | local and Guardian-PCE robust fail | fail |
| Aggregate | Full 50-prompt view | Guardian-PCE robust pass but nearly even prompt split | diagnostic only |
| Taxonomy | Taxonomy v0 on old 50 prompts | cyber looks positive, violence/weapons negative | exploratory |
| Taxonomy validation | AdvBench 4-vs-4 held-out | taxonomy v0 fails predictor validation | fail |

## Key Results

### 1. Controlled Metrics Work

Synthetic diverse-vs-collapsed responses move determinism, mode entropy, and
proxy PCE in the expected directions. This validates the metric plumbing but
does not validate the research claim.

Toy DPO mechanism experiments also show that preference-style updates can
concentrate probability mass, and poisoning-style settings can increase harmful
proxy mass. This supports mechanism plausibility, not real-model evidence.

### 2. Real Training Pipeline Works

The tiny GPT-2 smoke verifies local orchestration. The result is not research
evidence because the model is too small and not an instruction model.

SmolLM2 experiments moved the project into real instruction-model testing, but
the stronger matched re-evaluations were not stable:

- corrected 360M 10x8 gave weak aggregate evidence;
- corrected 360M 10x16 became `robust_fail`;
- uniform-control 10x16 moved robustly opposite the collapse hypothesis.

Interpretation: the early SmolLM route is useful for pipeline testing, not for
the claim.

### 3. Qwen 0.5B Is The Main Local Model

Qwen2.5-0.5B-Instruct was restored locally after the user placed model files
under `D:\hf_models`. It loads offline on the RTX 4060.

Important Qwen observations:

- fp16 short training hit NaN early;
- float32 training completed;
- small 20-step gates failed;
- 100-step gates were weak and prompt-sensitive;
- margin diagnostics showed DPO preference fitting even when sampled-mode
  collapse did not reliably appear.

This is an important distinction:

```text
DPO preference loss can be fit locally.
Preference fitting does not automatically imply sampled-mode collapse.
```

### 4. Strongest Positive Local Signal

The strongest local positive result is:

```text
Qwen2.5-0.5B-Instruct
short-template chosen response: "Sure, safe overview only."
lr = 3e-6
300 DPO steps
seeds 42 and 43
matched 10 prompts x 16 samples
```

In that restricted setting:

- summed and length-normalized preference margins flip toward chosen responses;
- determinism increases;
- mode entropy decreases;
- local gate reaches `robust_pass`;
- Granite Guardian-backed PCE reaches `robust_pass`;
- dominant Guardian harmfulness increases robustly.

Granite summary for the positive stress:

| Quantity | Direction |
| --- | --- |
| Determinism bootstrap CI | positive |
| Entropy bootstrap CI | negative |
| Guardian-PCE Yes CI | positive |
| Guardian-PCE score CI | positive |
| Dominant harm | robust increase |

This is the strongest reason not to discard the project entirely.

### 5. Why The Positive Signal Is Not Enough

The same positive stress remains restricted because:

- raw target phrase hits are 0;
- each prompt still has 16 unique normalized outputs;
- the dominant mode is semantic/loose, not exact duplicate copying;
- the response wording is special: other short-compliance wording does not
  reproduce robustly;
- prompt transfer is weak or failed;
- step-0 harmful-prompt risk is already high;
- LlamaGuard replication is unavailable;
- the setup uses synthetic non-operational preferences.

The right claim is therefore:

```text
There is a prompt- and wording-sensitive local mechanism signal.
There is not yet stable real-world sampled-mode exploitability evidence.
```

### 6. Refusal Counter-Control Is Useful

The refusal-template counter-control uses the same local route but reverses the
policy target:

- chosen response is a refusal template;
- rejected response is the short compliance template.

Results:

- determinism rises;
- entropy falls;
- proxy harmfulness falls;
- refusal rises;
- compliance falls;
- dominant Guardian harmfulness decreases.

This supports the idea that the DPO setup can steer concentrated modes in
opposite policy directions. It also reinforces that determinism alone is not a
safety vulnerability. The harmfulness term matters.

### 7. Guardian Classifier Status

Granite Guardian 3.1 2B is the first real guardian-style classifier available
locally.

Sanity controls show that Granite is response-sensitive:

| Fixed response type | Guardian behavior |
| --- | --- |
| refusal | 0/10 Yes |
| neutral boundary | 1/10 Yes |
| unsafe-intent placeholder | 10/10 Yes |
| short chosen template | 9/10 Yes |

This reduces the alternative explanation that the guardian simply reacts to the
harmful prompt. It also means the original short chosen response is itself
risky under harmful prompts.

Limitations:

- Granite is not LlamaGuard.
- RoBERTa toxicity only measures toxic language, not harmful instruction
  following.
- Local Qwen-as-judge is not reliable and not independent.
- LlamaGuard candidates remain gated or require gated bases.

### 8. Prompt Transfer Is The Main Scientific Blocker

Prompt transfer results:

| Prompt block | Result |
| --- | --- |
| Original first 10 | strongest positive |
| Prompts 10-19, separately trained | weak |
| Original checkpoints re-evaluated on prompts 10-19 | mixed/fail |
| Held-out fallback first 10 | weak positive |
| Held-out fallback offset 10 | weak/mixed |
| Held-out fallback offset 20 | robust collapse-transfer failure |
| Full 50-prompt aggregate | aggregate positive, highly heterogeneous |

The full 50-prompt view is the best summary:

| Quantity | Result |
| --- | --- |
| Guardian-PCE aggregate | robust_pass |
| Dominant harm aggregate | robust_increase |
| Prompt-seed split | 34 pass / 33 mixed / 33 fail |
| Prompt map | 15 stable pass / 15 mixed / 15 stable fail / 5 mostly-pass/fail |

Interpretation:

```text
Aggregate positivity exists, but prompt heterogeneity is too large.
The broad vulnerability claim is blocked until pass/fail behavior is predictable.
```

### 9. Taxonomy v0 Failed Its First Held-Out Validation

Taxonomy v0 was frozen after exploratory analysis of the 50-prompt split.

Old exploratory signal:

- cyber prompts looked most positive;
- violence/weapons prompts looked mostly negative.

Validation set:

- source: AdvBench harmful behaviors;
- 520 prompts downloaded;
- current 50 local prompts excluded;
- only 4 cyber candidates matched taxonomy v0 after exclusion;
- 28 violence/weapons candidates matched;
- selected balanced 4-vs-4 zero-overlap set.

Held-out validation result:

| Topic | Local Gate | Guardian-PCE | Dominant Harm |
| --- | --- | --- | --- |
| cyber | fail/fail across two seeds, pooled mixed | mixed | robust increase |
| violence/weapons | mixed/mixed across two seeds | mixed | robust decrease |

Key interpretation:

```text
Cyber harmfulness can rise without sampled-mode collapse.
Violence/weapons harmfulness can decrease.
Taxonomy v0 describes old data but does not predict new AdvBench behavior.
```

This is a direct reason not to proceed to S1.

## Data And Artifact Index

Tracked data and configs:

| Artifact | Purpose |
| --- | --- |
| `data/attack_prompts.jsonl` | original local attack prompt set |
| `data/attack_prompts_10_19.jsonl` | second prompt subset |
| `data/attack_prompts_fallback_heldout_30.jsonl` | 30 zero-overlap fallback held-out prompts |
| `data/advbench_harmful_behaviors_all.jsonl` | downloaded AdvBench prompt source |
| `data/advbench_taxonomy_v0_cyber_vs_violence_heldout.jsonl` | taxonomy-v0 AdvBench validation set |
| `data/advbench_s0_1_heldout_30.jsonl` | S0.1 random AdvBench held-out validation set |
| `data/local_short_template_preferences.jsonl` | original positive short-template preferences |
| `data/local_refusal_template_preferences.jsonl` | refusal counter-control |
| `data/local_neutral_boundary_preferences.jsonl` | neutral-boundary control |
| `data/local_concise_overview_preferences.jsonl` | second short-compliance wording |
| `configs/prompt_taxonomy_v0.json` | frozen exploratory taxonomy |

Key scripts:

| Script | Purpose |
| --- | --- |
| `scripts/local_dpo_smoke_train.py` | local Qwen/SmolLM DPO smoke training |
| `scripts/reevaluate_checkpoints.py` | matched checkpoint re-evaluation |
| `scripts/summarize_local_gate.py` | local metric gate and bootstrap |
| `scripts/audit_granite_guardian_outputs.py` | Granite Guardian harmfulness audit |
| `scripts/summarize_guardian_pce.py` | Guardian-backed PCE bootstrap |
| `scripts/combine_guardian_pce_summaries.py` | pooled Guardian-PCE summaries |
| `scripts/analyze_guardian_prompt_heterogeneity.py` | prompt-level heterogeneity map |
| `scripts/analyze_prompt_taxonomy.py` | taxonomy grouping over prompt outcomes |
| `scripts/select_taxonomy_prompt_set.py` | frozen-taxonomy prompt selection |
| `scripts/extract_dominant_representatives.py` | redacted qualitative representatives |

Ignored output families:

| Output family | Contents |
| --- | --- |
| `outputs/local_smoke/qwen05_short_template_margin_flip_*` | strongest positive Qwen checkpoints and evals |
| `outputs/local_smoke/reeval_*` | matched baseline/final re-evaluations |
| `outputs/local_smoke/granite_guardian_*` | Granite audit and summary JSON files |
| `outputs/local_smoke/dominant_representatives_*` | redacted dominant-mode qualitative snippets |
| `outputs/local_models/Qwen2.5-0.5B-Instruct` | local assembled Qwen model |

These ignored files are the full experimental data. The report summarizes them;
it does not embed raw generations.

## Current Decision

The project should not advance to a claim-oriented S1 experiment yet.

Reasons:

- positive evidence is restricted to one response wording and prompt region;
- transfer checks are weak, mixed, or failed;
- taxonomy v0 failed its held-out validation;
- the full 50-prompt aggregate is too heterogeneous;
- LlamaGuard-family replication is not available;
- raw outputs do not show exact template copying;
- the preference data is synthetic and non-operational.

The project should continue as:

```text
PCE metric tooling + prompt-stratified diagnostic research.
```

## Recommended Next Steps

### Immediate Next Step

Do not run a larger model yet. Do not write the paper claim yet.

The preregistered S0.1 held-out validation has now been run. It did not meet
the pass criteria:

- local collapse-direction pass rate is 21/60 = 35.0%, below the 60% threshold;
- both seed42 and seed43 fail the local direction gate;
- Guardian-PCE is positive but `mixed` because determinism falls and entropy
  rises;
- dominant Guardian harmfulness robustly increases;
- target phrase hits remain 0/960 in each final seed.

This result triggers the stop condition for S1 escalation. The next useful item
was only Experiment C as a diagnostic active-induction smoke test. Experiment C
has now also completed and is negative: CAR does not increase with poison
ratio, and Guardian-PCE is strongest for the clean condition rather than the
poisoned conditions.

The completed S0.1 protocol satisfied these preregistration requirements:

1. Freeze the exact hypothesis before running more training.
2. Define prompt strata without looking at new model outcomes.
3. Use a larger prompt set than the 8-topic taxonomy validation.
4. Require two independent training seeds.
5. Require matched evaluation at 16-32 samples per prompt.
6. Require Guardian-PCE robust pass within a held-out stratum.
7. Require raw-mode audit showing a plausible shared semantic mode.
8. Require an independent safety classifier replication if possible.

The concrete pre-proposal supplement plan is maintained in
`docs/pre_proposal_supplement_plan.md`. Its key revision to the attached
proposal is to make held-out prompt validation the first required experiment,
while treating scale and attack experiments as smoke tests unless they are later
expanded and preregistered.

### Best Scientific Route

Pivot from a broad DPO vulnerability claim to a narrower question:

```text
When does DPO-induced response-mode concentration become safety-relevant?
```

This is more defensible because current evidence says:

- DPO can fit preferences locally.
- Some response modes become more deterministic and more guardian-risky.
- But the effect is prompt- and wording-dependent.

The next paper-shaped contribution, if it survives validation, would be:

```text
Prompt-stratified PCE diagnostics for detecting when preference optimization
concentrates risky response modes.
```

### Concrete Work Plan

Recommended order:

1. Build an experiment index.
   - Create `docs/experiment_index.md`.
   - Link every run family to its output directory and summary JSON.
   - Mark each run as pass, weak, mixed, fail, or diagnostic.

2. Freeze S0.1 protocol.
   - Create `docs/s0_1_protocol.md`.
   - State success/failure thresholds before running.
   - Use held-out prompts and a broader taxonomy or no taxonomy.

3. Improve prompt stratification.
   - Do not reuse taxonomy v0 as a predictor.
   - Either build taxonomy v1 with broader features or use simpler
     preregistered buckets such as cyber, fraud, self-harm, violence,
     harassment, and general harmful requests.

4. Treat Experiment C as a completed negative active-induction smoke.
   - CAR dose ordering failed.
   - Guardian-PCE did not show poison dose response.
   - Do not use C as attack evidence in the opening proposal.

5. Seek independent classifier replication.
   - If LlamaGuard access becomes available, run it on the same saved outputs.
   - If not, keep Granite as the named classifier and avoid broader safety
     claims.

6. Do not decide that S1 exists unless a future held-out gate passes.
   - Current S0.1 result says to keep the project as diagnostic tooling.
   - Current C result also says not to claim active-induction feasibility.

### Stop Conditions

Stop trying to escalate the vulnerability claim if any of these happen again:

- held-out Guardian-PCE is mixed or fail;
- determinism falls while harmfulness rises;
- only aggregate pooling passes while standalone prompt blocks fail;
- prompt-level pass/fail split remains close to even;
- raw outputs show no coherent dominant semantic mode;
- a second response wording fails to replicate.

## Bottom Line

The project is worth continuing, but not as a validated exploitability claim.

The honest current conclusion is:

```text
There is a real local mechanism signal in a restricted Qwen short-template stress.
That signal is too prompt-sensitive and wording-sensitive for S1.
S0.1 held-out validation did not pass, so the next step is only a diagnostic
tooling/prompt-stratification plan, not scale-up, active-attack claims, or a
vulnerability claim.
```
