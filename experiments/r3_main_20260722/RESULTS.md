# R3 Qwen3-1.7B multi-seed main result

## Status and decision

The computational R3 package is complete. D1 and D2 were evaluated at the frozen 100-step checkpoint with three training seeds; D4 was evaluated separately at the KL-safe step-30 rescue checkpoint. The preregistered D2 key checkpoint was also repeated at 64 generations per prompt.

**Gate R3: Stop.** None of the promoted conditions passes the computational gate:

- D1 reduces risk consistently, but harmful entropy also falls beyond the non-inferiority margin.
- D2 has a positive mean on some endpoints, but the direction reverses across training seeds and safety judges; the hierarchical-bootstrap risk CI crosses zero.
- D4 consistently reduces both risk and harmful-mode diversity while increasing safe-prompt refusal.

The required blinded human-audit packets are prepared server-side but are not yet annotated. Human review remains required before publishing qualitative claims, but it cannot make the R3 gate pass because multiple necessary computational criteria already fail. Per the stop rule, do not start R4 or rent 48/80GB hardware on the current evidence.

## Frozen protocol

- Model: Qwen3-1.7B with LoRA DPO on RTX 4090 24GB.
- Main conditions: D1 clean safety and repaired D2 helpfulness-safety conflict.
- Training seeds: 42, 43, 44; 100 steps; checkpoints at steps 50 and 100.
- Evaluation: 100 HarmBench prompts x 32 samples, full 450-prompt XSTest.
- Key-checkpoint confirmation: base and all D2 seeds x 64 samples.
- Judges/metrics: HarmBench Mistral, Granite Guardian, StrongREJECT, semantic clustering, harmful/risk-mass entropy, and 5,000-sample hierarchical bootstrap.
- All main-run KL and output-quality gates passed.

## D1 clean safety

All three safety judges agree that risk decreases across all three seeds. Relative to base, the three-seed mean HarmBench harm rate changes by -0.0885 and harmful semantic entropy by -0.1442.

The bootstrap confirms both decreases:

- harm-rate delta: -0.0885, 95% CI [-0.1440, -0.0456];
- harmful-entropy delta: -0.1136, 95% CI [-0.1802, -0.0540].

D1 therefore fails the risk-structure non-inferiority gate: it is a coupled risk-and-harmful-diversity reduction, not risk-diversity decoupling.

## D2 repaired conflict

At 32 samples, the three-seed mean HarmBench harm-rate delta is +0.0381, but the individual seed harm rates are 0.2513, 0.3391, and 0.0238 versus base 0.1666. HarmBench, Granite, and StrongREJECT all fail the seed-direction consistency criterion.

The 32-sample hierarchical bootstrap gives:

- harm-rate delta: +0.0383, 95% CI [-0.1353, 0.1725];
- harmful-entropy delta: -0.0074, 95% CI [-0.1626, 0.1257].

The [64-sample confirmation](../r3_d2_64_20260722/RESULTS.md) reproduces the same seed ordering and again fails the gate. The instability is therefore attributable to training-seed heterogeneity, not insufficient generation samples.

## D4 sensitivity analysis

The original D4 100-step and step-50 protocols failed the KL gate, so their outputs were not used. The separate [step-30 sensitivity analysis](../r3_d4_step30_20260722/RESULTS.md) passes KL for all seeds and consistently lowers risk, harmful entropy, and unsafe-answer rates while increasing XSTest over-refusal. Its computational gate also fails because harmful entropy is not non-inferior.

## Reproducibility and failures

- Paid runs record Git commit, data hashes, environment, wall time, VRAM, tokens, KL, checkpoints, generation metadata, and quality gates in the committed summaries/manifests.
- A server shutdown interrupted the initial HarmBench judge. The runner incorrectly treated a partial JSON as complete; commit `5b67813` adds status-aware condition-level resume and regression tests. The resumed audit reused completed conditions and finished without changing prior results.
- During the 64-sample supplement, a temporary parallel seed44 worker overlapped with the main runner on the same deterministic output shard. The extra worker was stopped immediately; the surviving shard was verified at 84 unique prompts / 2,688 answers with matching manifest state, then resumed to 100 prompts / 6,400 answers. Final IDs, per-prompt counts, hashes, quality gates, and merged manifests all passed.
- Raw generations, model weights, large judge records, logs, and human-audit answer keys remain server-only.

Machine-readable evidence is in `metrics/r3_summary.json` and `bootstrap/hierarchical_bootstrap.json`.
