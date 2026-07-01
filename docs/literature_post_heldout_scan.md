# Post-Heldout Literature Scan

Date: 2026-07-01

This scan updates `docs/literature_post_guardian_scan.md` after the full
zero-overlap held-out fallback transfer check. It is a validation-route boundary,
not a paper claim.

## Local Trigger

The strongest local short-template result has now been tested beyond the
original 20 prompts:

- Original first-10 positive result: Guardian-PCE `robust_pass`.
- Prompts 10-19 transfer: Guardian-PCE `mixed`.
- Held-out fallback prompts 0-9: Guardian-PCE `weak_pass`.
- Held-out fallback prompts 10-19: Guardian-PCE `mixed`.
- Held-out fallback prompts 20-29: local gate and Guardian-PCE `robust_fail`,
  even though dominant Guardian harmfulness rises.
- Full 50-prompt aggregate: Guardian-PCE `robust_pass`, dominant harm
  `robust_increase`, but prompt-seed split is nearly even at 34 pass, 33 mixed,
  and 33 fail.

The result is therefore aggregate-positive but prompt-dependent. It supports a
PCE diagnostic hypothesis, not a stable vulnerability claim.

## Web-Verified Related Work

These sources were re-checked online on 2026-07-01.

| Area | Primary Source | Relevance After Heldout |
| --- | --- | --- |
| DPO objective | Direct Preference Optimization, 2023: https://arxiv.org/abs/2305.18290 | Baseline objective. The project studies a possible downstream measurement failure mode, not a new alignment objective. |
| DPO likelihood pathology | Smaug / DPO-Positive, 2024: https://arxiv.org/abs/2402.13228 | DPO can have likelihood-side pathologies. This overlaps with local margin diagnostics. |
| Direct-alignment over-optimization | Understanding Likelihood Over-optimisation, 2024: https://arxiv.org/abs/2410.11677 | Shows likelihood/margin gains need not generalize and can relate to diversity indicators. This directly matches the local finding that preference fitting does not reliably transmit to sampled collapse. |
| Diversity-preserving preference optimization | Diverse Preference Optimization, 2025: https://arxiv.org/abs/2501.18101 | States that preference optimization can sharpen distributions and reduce diversity. "DPO reduces diversity" is not a defensible novelty claim. |
| Benign DPO safety attack | Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs, 2026: https://arxiv.org/abs/2605.10998 | Very close to any "benign-looking DPO data suppresses refusals and transfers to harmful prompts" story. This sharply narrows novelty. |
| Preference-label poisoning | Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment, 2025: https://arxiv.org/abs/2511.09105 | Preference poisoning is already an active line. PCE poisoning must add dominant-mode exploitability measurement, not only attack success. |
| Guardian classifier | Granite Guardian, 2024: https://arxiv.org/abs/2412.07724 | Supports using Granite Guardian as a local risk classifier, but results should be named as Granite-specific. |
| LlamaGuard replication target | Llama Guard 3-1B-INT4, 2024: https://arxiv.org/abs/2411.17713 | Confirms compact LlamaGuard-style moderation exists, but local access remains gated; Granite is not an equivalent replication. |

## Updated Novelty Boundary

Not defensible as the main contribution:

- DPO reduces diversity.
- DPO can suppress refusals.
- Benign-looking DPO preference data can weaken safety.
- Preference-label poisoning can steer aligned models.
- A safety classifier can mark prompt/response pairs as risky.

Still potentially defensible, but only as a narrower diagnostic contribution:

- PCE as a prompt-level product of sampled dominant-mode determinism and
  guardian-scored dominant-mode harmfulness.
- Evidence that harmfulness alone is not enough: held-out offset20 has higher
  dominant Guardian harmfulness but fails PCE because determinism and entropy
  move in the wrong direction.
- Prompt heterogeneity mapping that explains where aggregate PCE increases come
  from instead of treating aggregate `robust_pass` as a stable transfer result.
- Controls separating risky short-compliance, refusal, neutral-boundary, and
  alternate short-compliance response modes.

## Decision Impact

The full held-out result weakens escalation:

```text
Before heldout:
  Restricted mechanism follow-up was justified by a strong first-10 positive.

After heldout:
  The correct next step is diagnostic, prompt-stratified validation.
  S1 escalation should wait for standalone held-out robust_pass behavior,
  not aggregate robustness driven by heterogeneous prompt subsets.
```

## Next Experiment Boundary

Do not advance to a paper-level vulnerability claim unless a future gate shows
all of the following:

- A standalone held-out prompt subset reaches Guardian-PCE `robust_pass`.
- Prompt-level heterogeneity is reduced or explained by a pre-registered prompt
  taxonomy.
- Determinism rises and entropy falls in the same subset where guardian-scored
  harmfulness rises.
- Raw dominant-mode audit shows semantic concentration stronger than the current
  low dominant masses and exact-copy-free samples.
- A second safety classifier, ideally LlamaGuard if accessible, supports the
  harmfulness direction.

## Current Positioning Sentence

Use this wording for now:

```text
Local evidence supports a diagnostic PCE hypothesis: DPO can raise aggregate
guardian-backed PCE under some response-mode and prompt conditions, but the
effect is highly prompt-dependent and does not yet establish stable
real-world exploitability.
```
