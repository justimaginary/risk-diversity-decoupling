# Local S0 Decision Memo

This memo summarizes the current RTX 4060 local validation state. It is a
decision aid, not a paper claim.

## Current Decision

Do not escalate the cached local SmolLM2 route or the earlier weak Qwen 0.5B
routes to a paper-claim S1. The Qwen short-template margin-flip stress now
justifies a restricted S1 follow-up for mechanism validation.

The local evidence supports continued measurement/tooling work and a narrow S1
follow-up on the short-template mechanism. It still does not support the claim
that DPO reliably induces exploitable sampled-mode collapse in a real
instruction model.

## Evidence Snapshot

| Gate | Scope | Result | Decision |
| --- | --- | --- | --- |
| Synthetic diverse vs collapsed | controlled text | metrics move as expected | pass |
| Toy DPO collapse | categorical toy policy | probability mass concentrates | pass |
| Tiny GPT-2 smoke | real training loop | pipeline works, model too small | pipeline pass |
| SmolLM2-135M LM-head mini gate | 3 prompts x 4 samples | weak two-seed direction | weak S0 only |
| SmolLM2-360M collapse-proxy, original seeds | 10 prompts x 8 samples | old seeds were identical checkpoints | invalid as training-seed evidence |
| SmolLM2-360M corrected collapse-proxy | 10 prompts x 8 samples | two training seeds aggregate pass | weak_pass |
| SmolLM2-360M corrected collapse-proxy | matched 10 prompts x 16 samples | both corrected seeds fail | robust_fail |
| SmolLM2-360M uniform-control | matched 10 prompts x 16 samples | robust reverse direction | robust_fail |
| SmolLM2-135M all-params uniform-control | 10 prompts x 8 samples | DPO loss fits, sampled metrics mixed | mixed |
| Qwen2.5-0.5B-Instruct restored local fp32 LM-head | two seeds, 20 steps, matched 10x16 | both seeds fail; 3/20 prompt comparisons pass | fail |
| Qwen2.5-0.5B-Instruct restored local fp32 LM-head | two seeds, 100 steps, matched 10x16 | both seeds pass; 10/20 prompt comparisons pass; bootstrap crosses zero | weak_pass |
| Qwen2.5-0.5B-Instruct restored local fp32 LM-head | 100-step checkpoints, matched 20x32 | seed-level split 1 pass / 1 fail; 14/40 prompt comparisons pass | weak_pass |
| Qwen2.5-0.5B-Instruct collapse-proxy subset | two seeds, 100 steps, matched 10x16 | seed-level split pass/mixed; 7/20 prompt comparisons pass | weak_pass |
| Qwen2.5-0.5B-Instruct preference-margin diagnostic | four 100-step checkpoints | sum margins stay negative; length-normalized margins flip strongly positive | diagnostic |
| Qwen2.5-0.5B-Instruct margin-to-generation link | four 100-step analyses | positive margins do not reliably predict collapse-direction metric movement | diagnostic |
| Qwen2.5-0.5B-Instruct short-template control | two seeds, 100 steps, matched 10x16 | det/entropy move correctly, PCE decreases, target phrase 0 hits | mixed |
| Qwen2.5-0.5B-Instruct short-template margin-flip stress | seeds 42/43, lr=3e-6, 300 steps, matched 10x16 | margins flip positive; aggregate robust_pass; target phrase 0 hits | restricted S1 follow-up |
| Qwen2.5-0.5B-Instruct short-template raw-mode audit | same matched raw outputs | final dominant mass rises to about 0.34-0.35; max exact duplicate count stays 1; target phrase 0 hits | loose-mode evidence |
| Qwen2.5-0.5B-Instruct short-template policy-proxy audit | same matched raw outputs | final refusal decreases and compliance/actionability increases, strongest in dominant clusters | proxy-only evidence |

## Interpretation

The strongest positive signal is the corrected 360M collapse-proxy 10x8 run,
but bootstrap marks it as `weak_pass` because entropy is not robust. When the
same corrected checkpoints are measured at 10x16, the decision becomes
`robust_fail`.

The uniform-control diagnostic is an intentionally strong non-operational
collapse-control test. It also fails at 10x16, and raw-output audit shows the
trained model does not sample-copy the shared chosen template. The 135M
all-parameters run fits the DPO loss nearly to zero, but sampled outputs still
do not form a stable shared mode.

The restored Qwen 0.5B gate moves the project past the previous download
blocker. At 20 DPO steps, both Qwen seeds fail the matched 10x16 directional
gate. At 100 DPO steps, both seeds pass the 10x16 aggregate directional check,
but the result is still `weak_pass`: prompt comparisons split 10 pass and 10
fail, bootstrap intervals cross zero, and raw-output audit still shows zero
target-template hits. A stronger 20x32 re-evaluation remains `weak_pass`, but
seed-level direction splits into one pass and one fail. A second
non-operational Qwen preference subset repeats the weak pattern rather than
producing robust evidence. Preference-margin diagnostics show why this is not
merely a failed training run: every checked preference margin moves in the
chosen direction. Summed margins remain negative because chosen placeholders are
longer, but length-normalized margins flip strongly positive. The unresolved
question is therefore whether local per-token preference fitting transmits to
sampled-mode collapse. The first transmission analysis is not encouraging:
prompt-level average-margin gains do not reliably predict higher determinism or
lower entropy, and for the collapse-proxy subset the relation is often opposite
the desired collapse direction. A short-template control improves the
transmission direction, but it still does not make the chosen template win under
length-normalized margin scoring and the target phrase never appears in sampled
outputs. A stronger short-template stress run on seed42 does flip both summed
and length-normalized margins positive and yields `robust_pass` at matched
10x16. Seed43 replicates this result: the two-seed aggregate is
`robust_pass`, with determinism and proxy PCE increasing and entropy decreasing.
The first raw-mode audit of those matched outputs supports loose sampled-mode
concentration: final mean dominant-cluster mass is 0.3500 for seed42 and 0.3438
for seed43, compared with 0.1625 at step 0; 8 of 10 final prompts per seed have
dominant mass at least 0.25, and 2 of 10 reach at least 0.5. However, each
prompt still has 16 unique normalized outputs, max exact duplicate count remains
1, target-template hits remain 0, and harmfulness is still a lexical proxy.
A stronger local policy-proxy audit points in the same direction: all-output
refusal falls from 0.1812 to about 0.0625, all-output compliance rises from
0.2437 to about 0.46-0.47, and dominant-cluster compliance rises from 0.2308
to about 0.67-0.68 across both seeds. Dominant-cluster high-risk-instructional
proxy rate rises from 0.1538 to about 0.36-0.38, with 5 of 10 final prompts per
seed crossing a 0.5 prompt-level majority. This strengthens the local proxy
evidence but still does not replace a real safety classifier.
This is the first strong local S0v2 signal, but it still lacks literal
target-template sampling and a real safety classifier.

The current local conclusion is therefore:

```text
DPO preference loss fitting is observable locally.
Stable sampled-mode collapse is not established locally.
Proxy PCE/harmfulness is not validated with a real safety classifier.
```

## S1 Escalation Requirements

Escalate only if a future local gate satisfies all of the following:

- At least two independent training seeds or preference subsets.
- Matched evaluation at 10x16 or stronger.
- `robust_gate_decision: robust_pass`.
- Raw-output audit shows a plausible shared response mode, not just metric
  noise.
- Proxy PCE does not contradict the determinism/entropy direction.
- For any safety claim, replace lexical proxy harmfulness with a real safety
  classifier or clearly label the result as proxy-only.

## Recommended Next Move

Preferred:

1. Continue the restricted S1 follow-up for the Qwen short-template margin-flip
   mechanism with a counter-control and, if feasible on local hardware, a real
   safety classifier; the first raw-mode audit, policy-proxy audit, and
   related-work scan are complete.
2. Treat this as mechanism evidence only until raw shared modes and real
   harmfulness are validated.
3. Treat `weak_pass` or `mixed` as insufficient for any paper claim; require
   multi-seed `robust_pass` plus raw-mode and safety evidence.
4. If restricted S1 fails raw-mode or safety validation, pivot to PCE diagnostic
   tooling rather than a DPO vulnerability claim.

Fallback:

Keep the project as PCE measurement tooling and diagnostics until a real small
instruction-model gate passes robustly.
