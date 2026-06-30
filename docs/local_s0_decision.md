# Local S0 Decision Memo

This memo summarizes the current RTX 4060 local validation state. It is a
decision aid, not a paper claim.

## Current Decision

Do not escalate the cached local SmolLM2 route or the restored local Qwen 0.5B
route to S1.

The local evidence supports continued measurement/tooling work and possibly a
different small-model gate. It does not support the claim that DPO reliably
induces exploitable sampled-mode collapse in a real instruction model.

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
outputs.

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

1. Redesign the local S0 protocol if continuing. A clearer next criterion is to
   identify conditions where positive length-normalized preference margins
   actually transmit to sampled output-mode collapse and target-template
   sampling.
2. Treat `weak_pass` or `mixed` as insufficient for S1; require `robust_pass`.
3. If a redesigned <=500M gate cannot connect positive margins to robust
   sampled collapse, pivot to PCE diagnostic tooling rather than a DPO
   vulnerability claim.

Fallback:

Keep the project as PCE measurement tooling and diagnostics until a real small
instruction-model gate passes robustly.
