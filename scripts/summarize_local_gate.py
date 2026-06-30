"""
Summarize local proxy-PCE gate runs.

The script expects run directories containing step0.json and final.json files
produced by scripts/local_dpo_smoke_train.py. It prints a compact table and an
aggregate pass/mixed/fail judgement.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptDelta:
    determinism: float
    mode_entropy: float
    proxy_pce: float


@dataclass
class RunSummary:
    name: str
    step0_det: float
    final_det: float
    step0_entropy: float
    final_entropy: float
    step0_pce: float
    final_pce: float
    prompt_pass: int
    prompt_mixed: int
    prompt_fail: int
    prompt_deltas: list[PromptDelta]

    @property
    def delta_det(self) -> float:
        return self.final_det - self.step0_det

    @property
    def delta_entropy(self) -> float:
        return self.final_entropy - self.step0_entropy

    @property
    def delta_pce(self) -> float:
        return self.final_pce - self.step0_pce

    @property
    def judgement(self) -> str:
        det_ok = self.delta_det > 0
        entropy_ok = self.delta_entropy < 0
        pce_ok = self.delta_pce >= 0
        if det_ok and entropy_ok and pce_ok:
            return "pass"
        if sum([det_ok, entropy_ok, pce_ok]) >= 2:
            return "mixed"
        return "fail"

    @property
    def prompt_total(self) -> int:
        return self.prompt_pass + self.prompt_mixed + self.prompt_fail


def load_metric(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prompt_judgement(step0_prompt: dict, final_prompt: dict) -> str:
    det_ok = float(final_prompt["determinism"]) - float(step0_prompt["determinism"]) > 0
    entropy_ok = float(final_prompt["mode_entropy"]) - float(step0_prompt["mode_entropy"]) < 0
    pce_ok = float(final_prompt["proxy_pce"]) - float(step0_prompt["proxy_pce"]) >= 0
    if det_ok and entropy_ok and pce_ok:
        return "pass"
    if sum([det_ok, entropy_ok, pce_ok]) >= 2:
        return "mixed"
    return "fail"


def summarize_prompt_directions(step0: dict, final: dict) -> tuple[int, int, int, list[PromptDelta]]:
    step0_prompts = {entry["prompt"]: entry for entry in step0.get("prompt_metrics", [])}
    final_prompts = {entry["prompt"]: entry for entry in final.get("prompt_metrics", [])}
    shared_prompts = sorted(set(step0_prompts) & set(final_prompts))

    pass_count = 0
    mixed_count = 0
    fail_count = 0
    prompt_deltas: list[PromptDelta] = []
    for prompt in shared_prompts:
        step0_prompt = step0_prompts[prompt]
        final_prompt = final_prompts[prompt]
        prompt_deltas.append(
            PromptDelta(
                determinism=float(final_prompt["determinism"]) - float(step0_prompt["determinism"]),
                mode_entropy=float(final_prompt["mode_entropy"]) - float(step0_prompt["mode_entropy"]),
                proxy_pce=float(final_prompt["proxy_pce"]) - float(step0_prompt["proxy_pce"]),
            )
        )
        judgement = prompt_judgement(step0_prompt, final_prompt)
        if judgement == "pass":
            pass_count += 1
        elif judgement == "mixed":
            mixed_count += 1
        else:
            fail_count += 1
    return pass_count, mixed_count, fail_count, prompt_deltas


def summarize_run(run_dir: Path) -> RunSummary:
    step0 = load_metric(run_dir / "step0.json")
    final = load_metric(run_dir / "final.json")
    prompt_pass, prompt_mixed, prompt_fail, prompt_deltas = summarize_prompt_directions(step0, final)
    return RunSummary(
        name=run_dir.name,
        step0_det=float(step0["mean_determinism"]),
        final_det=float(final["mean_determinism"]),
        step0_entropy=float(step0["mean_mode_entropy"]),
        final_entropy=float(final["mean_mode_entropy"]),
        step0_pce=float(step0["mean_proxy_pce"]),
        final_pce=float(final["mean_proxy_pce"]),
        prompt_pass=prompt_pass,
        prompt_mixed=prompt_mixed,
        prompt_fail=prompt_fail,
        prompt_deltas=prompt_deltas,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    if samples <= 0:
        value = mean(values)
        return value, value, value

    rng = random.Random(seed)
    sample_count = len(values)
    bootstrap_means = []
    for _ in range(samples):
        bootstrap_means.append(mean([values[rng.randrange(sample_count)] for _ in range(sample_count)]))
    bootstrap_means.sort()
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return mean(values), bootstrap_means[lower_index], bootstrap_means[upper_index]


def print_bootstrap_summary(summaries: list[RunSummary], samples: int, seed: int) -> None:
    deltas = [delta for summary in summaries for delta in summary.prompt_deltas]
    intervals = []
    fields = [
        ("det_delta", [delta.determinism for delta in deltas]),
        ("entropy_delta", [delta.mode_entropy for delta in deltas]),
        ("pce_delta", [delta.proxy_pce for delta in deltas]),
    ]
    print(f"\nprompt bootstrap: pooled_n={len(deltas)}, samples={samples}, ci=95%")
    for offset, (name, values) in enumerate(fields):
        center, lower, upper = bootstrap_mean_ci(values, samples=samples, seed=seed + offset)
        intervals.append((name, center, lower, upper))
        print(f"{name}\tmean={center:+.4f}\tci=[{lower:+.4f}, {upper:+.4f}]")

    interval_by_name = {name: (center, lower, upper) for name, center, lower, upper in intervals}
    det_center, det_lower, det_upper = interval_by_name["det_delta"]
    entropy_center, entropy_lower, entropy_upper = interval_by_name["entropy_delta"]
    pce_center, pce_lower, pce_upper = interval_by_name["pce_delta"]

    robust_pass = det_lower > 0 and entropy_upper < 0 and pce_lower >= 0
    robust_fail = det_upper <= 0 or entropy_lower >= 0 or pce_upper < 0
    mean_direction_pass = det_center > 0 and entropy_center < 0 and pce_center >= 0

    if robust_pass:
        decision = "robust_pass"
    elif robust_fail:
        decision = "robust_fail"
    elif mean_direction_pass:
        decision = "weak_pass"
    else:
        decision = "mixed"
    print(f"robust_gate_decision: {decision}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local gate runs.")
    parser.add_argument("run_dirs", nargs="+", help="Directories containing step0.json and final.json")
    parser.add_argument("--bootstrap_samples", type=int, default=0)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    args = parser.parse_args()

    summaries = [summarize_run(Path(path)) for path in args.run_dirs]
    print("run\tdet_delta\tentropy_delta\tpce_delta\tjudgement\tprompt_pass/mixed/fail")
    for summary in summaries:
        print(
            f"{summary.name}\t{summary.delta_det:+.4f}\t"
            f"{summary.delta_entropy:+.4f}\t{summary.delta_pce:+.4f}\t"
            f"{summary.judgement}\t"
            f"{summary.prompt_pass}/{summary.prompt_mixed}/{summary.prompt_fail}"
        )

    pass_count = sum(1 for summary in summaries if summary.judgement == "pass")
    mixed_count = sum(1 for summary in summaries if summary.judgement == "mixed")
    fail_count = sum(1 for summary in summaries if summary.judgement == "fail")
    prompt_pass = sum(summary.prompt_pass for summary in summaries)
    prompt_mixed = sum(summary.prompt_mixed for summary in summaries)
    prompt_fail = sum(summary.prompt_fail for summary in summaries)
    prompt_total = sum(summary.prompt_total for summary in summaries)
    print(f"\naggregate: pass={pass_count}, mixed={mixed_count}, fail={fail_count}, total={len(summaries)}")
    print(
        "prompt aggregate: "
        f"pass={prompt_pass}, mixed={prompt_mixed}, fail={prompt_fail}, total={prompt_total}"
    )
    if pass_count == len(summaries):
        print("overall: pass")
    elif pass_count > 0 or mixed_count > 0:
        print("overall: mixed")
    else:
        print("overall: fail")

    if args.bootstrap_samples:
        print_bootstrap_summary(summaries, samples=args.bootstrap_samples, seed=args.bootstrap_seed)


if __name__ == "__main__":
    main()
