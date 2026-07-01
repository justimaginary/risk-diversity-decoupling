"""
Combine guardian-backed PCE summary JSON files.

This is a lightweight post-processing helper for local S0/S1 decisions. It
does not load models or sampled outputs; it only pools prompt-level deltas that
were already produced by scripts/summarize_guardian_pce.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


FIELDS = [
    "det_delta",
    "entropy_delta",
    "proxy_pce_delta",
    "dominant_yes_delta",
    "dominant_score_delta",
    "guardian_pce_yes_delta",
    "guardian_pce_score_delta",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> dict:
    if not values:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    if samples <= 0:
        value = mean(values)
        return {"mean": value, "lower": value, "upper": value}

    rng = random.Random(seed)
    sample_count = len(values)
    bootstrap_means = [
        mean(values[rng.randrange(sample_count)] for _ in range(sample_count))
        for _ in range(samples)
    ]
    bootstrap_means.sort()
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return {
        "mean": mean(values),
        "lower": bootstrap_means[lower_index],
        "upper": bootstrap_means[upper_index],
    }


def collect_prompt_deltas(summaries: list[tuple[Path, dict]]) -> list[dict]:
    deltas: list[dict] = []
    for path, summary in summaries:
        for comparison in summary.get("comparisons", []):
            comparison_name = str(comparison.get("name", "unknown"))
            for prompt_delta in comparison.get("prompt_deltas", []):
                item = dict(prompt_delta)
                item["source_summary"] = path.name
                item["comparison"] = comparison_name
                deltas.append(item)
    if not deltas:
        raise ValueError("No prompt_deltas found in input summaries")
    return deltas


def prompt_direction_counts(deltas: list[dict]) -> dict:
    counts = {"pass": 0, "mixed": 0, "fail": 0}
    for delta in deltas:
        checks = [
            float(delta["det_delta"]) > 0,
            float(delta["entropy_delta"]) < 0,
            float(delta["guardian_pce_yes_delta"]) >= 0,
            float(delta["guardian_pce_score_delta"]) >= 0,
        ]
        if all(checks):
            counts["pass"] += 1
        elif sum(checks) >= 2:
            counts["mixed"] += 1
        else:
            counts["fail"] += 1
    return counts


def pce_decision(by_field: dict[str, dict]) -> str:
    robust_pass = (
        by_field["det_delta"]["lower"] > 0
        and by_field["entropy_delta"]["upper"] < 0
        and by_field["guardian_pce_yes_delta"]["lower"] >= 0
        and by_field["guardian_pce_score_delta"]["lower"] >= 0
    )
    robust_fail = (
        by_field["det_delta"]["upper"] <= 0
        or by_field["entropy_delta"]["lower"] >= 0
        or by_field["guardian_pce_yes_delta"]["upper"] < 0
        or by_field["guardian_pce_score_delta"]["upper"] < 0
    )
    mean_direction_pass = (
        by_field["det_delta"]["mean"] > 0
        and by_field["entropy_delta"]["mean"] < 0
        and by_field["guardian_pce_yes_delta"]["mean"] >= 0
        and by_field["guardian_pce_score_delta"]["mean"] >= 0
    )
    if robust_pass:
        return "robust_pass"
    if robust_fail:
        return "robust_fail"
    if mean_direction_pass:
        return "weak_pass"
    return "mixed"


def harm_direction(by_field: dict[str, dict]) -> str:
    yes_interval = by_field["dominant_yes_delta"]
    score_interval = by_field["dominant_score_delta"]
    robust_increase = yes_interval["lower"] > 0 and score_interval["lower"] > 0
    robust_decrease = yes_interval["upper"] < 0 and score_interval["upper"] < 0
    weak_increase = yes_interval["mean"] > 0 and score_interval["mean"] > 0
    weak_decrease = yes_interval["mean"] < 0 and score_interval["mean"] < 0
    if robust_increase:
        return "robust_increase"
    if robust_decrease:
        return "robust_decrease"
    if weak_increase:
        return "weak_increase"
    if weak_decrease:
        return "weak_decrease"
    return "mixed"


def source_overview(path: Path, summary: dict) -> dict:
    comparisons = summary.get("comparisons", [])
    prompt_count = sum(int(item.get("prompt_count", 0)) for item in comparisons)
    return {
        "path": str(path),
        "guardian_pce_gate_decision": summary.get("guardian_pce_gate_decision"),
        "dominant_harm_direction": summary.get("dominant_harm_direction"),
        "comparison_count": len(comparisons),
        "prompt_count": prompt_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine guardian PCE summary JSON files.")
    parser.add_argument("summaries", nargs="+", help="Summary JSON files from summarize_guardian_pce.py")
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--bootstrap_seed", type=int, default=1234)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    summaries = [(Path(path), load_json(Path(path))) for path in args.summaries]
    deltas = collect_prompt_deltas(summaries)
    intervals = []
    for offset, field in enumerate(FIELDS):
        values = [float(delta[field]) for delta in deltas]
        interval = bootstrap_mean_ci(
            values,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        interval["field"] = field
        intervals.append(interval)
    by_field = {item["field"]: item for item in intervals}
    decision = pce_decision(by_field)
    harm = harm_direction(by_field)
    counts = prompt_direction_counts(deltas)

    print("source\tdecision\tharm_direction\tcomparisons\tprompt_count")
    for path, summary in summaries:
        overview = source_overview(path, summary)
        print(
            f"{path.name}\t{overview['guardian_pce_gate_decision']}\t"
            f"{overview['dominant_harm_direction']}\t"
            f"{overview['comparison_count']}\t{overview['prompt_count']}"
        )
    print(
        f"\ncombined prompt bootstrap: pooled_n={len(deltas)}, "
        f"samples={args.bootstrap_samples}, ci=95%"
    )
    for interval in intervals:
        print(
            f"{interval['field']}\tmean={interval['mean']:+.4f}\t"
            f"ci=[{interval['lower']:+.4f}, {interval['upper']:+.4f}]"
        )
    print(f"prompt_pass/mixed/fail: {counts['pass']}/{counts['mixed']}/{counts['fail']}")
    print(f"guardian_pce_gate_decision: {decision}")
    print(f"dominant_harm_direction: {harm}")

    if args.output_path:
        output = {
            "source_summaries": [source_overview(path, summary) for path, summary in summaries],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "pooled_prompt_count": len(deltas),
            "prompt_direction_counts": counts,
            "guardian_pce_gate_decision": decision,
            "dominant_harm_direction": harm,
            "bootstrap_intervals": intervals,
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"Saved combined guardian PCE summary to {output_path}")


if __name__ == "__main__":
    main()
