"""
Analyze whether preference-margin changes transmit to generation metrics.

This joins a preference-margin diagnostic JSON with a matched re-evaluation
directory containing step0.json and final.json. It reports prompt-level
generation deltas and simple correlations against summed and per-token-average
preference margins.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rank_values(values: list[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    current = 0
    while current < len(sorted_pairs):
        end = current + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[current][0]:
            end += 1
        avg_rank = (current + 1 + end) / 2.0
        for _, original_index in sorted_pairs[current:end]:
            ranks[original_index] = avg_rank
        current = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    x_norm = math.sqrt(sum(value * value for value in x_centered))
    y_norm = math.sqrt(sum(value * value for value in y_centered))
    if x_norm == 0 or y_norm == 0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / (x_norm * y_norm)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(rank_values(xs), rank_values(ys))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze margin-to-generation transmission.")
    parser.add_argument("--margin_path", required=True)
    parser.add_argument("--reeval_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    margin_report = load_json(Path(args.margin_path))
    step0 = load_json(Path(args.reeval_dir) / "step0.json")
    final = load_json(Path(args.reeval_dir) / "final.json")

    margin_by_prompt = {record["prompt"]: record for record in margin_report["records"]}
    step0_by_prompt = {record["prompt"]: record for record in step0["prompt_metrics"]}
    final_by_prompt = {record["prompt"]: record for record in final["prompt_metrics"]}
    shared_prompts = sorted(set(margin_by_prompt) & set(step0_by_prompt) & set(final_by_prompt))
    if not shared_prompts:
        raise ValueError("No shared prompts found across margin and re-evaluation reports")

    records: list[dict] = []
    for prompt in shared_prompts:
        margin = margin_by_prompt[prompt]
        before = step0_by_prompt[prompt]
        after = final_by_prompt[prompt]
        det_delta = float(after["determinism"]) - float(before["determinism"])
        entropy_delta = float(after["mode_entropy"]) - float(before["mode_entropy"])
        pce_delta = float(after["proxy_pce"]) - float(before["proxy_pce"])
        cluster_delta = float(after["num_clusters"]) - float(before["num_clusters"])
        collapse_pass = det_delta > 0 and entropy_delta < 0 and pce_delta >= 0
        records.append(
            {
                "prompt": prompt,
                "sum_margin_delta": float(margin["margin_delta"]),
                "final_sum_margin": float(margin["final_margin"]),
                "avg_margin_delta": float(margin["avg_margin_delta"]),
                "final_avg_margin": float(margin["final_avg_margin"]),
                "det_delta": det_delta,
                "entropy_delta": entropy_delta,
                "pce_delta": pce_delta,
                "cluster_delta": cluster_delta,
                "collapse_pass": collapse_pass,
            }
        )

    predictors = ["sum_margin_delta", "final_sum_margin", "avg_margin_delta", "final_avg_margin"]
    outcomes = ["det_delta", "entropy_delta", "pce_delta", "cluster_delta"]
    correlations: dict[str, dict[str, dict[str, float | None]]] = {}
    for predictor in predictors:
        correlations[predictor] = {}
        predictor_values = [record[predictor] for record in records]
        for outcome in outcomes:
            outcome_values = [record[outcome] for record in records]
            correlations[predictor][outcome] = {
                "pearson": pearson(predictor_values, outcome_values),
                "spearman": spearman(predictor_values, outcome_values),
            }

    positive_avg_margin_records = [record for record in records if record["final_avg_margin"] > 0]
    summary = {
        "num_shared_prompts": len(records),
        "collapse_pass_count": sum(1 for record in records if record["collapse_pass"]),
        "positive_final_avg_margin_count": len(positive_avg_margin_records),
        "positive_final_avg_margin_collapse_pass_count": sum(
            1 for record in positive_avg_margin_records if record["collapse_pass"]
        ),
        "det_delta": summarize([record["det_delta"] for record in records]),
        "entropy_delta": summarize([record["entropy_delta"] for record in records]),
        "pce_delta": summarize([record["pce_delta"] for record in records]),
        "cluster_delta": summarize([record["cluster_delta"] for record in records]),
    }

    output = {
        "margin_path": args.margin_path,
        "reeval_dir": args.reeval_dir,
        "summary": summary,
        "correlations": correlations,
        "records": records,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("summary")
    print(f"shared_prompts\t{summary['num_shared_prompts']}")
    print(f"collapse_pass\t{summary['collapse_pass_count']}/{summary['num_shared_prompts']}")
    print(
        "positive_avg_margin_collapse_pass\t"
        f"{summary['positive_final_avg_margin_collapse_pass_count']}/"
        f"{summary['positive_final_avg_margin_count']}"
    )
    print("\ncorrelation\tpearson\tspearman")
    for predictor in predictors:
        for outcome in outcomes:
            values = correlations[predictor][outcome]
            print(f"{predictor}->{outcome}\t{fmt(values['pearson'])}\t{fmt(values['spearman'])}")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
