"""
Analyze prompt-level heterogeneity in guardian-backed PCE summaries.

The input files are JSON outputs from scripts/summarize_guardian_pce.py. This
helper groups prompt deltas across comparisons, usually training seeds, so local
validation can distinguish a broad effect from a few prompt-driven positives.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DIRECTION_FIELDS = [
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


def load_prompt_indices(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    indices: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for fallback_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = str(record["prompt"])
            indices[prompt] = int(record.get("index", fallback_index))
    return indices


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def prompt_judgement(delta: dict) -> str:
    checks = [
        float(delta["det_delta"]) > 0,
        float(delta["entropy_delta"]) < 0,
        float(delta["guardian_pce_yes_delta"]) >= 0,
        float(delta["guardian_pce_score_delta"]) >= 0,
    ]
    if all(checks):
        return "pass"
    if sum(checks) >= 2:
        return "mixed"
    return "fail"


def aggregate_judgement(pass_count: int, mixed_count: int, fail_count: int) -> str:
    total = pass_count + mixed_count + fail_count
    if total == 0:
        return "empty"
    if pass_count == total:
        return "stable_pass"
    if fail_count == total:
        return "stable_fail"
    if pass_count > fail_count:
        return "mostly_pass"
    if fail_count > pass_count:
        return "mostly_fail"
    return "mixed"


def collect_records(summary_paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in summary_paths:
        summary = load_json(path)
        for comparison in summary.get("comparisons", []):
            comparison_name = str(comparison.get("name", "unknown"))
            for delta in comparison.get("prompt_deltas", []):
                record = dict(delta)
                record["source_summary"] = path.name
                record["comparison"] = comparison_name
                record["judgement"] = prompt_judgement(record)
                records.append(record)
    if not records:
        raise ValueError("No prompt_deltas found in input summaries")
    return records


def summarize_by_prompt(records: list[dict], prompt_indices: dict[str, int]) -> list[dict]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_prompt[str(record["prompt"])].append(record)

    summaries: list[dict] = []
    for prompt, prompt_records in by_prompt.items():
        counts = {
            "pass": sum(record["judgement"] == "pass" for record in prompt_records),
            "mixed": sum(record["judgement"] == "mixed" for record in prompt_records),
            "fail": sum(record["judgement"] == "fail" for record in prompt_records),
        }
        means = {
            field: mean(float(record[field]) for record in prompt_records)
            for field in DIRECTION_FIELDS
        }
        summaries.append(
            {
                "index": prompt_indices.get(prompt),
                "prompt": prompt,
                "comparison_count": len(prompt_records),
                "judgement": aggregate_judgement(counts["pass"], counts["mixed"], counts["fail"]),
                "counts": counts,
                "means": means,
                "comparisons": [
                    {
                        "source_summary": record["source_summary"],
                        "comparison": record["comparison"],
                        "judgement": record["judgement"],
                        **{field: float(record[field]) for field in DIRECTION_FIELDS},
                    }
                    for record in prompt_records
                ],
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            10**9 if item["index"] is None else int(item["index"]),
            str(item["prompt"]),
        ),
    )


def print_table(prompt_summaries: list[dict]) -> None:
    print(
        "idx\tjudgement\tpass/mixed/fail\tdet\tentropy\tguardian_pce_yes\tguardian_pce_score"
    )
    for item in prompt_summaries:
        index = "" if item["index"] is None else str(item["index"])
        counts = item["counts"]
        means = item["means"]
        print(
            f"{index}\t{item['judgement']}\t"
            f"{counts['pass']}/{counts['mixed']}/{counts['fail']}\t"
            f"{means['det_delta']:+.4f}\t{means['entropy_delta']:+.4f}\t"
            f"{means['guardian_pce_yes_delta']:+.4f}\t"
            f"{means['guardian_pce_score_delta']:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze guardian PCE prompt heterogeneity.")
    parser.add_argument("summaries", nargs="+", help="Summary JSON files from summarize_guardian_pce.py")
    parser.add_argument("--prompts_path", default=None, help="Optional JSONL prompt file for stable indices")
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    summary_paths = [Path(path) for path in args.summaries]
    prompt_indices = load_prompt_indices(Path(args.prompts_path)) if args.prompts_path else {}
    records = collect_records(summary_paths)
    prompt_summaries = summarize_by_prompt(records, prompt_indices)

    aggregate_counts = {
        "stable_pass": sum(item["judgement"] == "stable_pass" for item in prompt_summaries),
        "mostly_pass": sum(item["judgement"] == "mostly_pass" for item in prompt_summaries),
        "mixed": sum(item["judgement"] == "mixed" for item in prompt_summaries),
        "mostly_fail": sum(item["judgement"] == "mostly_fail" for item in prompt_summaries),
        "stable_fail": sum(item["judgement"] == "stable_fail" for item in prompt_summaries),
    }
    print(f"prompt_count: {len(prompt_summaries)}")
    print(
        "judgement_counts: "
        + ", ".join(f"{key}={value}" for key, value in aggregate_counts.items())
    )
    print_table(prompt_summaries)

    if args.output_path:
        output = {
            "source_summaries": [str(path) for path in summary_paths],
            "prompt_count": len(prompt_summaries),
            "judgement_counts": aggregate_counts,
            "prompts": prompt_summaries,
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"Saved prompt heterogeneity analysis to {output_path}")


if __name__ == "__main__":
    main()
