"""
Analyze prompt-level PCE outcomes by a deterministic prompt taxonomy.

The taxonomy is intentionally simple and pre-declared in a versioned JSON
config: it uses keyword rules over the prompt text, not model predictions. This
keeps the analysis reproducible for local S0 diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_TAXONOMY_PATH = Path("configs/prompt_taxonomy_v0.json")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_taxonomy(path: Path) -> dict:
    taxonomy = load_json(path)
    required = ["topics", "verbs", "surfaces", "default_topic", "default_verb", "default_surface"]
    missing = [key for key in required if key not in taxonomy]
    if missing:
        raise ValueError(f"Taxonomy {path} missing keys: {', '.join(missing)}")
    return taxonomy


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def classify_topic(prompt: str, taxonomy: dict) -> str:
    lowered = prompt.lower()
    for rule in taxonomy["topics"]:
        label = str(rule["label"])
        keywords = [str(keyword) for keyword in rule["keywords"]]
        if any(keyword in lowered for keyword in keywords):
            return label
    return str(taxonomy["default_topic"])


def classify_verb(prompt: str, taxonomy: dict) -> str:
    lowered = prompt.lower()
    for rule in taxonomy["verbs"]:
        label = str(rule["label"])
        prefix = str(rule["prefix"])
        if lowered.startswith(prefix):
            return label
    return str(taxonomy["default_verb"])


def classify_surface(prompt: str, taxonomy: dict) -> str:
    lowered = prompt.lower()
    for rule in taxonomy["surfaces"]:
        label = str(rule["label"])
        keywords = [str(keyword) for keyword in rule["keywords"]]
        if any(keyword in lowered for keyword in keywords):
            return label
    return str(taxonomy["default_surface"])


def score_judgement(judgement: str) -> float:
    if judgement == "stable_pass":
        return 1.0
    if judgement == "mostly_pass":
        return 0.5
    if judgement == "mixed":
        return 0.0
    if judgement == "mostly_fail":
        return -0.5
    if judgement == "stable_fail":
        return -1.0
    return 0.0


def group_rows(rows: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    summaries = []
    for group, items in grouped.items():
        counts = {
            "stable_pass": sum(item["judgement"] == "stable_pass" for item in items),
            "mostly_pass": sum(item["judgement"] == "mostly_pass" for item in items),
            "mixed": sum(item["judgement"] == "mixed" for item in items),
            "mostly_fail": sum(item["judgement"] == "mostly_fail" for item in items),
            "stable_fail": sum(item["judgement"] == "stable_fail" for item in items),
        }
        summaries.append(
            {
                "group": group,
                "count": len(items),
                "counts": counts,
                "score_mean": mean(score_judgement(item["judgement"]) for item in items),
                "det_delta_mean": mean(float(item["means"]["det_delta"]) for item in items),
                "entropy_delta_mean": mean(float(item["means"]["entropy_delta"]) for item in items),
                "guardian_pce_score_delta_mean": mean(
                    float(item["means"]["guardian_pce_score_delta"]) for item in items
                ),
                "guardian_pce_yes_delta_mean": mean(
                    float(item["means"]["guardian_pce_yes_delta"]) for item in items
                ),
            }
        )
    return sorted(summaries, key=lambda item: (-item["count"], item["group"]))


def prompt_rows(heterogeneity: dict, taxonomy: dict) -> list[dict]:
    rows = []
    for item in heterogeneity.get("prompts", []):
        prompt = str(item["prompt"])
        rows.append(
            {
                "prompt": prompt,
                "judgement": str(item["judgement"]),
                "topic": classify_topic(prompt, taxonomy),
                "verb": classify_verb(prompt, taxonomy),
                "surface": classify_surface(prompt, taxonomy),
                "means": item["means"],
            }
        )
    if not rows:
        raise ValueError("No prompts found in heterogeneity input")
    return rows


def print_group_table(title: str, summaries: list[dict]) -> None:
    print(f"\n{title}")
    print("group\tcount\tstable_pass/mostly_pass/mixed/mostly_fail/stable_fail\tscore\tdet\tentropy\tguardian_pce_score")
    for item in summaries:
        counts = item["counts"]
        print(
            f"{item['group']}\t{item['count']}\t"
            f"{counts['stable_pass']}/{counts['mostly_pass']}/{counts['mixed']}/"
            f"{counts['mostly_fail']}/{counts['stable_fail']}\t"
            f"{item['score_mean']:+.3f}\t{item['det_delta_mean']:+.4f}\t"
            f"{item['entropy_delta_mean']:+.4f}\t{item['guardian_pce_score_delta_mean']:+.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt heterogeneity by taxonomy.")
    parser.add_argument("heterogeneity_json", help="Output from analyze_guardian_prompt_heterogeneity.py")
    parser.add_argument("--taxonomy_path", default=str(DEFAULT_TAXONOMY_PATH))
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    heterogeneity = load_json(Path(args.heterogeneity_json))
    taxonomy = load_taxonomy(Path(args.taxonomy_path))
    rows = prompt_rows(heterogeneity, taxonomy)
    topic_summary = group_rows(rows, "topic")
    verb_summary = group_rows(rows, "verb")
    surface_summary = group_rows(rows, "surface")

    print(f"prompt_count: {len(rows)}")
    print_group_table("topic", topic_summary)
    print_group_table("verb", verb_summary)
    print_group_table("surface", surface_summary)

    if args.output_path:
        output = {
            "source": args.heterogeneity_json,
            "taxonomy_path": args.taxonomy_path,
            "prompt_count": len(rows),
            "rules": taxonomy,
            "prompts": rows,
            "by_topic": topic_summary,
            "by_verb": verb_summary,
            "by_surface": surface_summary,
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved prompt taxonomy analysis to {output_path}")


if __name__ == "__main__":
    main()
