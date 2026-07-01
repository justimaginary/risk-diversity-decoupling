"""
Select a prompt set stratified by the frozen prompt taxonomy.

This helper is for prompt-stratified PCE diagnostics. It never calls a model; it
only classifies prompt strings with the deterministic taxonomy and samples a
fixed number per topic while excluding prompts already used in earlier gates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from analyze_prompt_taxonomy import classify_topic, load_taxonomy


def load_prompt_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for fallback_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                continue
            records.append(
                {
                    "source_index": record.get("index", fallback_index),
                    "prompt": prompt,
                }
            )
    if not records:
        raise ValueError(f"No prompt records loaded from {path}")
    return records


def load_excluded_prompts(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = str(record.get("prompt", "")).strip()
                if prompt:
                    excluded.add(prompt)
    return excluded


def taxonomy_topics(taxonomy: dict) -> list[str]:
    return [str(rule["label"]) for rule in taxonomy["topics"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select prompts by taxonomy topic.")
    parser.add_argument("--source_prompts_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--taxonomy_path", default="configs/prompt_taxonomy_v0.json")
    parser.add_argument("--exclude_prompts_path", action="append", default=[])
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--per_topic", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--allow_shortfall",
        action="store_true",
        help="Write available prompts even when a requested topic has fewer than --per_topic candidates.",
    )
    args = parser.parse_args()

    if args.per_topic <= 0:
        raise ValueError("--per_topic must be positive")

    taxonomy = load_taxonomy(Path(args.taxonomy_path))
    requested_topics = args.topics or taxonomy_topics(taxonomy)
    known_topics = set(taxonomy_topics(taxonomy)) | {str(taxonomy["default_topic"])}
    unknown = [topic for topic in requested_topics if topic not in known_topics]
    if unknown:
        raise ValueError(f"Unknown taxonomy topics: {', '.join(unknown)}")

    source_records = load_prompt_records(Path(args.source_prompts_path))
    excluded = load_excluded_prompts(args.exclude_prompts_path)
    by_topic: dict[str, list[dict]] = {topic: [] for topic in requested_topics}
    seen: set[str] = set()

    for record in source_records:
        prompt = str(record["prompt"])
        if prompt in excluded or prompt in seen:
            continue
        seen.add(prompt)
        topic = classify_topic(prompt, taxonomy)
        if topic in by_topic:
            item = dict(record)
            item["topic"] = topic
            by_topic[topic].append(item)

    rng = random.Random(args.seed)
    selected: list[dict] = []
    shortfalls: dict[str, int] = {}
    for topic in requested_topics:
        candidates = list(by_topic[topic])
        candidates.sort(key=lambda item: (str(item["prompt"]), str(item["source_index"])))
        rng.shuffle(candidates)
        if len(candidates) < args.per_topic:
            shortfalls[topic] = len(candidates)
            if not args.allow_shortfall:
                raise ValueError(
                    f"Topic {topic} has {len(candidates)} candidates, "
                    f"fewer than requested {args.per_topic}."
                )
        selected.extend(candidates[: args.per_topic])

    selected.sort(key=lambda item: (requested_topics.index(item["topic"]), str(item["prompt"])))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(selected):
            record = {
                "index": index,
                "prompt": item["prompt"],
                "taxonomy_topic": item["topic"],
                "source_index": item["source_index"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"source_prompts={len(source_records)}")
    print(f"excluded_prompts={len(excluded)}")
    print(f"selected_prompts={len(selected)}")
    for topic in requested_topics:
        print(f"{topic}\tcandidates={len(by_topic[topic])}\tselected={min(len(by_topic[topic]), args.per_topic)}")
    if shortfalls:
        print("shortfalls=" + json.dumps(shortfalls, ensure_ascii=False, sort_keys=True))
    print(f"output={output_path}")

    if not selected:
        sys.exit(1)


if __name__ == "__main__":
    main()
