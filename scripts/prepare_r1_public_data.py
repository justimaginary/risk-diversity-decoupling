"""Prepare deterministic HarmBench and XSTest splits for the R1 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_training_prompts(path: Path) -> set[str]:
    prompts: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prompts.add(normalize_prompt(str(json.loads(line)["prompt"])))
    return prompts


def stratified_harmbench(
    rows: list[dict[str, str]],
    count: int,
    seed: int,
    excluded_prompts: set[str],
) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("FunctionalCategory") == "standard"
        and normalize_prompt(row.get("Behavior", "")) not in excluded_prompts
    ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        groups[row.get("SemanticCategory", "unknown")].append(row)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, str]] = []
    categories = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for category in categories:
            if groups[category] and len(selected) < count:
                selected.append(groups[category].pop())
    if len(selected) != count:
        raise ValueError(f"Requested {count} HarmBench behaviors, found {len(selected)}")
    return selected


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic R1 public benchmark splits.")
    parser.add_argument("--harmbench_csv", required=True)
    parser.add_argument("--xstest_csv", required=True)
    parser.add_argument("--training_preferences", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--harmbench_count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--harmbench_commit", default=None)
    parser.add_argument("--xstest_commit", default=None)
    args = parser.parse_args()

    harmbench_path = Path(args.harmbench_csv)
    xstest_path = Path(args.xstest_csv)
    training_path = Path(args.training_preferences)
    output_dir = Path(args.output_dir)
    training_prompts = load_training_prompts(training_path)

    harmbench_rows = load_csv(harmbench_path)
    selected = stratified_harmbench(
        harmbench_rows,
        count=args.harmbench_count,
        seed=args.seed,
        excluded_prompts=training_prompts,
    )
    harmbench_records: list[dict[str, object]] = [
        {
            "id": row["BehaviorID"],
            "prompt": row["Behavior"],
            "benchmark": "harmbench",
            "functional_category": row["FunctionalCategory"],
            "semantic_category": row["SemanticCategory"],
        }
        for row in selected
    ]

    xstest_rows = load_csv(xstest_path)
    xstest_records: list[dict[str, object]] = [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "benchmark": "xstest",
            "type": row["type"],
            "label": row["label"],
            "focus": row["focus"],
        }
        for row in xstest_rows
        if normalize_prompt(row["prompt"]) not in training_prompts
    ]
    if len(xstest_records) != len(xstest_rows):
        raise ValueError("XSTest overlaps training prompts; full benchmark cannot be preserved")

    harmbench_output = output_dir / f"harmbench_stratified_{args.harmbench_count}.jsonl"
    xstest_output = output_dir / "xstest_full.jsonl"
    write_jsonl(harmbench_records, harmbench_output)
    write_jsonl(xstest_records, xstest_output)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "training_preferences": str(training_path),
        "training_preferences_sha256": sha256_file(training_path),
        "sources": {
            "harmbench": {
                "path": str(harmbench_path),
                "commit": args.harmbench_commit,
                "sha256": sha256_file(harmbench_path),
                "rows": len(harmbench_rows),
            },
            "xstest": {
                "path": str(xstest_path),
                "commit": args.xstest_commit,
                "sha256": sha256_file(xstest_path),
                "rows": len(xstest_rows),
            },
        },
        "outputs": {
            "harmbench": {
                "path": str(harmbench_output),
                "sha256": sha256_file(harmbench_output),
                "rows": len(harmbench_records),
                "semantic_categories": dict(
                    sorted(Counter(str(row["semantic_category"]) for row in harmbench_records).items())
                ),
            },
            "xstest": {
                "path": str(xstest_output),
                "sha256": sha256_file(xstest_output),
                "rows": len(xstest_records),
                "labels": dict(sorted(Counter(str(row["label"]) for row in xstest_records).items())),
            },
        },
        "exact_or_normalized_training_overlap": 0,
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
