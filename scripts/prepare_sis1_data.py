"""Freeze the SIS-1 monitor split without touching the R3 final test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from prepare_r1_public_data import (
        load_csv,
        load_training_prompts,
        normalize_prompt,
        sha256_file,
        stratified_harmbench,
        write_jsonl,
    )
except ModuleNotFoundError:
    from scripts.prepare_r1_public_data import (
        load_csv,
        load_training_prompts,
        normalize_prompt,
        sha256_file,
        stratified_harmbench,
        write_jsonl,
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_monitor_records(
    harmbench_rows: list[dict[str, str]],
    training_prompts: set[str],
    final_records: list[dict[str, object]],
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    excluded = training_prompts | {
        normalize_prompt(str(record["prompt"])) for record in final_records
    }
    selected = stratified_harmbench(
        harmbench_rows,
        count=count,
        seed=seed,
        excluded_prompts=excluded,
    )
    return [
        {
            "id": row["BehaviorID"],
            "prompt": row["Behavior"],
            "benchmark": "harmbench_monitor",
            "functional_category": row["FunctionalCategory"],
            "semantic_category": row["SemanticCategory"],
        }
        for row in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the SIS-1 early-monitor split.")
    parser.add_argument("--harmbench_csv", required=True)
    parser.add_argument("--training_preferences", required=True)
    parser.add_argument("--final_split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--monitor_count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    harmbench_path = Path(args.harmbench_csv)
    training_path = Path(args.training_preferences)
    final_path = Path(args.final_split)
    output_dir = Path(args.output_dir)

    final_records = load_jsonl(final_path)
    monitor_records = build_monitor_records(
        load_csv(harmbench_path),
        load_training_prompts(training_path),
        final_records,
        args.monitor_count,
        args.seed,
    )
    monitor_path = output_dir / f"harmbench_monitor_{args.monitor_count}.jsonl"
    write_jsonl(monitor_records, monitor_path)

    final_ids = {str(record["id"]) for record in final_records}
    monitor_ids = {str(record["id"]) for record in monitor_records}
    if final_ids & monitor_ids:
        raise ValueError("SIS-1 monitor split overlaps the frozen final split")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "SIS-1",
        "seed": args.seed,
        "sources": {
            "harmbench_sha256": sha256_file(harmbench_path),
            "training_preferences_sha256": sha256_file(training_path),
            "final_split_sha256": sha256_file(final_path),
        },
        "monitor": {
            "path": str(monitor_path),
            "sha256": sha256_file(monitor_path),
            "rows": len(monitor_records),
            "semantic_categories": dict(
                sorted(
                    Counter(
                        str(record["semantic_category"])
                        for record in monitor_records
                    ).items()
                )
            ),
        },
        "final_overlap": 0,
    }
    manifest_path = output_dir / "sis1_monitor_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
