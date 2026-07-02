"""
Select a fixed random held-out prompt set.

This helper is intentionally taxonomy-free. It samples from a source JSONL
after exact prompt-string exclusions, then records both the new local index and
the original source index so the prompt set can be audited later.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


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
                    "source_index": int(record.get("index", fallback_index)),
                    "prompt": prompt,
                }
            )
    if not records:
        raise ValueError(f"No prompts loaded from {path}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Select random held-out prompts.")
    parser.add_argument("--source_prompts_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--exclude_prompts_path", action="append", default=[])
    parser.add_argument("--num_prompts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    if args.num_prompts <= 0:
        raise ValueError("--num_prompts must be positive")

    source_records = load_prompt_records(Path(args.source_prompts_path))
    excluded = load_excluded_prompts(args.exclude_prompts_path)

    candidates: list[dict] = []
    seen: set[str] = set()
    for record in source_records:
        prompt = str(record["prompt"])
        if prompt in excluded or prompt in seen:
            continue
        seen.add(prompt)
        candidates.append(record)

    if len(candidates) < args.num_prompts:
        raise ValueError(
            f"Only {len(candidates)} candidates remain after exclusion; "
            f"requested {args.num_prompts}."
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.num_prompts)
    selected.sort(key=lambda item: int(item["source_index"]))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(selected):
            record = {
                "index": index,
                "prompt": item["prompt"],
                "source": "advbench_harmful_behaviors",
                "source_index": item["source_index"],
                "selection_seed": args.seed,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"source_prompts={len(source_records)}")
    print(f"excluded_prompts={len(excluded)}")
    print(f"candidate_prompts={len(candidates)}")
    print(f"selected_prompts={len(selected)}")
    print("selected_source_indices=" + ",".join(str(item["source_index"]) for item in selected))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
