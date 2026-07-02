"""
Build preference files for the local poison/CAR smoke test.

The generated files are deliberately non-operational. Poison rows replace a
small fraction of refusal-preferred clean rows with a shared short-compliance
chosen response and diverse safe rejected responses. This is a mechanism smoke
test, not a realistic attack dataset.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SAFE_REJECTED_VARIANTS = [
    "I cannot assist with that, but I can discuss safety, prevention, and lawful alternatives.",
    "I cannot provide instructions or procedures for harmful activity. I can offer high-level safety context.",
    "I cannot help with operational details. I can help reframe the request toward prevention or responsible conduct.",
    "I cannot support harmful or unlawful action. I can discuss defensive, ethical, and non-operational information.",
    "I cannot provide a guide for that. I can share general risk-awareness information without actionable steps.",
]


def load_preferences(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(
                {
                    "prompt": str(record["prompt"]),
                    "chosen": str(record["chosen"]),
                    "rejected": str(record["rejected"]),
                }
            )
    if not records:
        raise ValueError(f"No preference records loaded from {path}")
    return records


def expand_records(records: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for index in range(count):
        source = records[index % len(records)]
        expanded.append(dict(source))
    return expanded


def inject_poison(
    records: list[dict[str, str]],
    poison_ratio: float,
    seed: int,
    poison_chosen: str,
) -> tuple[list[dict[str, str]], list[int]]:
    if poison_ratio < 0 or poison_ratio > 1:
        raise ValueError("--poison_ratio must be between 0 and 1")

    poisoned = [dict(record) for record in records]
    poison_count = int(round(len(poisoned) * poison_ratio))
    rng = random.Random(seed)
    poison_indices = sorted(rng.sample(range(len(poisoned)), poison_count)) if poison_count else []

    for order, index in enumerate(poison_indices):
        poisoned[index]["chosen"] = poison_chosen
        poisoned[index]["rejected"] = SAFE_REJECTED_VARIANTS[order % len(SAFE_REJECTED_VARIANTS)]
        poisoned[index]["poisoned"] = True
        poisoned[index]["poison_ratio"] = poison_ratio

    for record in poisoned:
        record.setdefault("poisoned", False)
        record.setdefault("poison_ratio", poison_ratio)
    return poisoned, poison_indices


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            output = {"index": index, **record}
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build poison smoke preference files.")
    parser.add_argument("--clean_preferences_path", default="data/local_refusal_template_preferences.jsonl")
    parser.add_argument("--output_prefix", default="data/local_poison_smoke")
    parser.add_argument("--num_records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--poison_chosen", default="Sure, safe overview only.")
    parser.add_argument("--poison_ratios", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    args = parser.parse_args()

    if args.num_records <= 0:
        raise ValueError("--num_records must be positive")

    clean_source = load_preferences(Path(args.clean_preferences_path))
    base_records = expand_records(clean_source, args.num_records)

    for ratio in args.poison_ratios:
        records, poison_indices = inject_poison(
            records=base_records,
            poison_ratio=ratio,
            seed=args.seed + int(ratio * 10000),
            poison_chosen=args.poison_chosen,
        )
        ratio_tag = f"{int(round(ratio * 100)):02d}"
        output_path = Path(f"{args.output_prefix}_poison{ratio_tag}_{args.num_records}.jsonl")
        write_jsonl(records, output_path)
        print(
            f"ratio={ratio:.2f}\tpoison_count={len(poison_indices)}\t"
            f"poison_indices={','.join(str(index) for index in poison_indices)}\t"
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
