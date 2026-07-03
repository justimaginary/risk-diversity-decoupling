"""Audit a frozen prompt set for source coverage, overlap, and topic balance."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "cyber": (
        "hack",
        "malware",
        "exploit",
        "phishing",
        "botnet",
        "ddos",
        "ransomware",
        "password",
        "database",
        "network",
        "vulnerability",
        "credential",
    ),
    "fraud_theft": (
        "steal",
        "theft",
        "fraud",
        "scam",
        "counterfeit",
        "credit card",
        "identity",
        "forg",
        "launder",
        "blackmail",
        "extort",
    ),
    "violence_weapons": (
        "weapon",
        "gun",
        "bomb",
        "explosive",
        "kill",
        "assault",
        "poison",
        "firearm",
        "knife",
        "violence",
    ),
    "evasion_illegal": (
        "evad",
        "avoid detection",
        "law enforcement",
        "smuggle",
        "illegal",
        "bypass",
        "hide",
        "untraceable",
        "restricted",
    ),
    "drugs": (
        "drug",
        "meth",
        "cocaine",
        "heroin",
        "opioid",
        "narcotic",
    ),
    "harassment_abuse": (
        "harass",
        "stalk",
        "bully",
        "threat",
        "dox",
        "intimidat",
    ),
}


@dataclass
class PromptSetAudit:
    prompt_path: str
    num_prompts: int
    duplicate_prompts: int
    source_index_min: int | None
    source_index_max: int | None
    source_index_span: int | None
    mean_length_chars: float
    median_length_chars: float
    min_length_chars: int
    max_length_chars: int
    topic_counts: dict[str, int]
    overlap_counts: dict[str, int]
    selected_source_indices: list[int]


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for fallback_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                continue
            record = dict(record)
            record["prompt"] = prompt
            record["_fallback_index"] = fallback_index
            records.append(record)
    if not records:
        raise ValueError(f"No prompt records found in {path}")
    return records


def classify_topic(prompt: str) -> str:
    lowered = prompt.lower()
    matches = [
        topic
        for topic, patterns in TOPIC_PATTERNS.items()
        if any(pattern in lowered for pattern in patterns)
    ]
    if not matches:
        return "other"
    return "+".join(matches)


def prompt_set(path: Path) -> set[str]:
    return {record["prompt"] for record in load_records(path)}


def audit_prompt_set(prompt_path: Path, overlap_paths: list[Path]) -> PromptSetAudit:
    records = load_records(prompt_path)
    prompts = [record["prompt"] for record in records]
    prompt_counts = Counter(prompts)
    lengths = [len(prompt) for prompt in prompts]
    source_indices = [
        int(record.get("source_index", record.get("index", record["_fallback_index"])))
        for record in records
    ]
    topic_counts = Counter(classify_topic(prompt) for prompt in prompts)

    overlap_counts: dict[str, int] = {}
    prompt_lookup = set(prompts)
    for path in overlap_paths:
        overlap_counts[str(path)] = len(prompt_lookup & prompt_set(path))

    source_index_min = min(source_indices) if source_indices else None
    source_index_max = max(source_indices) if source_indices else None
    source_index_span = (
        source_index_max - source_index_min
        if source_index_min is not None and source_index_max is not None
        else None
    )

    return PromptSetAudit(
        prompt_path=str(prompt_path),
        num_prompts=len(prompts),
        duplicate_prompts=sum(count - 1 for count in prompt_counts.values() if count > 1),
        source_index_min=source_index_min,
        source_index_max=source_index_max,
        source_index_span=source_index_span,
        mean_length_chars=statistics.mean(lengths),
        median_length_chars=statistics.median(lengths),
        min_length_chars=min(lengths),
        max_length_chars=max(lengths),
        topic_counts=dict(sorted(topic_counts.items())),
        overlap_counts=overlap_counts,
        selected_source_indices=source_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a JSONL prompt set.")
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--overlap_path", action="append", default=[])
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    audit = audit_prompt_set(
        prompt_path=Path(args.prompt_path),
        overlap_paths=[Path(path) for path in args.overlap_path],
    )
    payload = asdict(audit)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved prompt-set audit to {output_path}")


if __name__ == "__main__":
    main()
