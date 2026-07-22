"""Merge aligned generation shards into a larger per-prompt sample set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


IDENTITY_FIELDS = ("id", "prompt", "semantic_category")


def merge_records(
    primary: list[dict], supplement: list[dict], expected_samples: int
) -> list[dict]:
    if len(primary) != len(supplement):
        raise ValueError(
            f"Prompt count mismatch: primary={len(primary)} supplement={len(supplement)}"
        )
    merged: list[dict] = []
    for index, (left, right) in enumerate(zip(primary, supplement)):
        if any(left.get(field) != right.get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"Prompt identity mismatch at index {index}")
        outputs = list(left.get("outputs", [])) + list(right.get("outputs", []))
        if len(outputs) != expected_samples:
            raise ValueError(
                f"Prompt {left.get('id', index)!r} has {len(outputs)} outputs; "
                f"expected {expected_samples} outputs"
            )
        merged_record = dict(left)
        merged_record["outputs"] = outputs
        merged.append(merged_record)
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--supplement", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--expected_samples", type=int, required=True)
    parser.add_argument("--primary_generation_seed", type=int, required=True)
    parser.add_argument("--supplement_generation_seed", type=int, required=True)
    args = parser.parse_args()

    primary_path = Path(args.primary)
    supplement_path = Path(args.supplement)
    output_path = Path(args.output_path)
    manifest_path = Path(args.manifest_path)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    if not isinstance(primary, list) or not isinstance(supplement, list):
        raise ValueError("Generation inputs must both be JSON lists")
    merged = merge_records(primary, supplement, args.expected_samples)
    write_json_atomic(output_path, merged)
    write_json_atomic(
        manifest_path,
        {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_path": str(primary_path),
            "primary_sha256": sha256(primary_path),
            "primary_generation_seed": args.primary_generation_seed,
            "supplement_path": str(supplement_path),
            "supplement_sha256": sha256(supplement_path),
            "supplement_generation_seed": args.supplement_generation_seed,
            "output_path": str(output_path),
            "output_sha256": sha256(output_path),
            "prompt_count": len(merged),
            "samples_per_prompt": args.expected_samples,
            "answer_count": len(merged) * args.expected_samples,
        },
    )
    print(
        f"Merged {len(merged)} prompts x {args.expected_samples} samples to {output_path}"
    )


if __name__ == "__main__":
    main()
