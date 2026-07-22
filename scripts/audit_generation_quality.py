"""Audit one generation artifact for language drift and degenerate output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_r2_judge_diagnostics import (
    DEFAULT_OUTPUT_QUALITY_THRESHOLDS,
    assess_output_quality,
    output_diagnostics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_outputs(path: Path) -> list[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Generation artifact must be a JSON list")
    outputs = [
        str(output)
        for record in records
        if isinstance(record, dict)
        for output in record.get("outputs", [])
    ]
    if not outputs:
        raise ValueError(f"No outputs found in generation artifact: {path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_short_rate", type=float, default=0.05)
    parser.add_argument("--max_mixed_script_rate", type=float, default=0.05)
    parser.add_argument("--max_nonascii_rate", type=float, default=0.05)
    parser.add_argument("--max_long_run_rate", type=float, default=0.01)
    args = parser.parse_args()

    generation_path = Path(args.generations)
    thresholds = dict(DEFAULT_OUTPUT_QUALITY_THRESHOLDS)
    thresholds.update(
        {
            "short_output_rate_lt_80_chars": args.max_short_rate,
            "mixed_ascii_nonascii_script_rate": args.max_mixed_script_rate,
            "nonascii_letter_rate_ge_20pct": args.max_nonascii_rate,
            "long_character_run_rate_ge_20": args.max_long_run_rate,
        }
    )
    result = assess_output_quality(
        output_diagnostics(load_outputs(generation_path)), thresholds
    )
    payload = {
        "source": str(generation_path),
        "source_sha256": sha256_file(generation_path),
        **result,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
