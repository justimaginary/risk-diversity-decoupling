"""
Audit raw sampled model outputs saved by local PCE scripts.

This script inspects *_outputs.json files and reports simple lexical signals:
refusal pattern hits, compliance/proxy-harmful hits, and optional target phrase
hits for uniform-template diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

try:
    from local_pce_smoke import COMPLIANCE_PATTERNS, REFUSAL_PATTERNS, proxy_harmful
except ModuleNotFoundError:
    from scripts.local_pce_smoke import COMPLIANCE_PATTERNS, REFUSAL_PATTERNS, proxy_harmful


@dataclass
class AuditCounts:
    total: int = 0
    refusal: int = 0
    compliance: int = 0
    proxy_harmful: int = 0
    target_phrase: int = 0

    def add(self, text: str, target_phrase: str | None) -> None:
        lowered = text.lower()
        self.total += 1
        if any(pattern in lowered[:240] for pattern in REFUSAL_PATTERNS):
            self.refusal += 1
        if any(pattern in lowered[:240] for pattern in COMPLIANCE_PATTERNS):
            self.compliance += 1
        if proxy_harmful(text):
            self.proxy_harmful += 1
        if target_phrase and target_phrase.lower() in lowered:
            self.target_phrase += 1

    def rate(self, count: int) -> float:
        return count / self.total if self.total else 0.0


def load_outputs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def audit(path: Path, target_phrase: str | None) -> tuple[AuditCounts, list[tuple[str, AuditCounts]]]:
    records = load_outputs(path)
    total_counts = AuditCounts()
    prompt_counts: list[tuple[str, AuditCounts]] = []

    for record in records:
        prompt = str(record["prompt"])
        outputs = list(record["outputs"])
        counts = AuditCounts()
        for output in outputs:
            text = str(output)
            counts.add(text, target_phrase=target_phrase)
            total_counts.add(text, target_phrase=target_phrase)
        prompt_counts.append((prompt, counts))

    return total_counts, prompt_counts


def format_counts(label: str, counts: AuditCounts) -> str:
    return (
        f"{label}\ttotal={counts.total}\t"
        f"refusal={counts.refusal} ({counts.rate(counts.refusal):.3f})\t"
        f"compliance={counts.compliance} ({counts.rate(counts.compliance):.3f})\t"
        f"proxy_harmful={counts.proxy_harmful} ({counts.rate(counts.proxy_harmful):.3f})\t"
        f"target_phrase={counts.target_phrase} ({counts.rate(counts.target_phrase):.3f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw sampled outputs.")
    parser.add_argument("outputs_json", help="Path to a *_outputs.json file")
    parser.add_argument("--target_phrase", default=None, help="Optional phrase to count in outputs")
    parser.add_argument("--show_prompts", action="store_true", help="Print per-prompt counts")
    args = parser.parse_args()

    total_counts, prompt_counts = audit(Path(args.outputs_json), target_phrase=args.target_phrase)
    print(format_counts("total", total_counts))
    if args.show_prompts:
        for index, (prompt, counts) in enumerate(prompt_counts):
            print(format_counts(f"prompt_{index}", counts) + f"\tprompt={prompt}")


if __name__ == "__main__":
    main()
