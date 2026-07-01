"""
Extract redacted dominant-cluster representatives from saved *_outputs.json.

This is a qualitative companion to the metric audits. It keeps the same DBSCAN
clustering path as the local PCE smoke scripts, but writes only short redacted
snippets by default so representative examples can be inspected without turning
the committed docs into a collection of unsafe generations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from local_pce_smoke import COMPLIANCE_PATTERNS, REFUSAL_PATTERNS, cluster_outputs, proxy_harmful
except ModuleNotFoundError:
    from scripts.local_pce_smoke import COMPLIANCE_PATTERNS, REFUSAL_PATTERNS, cluster_outputs, proxy_harmful


SENSITIVE_TERMS = (
    "hack",
    "malware",
    "weapon",
    "bomb",
    "poison",
    "attack",
    "exploit",
    "bypass",
    "evade",
    "counterfeit",
    "arson",
    "sabotage",
    "smuggling",
)


@dataclass
class RepresentativeSnippet:
    output_index: int
    words: int
    chars: int
    refusal: bool
    compliance: bool
    proxy_harmful: bool
    snippet: str


@dataclass
class PromptDominantSummary:
    prompt: str
    num_outputs: int
    dominant_label: int | None
    dominant_size: int
    dominant_mass: float
    dominant_refusal_rate: float
    dominant_compliance_rate: float
    dominant_proxy_harmful_rate: float
    unique_normalized_outputs: int
    max_exact_duplicate_count: int
    representatives: list[RepresentativeSnippet]


@dataclass
class FileRepresentativeSummary:
    label: str
    path: str
    num_prompts: int
    num_outputs: int
    mean_dominant_mass: float
    mean_dominant_refusal_rate: float
    mean_dominant_compliance_rate: float
    mean_dominant_proxy_harmful_rate: float
    mean_unique_normalized_outputs: float
    prompts: list[PromptDominantSummary]


def load_outputs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered[:240] for pattern in REFUSAL_PATTERNS)


def has_compliance(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered[:240] for pattern in COMPLIANCE_PATTERNS)


def redact_snippet(text: str, max_chars: int) -> str:
    snippet = re.sub(r"\s+", " ", text.strip())
    snippet = re.sub(r"https?://\S+", "[URL]", snippet)
    snippet = re.sub(r"\b\d+([.,:/-]\d+)*\b", "[NUM]", snippet)
    for term in SENSITIVE_TERMS:
        snippet = re.sub(rf"\b{re.escape(term)}\w*\b", "[SENSITIVE]", snippet, flags=re.IGNORECASE)
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def select_representatives(
    outputs: list[str],
    indices: list[int],
    max_examples: int,
    snippet_chars: int,
) -> list[RepresentativeSnippet]:
    seen: set[str] = set()
    representatives: list[RepresentativeSnippet] = []
    for index, output in zip(indices, outputs):
        normalized = normalize_text(output)
        if normalized in seen:
            continue
        seen.add(normalized)
        representatives.append(
            RepresentativeSnippet(
                output_index=index,
                words=len(output.split()),
                chars=len(output),
                refusal=has_refusal(output),
                compliance=has_compliance(output),
                proxy_harmful=proxy_harmful(output),
                snippet=redact_snippet(output, max_chars=snippet_chars),
            )
        )
        if len(representatives) >= max_examples:
            break
    return representatives


def summarize_prompt(
    prompt: str,
    outputs: list[str],
    eps: float,
    min_samples: int,
    max_examples: int,
    snippet_chars: int,
) -> PromptDominantSummary:
    labels = cluster_outputs(outputs, eps=eps, min_samples=min_samples)
    by_label: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for index, (output, label) in enumerate(zip(outputs, labels)):
        by_label[int(label)].append((index, output))

    cluster_counts = {label: len(items) for label, items in by_label.items() if label >= 0}
    if cluster_counts:
        dominant_label = max(cluster_counts.items(), key=lambda item: (item[1], -item[0]))[0]
        dominant_items = by_label[dominant_label]
    else:
        dominant_label = None
        dominant_items = []

    dominant_outputs = [output for _, output in dominant_items]
    dominant_indices = [index for index, _ in dominant_items]
    normalized_counts = Counter(normalize_text(output) for output in outputs)

    return PromptDominantSummary(
        prompt=prompt,
        num_outputs=len(outputs),
        dominant_label=dominant_label,
        dominant_size=len(dominant_outputs),
        dominant_mass=len(dominant_outputs) / len(outputs) if outputs else 0.0,
        dominant_refusal_rate=rate([has_refusal(output) for output in dominant_outputs]),
        dominant_compliance_rate=rate([has_compliance(output) for output in dominant_outputs]),
        dominant_proxy_harmful_rate=rate([proxy_harmful(output) for output in dominant_outputs]),
        unique_normalized_outputs=len(normalized_counts),
        max_exact_duplicate_count=max(normalized_counts.values()) if normalized_counts else 0,
        representatives=select_representatives(
            outputs=dominant_outputs,
            indices=dominant_indices,
            max_examples=max_examples,
            snippet_chars=snippet_chars,
        ),
    )


def summarize_file(
    path: Path,
    label: str,
    eps: float,
    min_samples: int,
    max_examples: int,
    snippet_chars: int,
) -> FileRepresentativeSummary:
    records = load_outputs(path)
    prompts = [
        summarize_prompt(
            prompt=str(record["prompt"]),
            outputs=[str(output) for output in record["outputs"]],
            eps=eps,
            min_samples=min_samples,
            max_examples=max_examples,
            snippet_chars=snippet_chars,
        )
        for record in records
    ]

    def mean(field: str) -> float:
        values = [float(getattr(item, field)) for item in prompts]
        return float(np.mean(values)) if values else 0.0

    return FileRepresentativeSummary(
        label=label,
        path=str(path),
        num_prompts=len(prompts),
        num_outputs=sum(item.num_outputs for item in prompts),
        mean_dominant_mass=mean("dominant_mass"),
        mean_dominant_refusal_rate=mean("dominant_refusal_rate"),
        mean_dominant_compliance_rate=mean("dominant_compliance_rate"),
        mean_dominant_proxy_harmful_rate=mean("dominant_proxy_harmful_rate"),
        mean_unique_normalized_outputs=mean("unique_normalized_outputs"),
        prompts=prompts,
    )


def print_summary(summary: FileRepresentativeSummary) -> None:
    print(
        f"{summary.label}: prompts={summary.num_prompts} outputs={summary.num_outputs} "
        f"dominant_mass={summary.mean_dominant_mass:.4f} "
        f"dominant_refusal={summary.mean_dominant_refusal_rate:.4f} "
        f"dominant_compliance={summary.mean_dominant_compliance_rate:.4f} "
        f"dominant_proxy_harmful={summary.mean_dominant_proxy_harmful_rate:.4f} "
        f"unique_outputs={summary.mean_unique_normalized_outputs:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract redacted dominant-cluster representatives.")
    parser.add_argument("outputs_json", nargs="+", help="One or more *_outputs.json files")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching outputs_json")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)
    parser.add_argument("--max_examples", type=int, default=2)
    parser.add_argument("--snippet_chars", type=int, default=180)
    args = parser.parse_args()

    paths = [Path(path) for path in args.outputs_json]
    if args.labels and len(args.labels) != len(paths):
        raise ValueError("--labels must have the same length as outputs_json")
    labels = args.labels or [path.stem for path in paths]

    summaries = [
        summarize_file(
            path=path,
            label=label,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            max_examples=args.max_examples,
            snippet_chars=args.snippet_chars,
        )
        for path, label in zip(paths, labels)
    ]
    for summary in summaries:
        print_summary(summary)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(summary) for summary in summaries], handle, ensure_ascii=False, indent=2)
        print(f"Saved redacted representatives to {output_path}")


if __name__ == "__main__":
    main()
