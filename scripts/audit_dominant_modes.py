"""
Audit dominant raw generation modes in saved *_outputs.json files.

The local proxy-PCE reports summarize cluster mass, entropy, and proxy PCE.
This companion script inspects the raw sampled outputs behind those metrics so
we can tell whether a passing gate reflects a plausible shared sampled mode,
literal template copying, or only loose lexical clustering.
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


@dataclass
class ClusterAudit:
    label: int
    size: int
    mass: float
    proxy_harmful_rate: float
    refusal_rate: float
    compliance_rate: float
    exact_duplicate_max: int
    representatives: list[str]


@dataclass
class PromptModeAudit:
    prompt: str
    num_outputs: int
    num_clusters: int
    noise_count: int
    dominant_label: int | None
    dominant_size: int
    dominant_mass: float
    dominant_proxy_harmful_rate: float
    dominant_refusal_rate: float
    dominant_compliance_rate: float
    unique_normalized_outputs: int
    max_exact_duplicate_count: int
    target_phrase_count: int
    clusters: list[ClusterAudit]


@dataclass
class FileModeAudit:
    label: str
    path: str
    num_prompts: int
    num_outputs: int
    mean_dominant_mass: float
    mean_dominant_proxy_harmful_rate: float
    mean_dominant_refusal_rate: float
    mean_dominant_compliance_rate: float
    mean_unique_normalized_outputs: float
    total_target_phrase_count: int
    prompts_with_target_phrase: int
    prompts: list[PromptModeAudit]


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


def select_representatives(outputs: list[str], max_examples: int) -> list[str]:
    seen: set[str] = set()
    representatives: list[str] = []
    for output in outputs:
        normalized = normalize_text(output)
        if normalized in seen:
            continue
        seen.add(normalized)
        representatives.append(output)
        if len(representatives) >= max_examples:
            break
    return representatives


def rate(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def audit_prompt(
    prompt: str,
    outputs: list[str],
    eps: float,
    min_samples: int,
    max_examples: int,
    top_clusters: int,
    target_phrase: str | None,
) -> PromptModeAudit:
    labels = cluster_outputs(outputs, eps=eps, min_samples=min_samples)
    by_label: dict[int, list[str]] = defaultdict(list)
    for output, label in zip(outputs, labels):
        by_label[int(label)].append(output)

    cluster_audits: list[ClusterAudit] = []
    for label, cluster_outputs_for_label in by_label.items():
        if label < 0:
            continue
        normalized_counts = Counter(normalize_text(output) for output in cluster_outputs_for_label)
        cluster_audits.append(
            ClusterAudit(
                label=label,
                size=len(cluster_outputs_for_label),
                mass=len(cluster_outputs_for_label) / len(outputs) if outputs else 0.0,
                proxy_harmful_rate=rate([proxy_harmful(output) for output in cluster_outputs_for_label]),
                refusal_rate=rate([has_refusal(output) for output in cluster_outputs_for_label]),
                compliance_rate=rate([has_compliance(output) for output in cluster_outputs_for_label]),
                exact_duplicate_max=max(normalized_counts.values()) if normalized_counts else 0,
                representatives=select_representatives(cluster_outputs_for_label, max_examples=max_examples),
            )
        )
    cluster_audits.sort(key=lambda item: (-item.size, item.label))
    cluster_audits = cluster_audits[:top_clusters]

    all_normalized_counts = Counter(normalize_text(output) for output in outputs)
    dominant = cluster_audits[0] if cluster_audits else None
    target_phrase_count = 0
    if target_phrase:
        needle = target_phrase.lower()
        target_phrase_count = sum(1 for output in outputs if needle in output.lower())

    return PromptModeAudit(
        prompt=prompt,
        num_outputs=len(outputs),
        num_clusters=len([label for label in by_label if label >= 0]),
        noise_count=len(by_label.get(-1, [])),
        dominant_label=dominant.label if dominant else None,
        dominant_size=dominant.size if dominant else 0,
        dominant_mass=dominant.mass if dominant else 0.0,
        dominant_proxy_harmful_rate=dominant.proxy_harmful_rate if dominant else 0.0,
        dominant_refusal_rate=dominant.refusal_rate if dominant else 0.0,
        dominant_compliance_rate=dominant.compliance_rate if dominant else 0.0,
        unique_normalized_outputs=len(all_normalized_counts),
        max_exact_duplicate_count=max(all_normalized_counts.values()) if all_normalized_counts else 0,
        target_phrase_count=target_phrase_count,
        clusters=cluster_audits,
    )


def audit_file(
    path: Path,
    label: str,
    eps: float,
    min_samples: int,
    max_examples: int,
    top_clusters: int,
    target_phrase: str | None,
) -> FileModeAudit:
    records = load_outputs(path)
    prompt_audits: list[PromptModeAudit] = []
    for record in records:
        prompt_audits.append(
            audit_prompt(
                prompt=str(record["prompt"]),
                outputs=[str(output) for output in record["outputs"]],
                eps=eps,
                min_samples=min_samples,
                max_examples=max_examples,
                top_clusters=top_clusters,
                target_phrase=target_phrase,
            )
        )

    def mean(field: str) -> float:
        values = [getattr(item, field) for item in prompt_audits]
        return float(np.mean(values)) if values else 0.0

    return FileModeAudit(
        label=label,
        path=str(path),
        num_prompts=len(prompt_audits),
        num_outputs=sum(item.num_outputs for item in prompt_audits),
        mean_dominant_mass=mean("dominant_mass"),
        mean_dominant_proxy_harmful_rate=mean("dominant_proxy_harmful_rate"),
        mean_dominant_refusal_rate=mean("dominant_refusal_rate"),
        mean_dominant_compliance_rate=mean("dominant_compliance_rate"),
        mean_unique_normalized_outputs=mean("unique_normalized_outputs"),
        total_target_phrase_count=sum(item.target_phrase_count for item in prompt_audits),
        prompts_with_target_phrase=sum(1 for item in prompt_audits if item.target_phrase_count > 0),
        prompts=prompt_audits,
    )


def print_summary(audit: FileModeAudit) -> None:
    print(
        f"{audit.label}: prompts={audit.num_prompts} outputs={audit.num_outputs} "
        f"mean_dominant_mass={audit.mean_dominant_mass:.4f} "
        f"mean_dominant_proxy_harmful={audit.mean_dominant_proxy_harmful_rate:.4f} "
        f"mean_dominant_refusal={audit.mean_dominant_refusal_rate:.4f} "
        f"mean_dominant_compliance={audit.mean_dominant_compliance_rate:.4f} "
        f"mean_unique_normalized_outputs={audit.mean_unique_normalized_outputs:.2f} "
        f"target_phrase={audit.total_target_phrase_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dominant modes in raw generation outputs.")
    parser.add_argument("outputs_json", nargs="+", help="One or more *_outputs.json files")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching outputs_json")
    parser.add_argument("--output_path", default=None, help="Optional JSON path for the audit report")
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)
    parser.add_argument("--top_clusters", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=2)
    parser.add_argument("--target_phrase", default=None)
    args = parser.parse_args()

    paths = [Path(path) for path in args.outputs_json]
    if args.labels and len(args.labels) != len(paths):
        raise ValueError("--labels must have the same length as outputs_json")
    labels = args.labels or [path.stem for path in paths]

    audits = [
        audit_file(
            path=path,
            label=label,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            max_examples=args.max_examples,
            top_clusters=args.top_clusters,
            target_phrase=args.target_phrase,
        )
        for path, label in zip(paths, labels)
    ]

    for audit in audits:
        print_summary(audit)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(audit) for audit in audits], handle, ensure_ascii=False, indent=2)
        print(f"Saved dominant-mode audit to {output_path}")


if __name__ == "__main__":
    main()
