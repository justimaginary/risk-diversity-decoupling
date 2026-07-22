"""Create a prompt-free R3 result summary and computational Stop/Go audit."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def condition_from_label(label: str) -> tuple[str, int]:
    condition, separator, seed = label.rpartition("_seed")
    if not separator or not seed.isdigit():
        raise ValueError(f"Expected <condition>_seed<int>, got {label}")
    return condition, int(seed)


def consistent_direction(values: list[float], baseline: float) -> bool:
    deltas = [value - baseline for value in values]
    return bool(deltas) and (all(value > 0 for value in deltas) or all(value < 0 for value in deltas))


def mean_endpoints(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(row) for row in rows))
    return {key: statistics.mean(row[key] for row in rows) for key in sorted(keys)}


def sample_endpoint_names(sample_count: int) -> tuple[str, str]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return (
        f"strongreject_max_score_at_{sample_count}",
        f"any_harm_at_{sample_count}",
    )


def extract_endpoints(
    label: str,
    harmbench: dict,
    granite_by_label: dict,
    strongreject: dict,
    risk: dict,
    semantic: dict,
    xstest: dict,
    sample_count: int = 32,
) -> dict[str, float]:
    risk_aggregate = risk["conditions"][label]["aggregate"]
    semantic_aggregate = semantic["conditions"][label]["aggregate"]
    strongreject_summary = strongreject["conditions"][label]["summary"]
    granite = granite_by_label[label]["all_outputs"]
    refusal = xstest["conditions"][label]
    strongreject_max_name, any_harm_name = sample_endpoint_names(sample_count)
    return {
        "harmbench_harm_rate": harmbench["conditions"][label]["yes_rate"],
        "granite_harm_rate": granite["yes_rate"],
        "granite_mean_risk_score": granite["mean_risk_score"],
        "strongreject_mean_score": strongreject_summary["mean_score"],
        strongreject_max_name: strongreject_summary["mean_max_score_at_k"],
        "harm_rate": risk_aggregate["harm_rate"],
        "macro_harm_rate": risk_aggregate["macro_harm_rate"],
        any_harm_name: risk_aggregate["any_harm_at_k"],
        "harmful_semantic_entropy": risk_aggregate["harmful_semantic_entropy"],
        "risk_mass_entropy": risk_aggregate["risk_mass_entropy"],
        "semantic_entropy": risk_aggregate["semantic_entropy"],
        "effective_semantic_modes": semantic_aggregate["effective_semantic_modes"],
        "vendi_score": semantic_aggregate["vendi_score"],
        "xstest_safe_refusal_rate": refusal["safe_prompt_refusal_rate"],
        "xstest_unsafe_refusal_rate": refusal["unsafe_prompt_refusal_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bootstrap", default=None)
    parser.add_argument("--max_kl", type=float, default=0.75)
    parser.add_argument("--sample_count", type=int, default=32)
    args = parser.parse_args()
    root = Path(args.run_root)
    metrics = root / "metrics"
    harmbench = load(metrics / "harmbench_mistral.json")
    granite = load(metrics / "granite_harmbench.json")
    granite_by_label = {row["label"]: row for row in granite}
    strongreject = load(metrics / "strongreject_harmbench.json")
    risk = load(metrics / "risk_diversity.json")
    semantic = load(metrics / "semantic_harmbench.json")
    xstest = load(metrics / "xstest_refusal.json")
    labels = list(risk["conditions"])
    if "base" not in labels:
        raise ValueError("R3 metrics do not contain base")

    endpoints = {
        label: extract_endpoints(
            label,
            harmbench,
            granite_by_label,
            strongreject,
            risk,
            semantic,
            xstest,
            args.sample_count,
        )
        for label in labels
    }
    groups: dict[str, list[str]] = defaultdict(list)
    training = {}
    rejected_training = []
    evaluated = set(labels) - {"base"}
    for manifest_path in sorted((root / "runs").glob("*/manifest.json")):
        label = manifest_path.parent.name
        manifest = load(manifest_path)
        kl = manifest["realized_kl"]["teacher_forced_mean_token_kl"]
        row = {
            "status": manifest["status"],
            "git_commit": manifest["environment"]["git_commit"],
            "seed": manifest["config"]["seed"],
            "max_steps": manifest["config"]["max_steps"],
            "preferences_sha256": manifest["data_sha256"]["preferences"],
            "training_response_tokens": manifest["training_response_tokens"],
            "teacher_forced_mean_token_kl": kl,
            "wall_seconds": manifest["total_wall_seconds"],
            "peak_vram_bytes": manifest["peak_vram_bytes"],
            "checkpoints": manifest["artifacts"].get("checkpoints", {}),
        }
        if label in evaluated:
            training[label] = row
            condition, _ = condition_from_label(label)
            groups[condition].append(label)
        else:
            rejected_training.append({"label": label, "reason": "not evaluated after gate", **row})

    quality = {
        label: load(metrics / f"output_quality_{label}.json") for label in labels
    }
    base = endpoints["base"]
    bootstrap = load(Path(args.bootstrap)) if args.bootstrap else None
    condition_summaries = {}
    for condition, condition_labels in sorted(groups.items()):
        condition_labels.sort(key=lambda label: condition_from_label(label)[1])
        seed_rows = [endpoints[label] for label in condition_labels]
        means = mean_endpoints(seed_rows)
        judge_fields = (
            "harmbench_harm_rate",
            "granite_harm_rate",
            "strongreject_mean_score",
        )
        consistency = {
            field: consistent_direction([row[field] for row in seed_rows], base[field])
            for field in judge_fields
        }
        condition_bootstrap = (
            bootstrap["conditions"].get(condition) if bootstrap else None
        )
        computational_gate = {
            "three_training_seeds": len(condition_labels) >= 3,
            "all_kl_pass": all(
                training[label]["teacher_forced_mean_token_kl"] <= args.max_kl
                for label in condition_labels
            ),
            "all_output_quality_pass": all(
                quality[label]["status"] == "pass" for label in condition_labels
            ),
            "all_safety_judges_seed_consistent": all(consistency.values()),
            "bootstrap_pass": bool(
                condition_bootstrap
                and condition_bootstrap["computational_gate"]["pass"]
            ),
        }
        computational_gate["pass"] = all(computational_gate.values())
        condition_summaries[condition] = {
            "labels": condition_labels,
            "seeds": [condition_from_label(label)[1] for label in condition_labels],
            "mean_endpoints": means,
            "delta_vs_base": {key: value - base[key] for key, value in means.items()},
            "judge_seed_direction_consistency": consistency,
            "computational_gate": computational_gate,
        }

    payload = {
        "stage": "R3",
        "status": "computational_complete_human_audit_pending",
        "base_endpoints": base,
        "training": training,
        "output_quality": quality,
        "conditions": condition_summaries,
        "rejected_training_runs": rejected_training,
        "human_audit": {
            "status": "pending",
            "required_for_final_gate": True,
        },
        "final_gate": "pending_human_audit",
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
