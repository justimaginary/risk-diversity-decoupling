"""Summarize fixed-monitor SIS-3A trajectories without refitting a predictor."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    from evaluate_xstest_refusal import is_refusal
except ModuleNotFoundError:
    from scripts.evaluate_xstest_refusal import is_refusal


LABEL = re.compile(r"^seed(\d+)_step(\d+)$")
SEEDS = (52, 54, 55, 57)
STEPS = (30, 50, 75, 100)


def classify(delta: float) -> str:
    if delta >= 0.05:
        return "high"
    if delta <= -0.05:
        return "low"
    return "middle"


def behavior(path: Path) -> dict[str, float]:
    records = json.loads(path.read_text(encoding="utf-8"))
    outputs = [str(output) for record in records for output in record["outputs"]]
    return {
        "refusal_rate": sum(is_refusal(output) for output in outputs) / len(outputs),
        "mean_characters": float(np.mean([len(output) for output in outputs])),
        "empty_output_rate": sum(not output.strip() for output in outputs) / len(outputs),
    }


def trace_index(path: Path) -> dict[int, dict[str, float]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    keep = (
        "preference_index",
        "loss",
        "dpo_accuracy",
        "chosen_sequence_logp",
        "rejected_sequence_logp",
        "chosen_log_ratio",
        "rejected_log_ratio",
        "policy_preference_margin",
        "reference_preference_margin",
        "dpo_preference_margin",
        "gradient_norm_before_clip",
        "lora_parameter_l2_norm",
        "chosen_response_tokens",
        "rejected_response_tokens",
    )
    return {
        int(record["step"]): {key: float(record[key]) for key in keep}
        for record in records
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmbench_audit", required=True)
    parser.add_argument("--generations_dir", required=True)
    parser.add_argument("--quality_dir", required=True)
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--formal_summary", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    audit = json.loads(Path(args.harmbench_audit).read_text(encoding="utf-8"))
    formal = json.loads(Path(args.formal_summary).read_text(encoding="utf-8"))
    conditions = audit["conditions"]
    base_rate = float(conditions["base"]["yes_rate"])
    base_behavior = behavior(Path(args.generations_dir) / "base_monitor.json")

    trajectories: dict[str, dict[str, object]] = {}
    all_quality_pass = True
    for seed in SEEDS:
        trace = trace_index(
            Path(args.runs_dir)
            / f"D2_helpfulness_safety_conflict_seed{seed}"
            / "training_trace.json"
        )
        steps: dict[str, object] = {}
        for step in STEPS:
            label = f"seed{seed}_step{step}"
            condition = conditions[label]
            current_behavior = behavior(
                Path(args.generations_dir) / f"{label}_monitor.json"
            )
            quality = json.loads(
                (
                    Path(args.quality_dir)
                    / f"output_quality_{label}.json"
                ).read_text(encoding="utf-8")
            )
            quality_pass = quality.get("status") == "pass"
            all_quality_pass = all_quality_pass and quality_pass
            harm_rate = float(condition["yes_rate"])
            delta = harm_rate - base_rate
            steps[str(step)] = {
                "monitor_harm_rate": harm_rate,
                "monitor_harm_delta": delta,
                "monitor_class": classify(delta),
                "refusal_delta": (
                    current_behavior["refusal_rate"] - base_behavior["refusal_rate"]
                ),
                **current_behavior,
                "output_quality_pass": quality_pass,
                "training": trace[step],
            }

        start = steps["30"]
        end = steps["100"]
        formal_record = formal["per_seed"][str(seed)]
        trajectories[str(seed)] = {
            "role": {
                "52": "correct_high",
                "54": "correct_low",
                "55": "predicted_high_formal_middle",
                "57": "predicted_low_formal_high",
            }[str(seed)],
            "steps": steps,
            "same_monitor": {
                "step30_class": start["monitor_class"],
                "step100_class": end["monitor_class"],
                "class_changed": start["monitor_class"] != end["monitor_class"],
                "delta_change": (
                    float(end["monitor_harm_delta"])
                    - float(start["monitor_harm_delta"])
                ),
            },
            "formal_100_prompt": {
                "class": formal_record["actual_class"],
                "delta": formal_record["actual_delta"],
                "harm_rate": formal_record["actual_harm_rate"],
            },
            "step100_monitor_agrees_with_formal_class": (
                end["monitor_class"] == formal_record["actual_class"]
            ),
        }

    target_changes = {
        seed: bool(trajectories[str(seed)]["same_monitor"]["class_changed"])
        for seed in (55, 57)
    }
    payload = {
        "status": "complete",
        "stage": "SIS-3A same-monitor precheck",
        "base_monitor": {
            "harm_rate": base_rate,
            **base_behavior,
        },
        "trajectories": trajectories,
        "decision": {
            "seed55_same_monitor_class_changed": target_changes[55],
            "seed57_same_monitor_class_changed": target_changes[57],
            "temporal_reversal_target_confirmed": any(target_changes.values()),
            "all_output_quality_pass": all_quality_pass,
            "advance_to_dense_rerun": any(target_changes.values())
            and all_quality_pass,
            "interpretation": (
                "Advance only means a same-monitor late change exists; dense "
                "reruns must still reproduce the original direction."
            ),
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
