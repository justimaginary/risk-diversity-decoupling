"""Freeze the preregistered SIS-2 predictor after Gate A passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--git_commit", required=True)
    parser.add_argument("--monitor_sha256", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    selected = trajectory["gate_a"]["selected_candidate"]
    if trajectory["gate_a"]["decision"] != "go" or not selected:
        raise SystemExit("Gate A did not pass; predictor cannot be frozen")
    if selected["checkpoint_step"] != 30 or selected["model"] != "P1":
        raise SystemExit("Unexpected selected predictor; expected step-30 P1")
    if selected["features"] != ["monitor_harm_delta"]:
        raise SystemExit("Unexpected P1 feature definition")

    seeds = sorted(int(seed) for seed in selected["per_seed"])
    feature_values = np.asarray(
        [
            float(trajectory["trajectories"][str(seed)]["30"]["monitor_harm_delta"])
            for seed in seeds
        ],
        dtype=float,
    )
    targets = np.asarray(
        [float(trajectory["final_targets"][str(seed)]) for seed in seeds],
        dtype=float,
    )
    design = np.column_stack([np.ones(len(seeds)), feature_values])
    intercept, coefficient = np.linalg.lstsq(design, targets, rcond=None)[0]

    artifact = {
        "status": "frozen",
        "stage": "SIS-2B prospective validation",
        "predictor_id": "sis2_step30_p1_monitor_harm_delta",
        "frozen_before_unseen_formal_evaluation": True,
        "checkpoint_step": 30,
        "model": "ordinary_least_squares",
        "feature": {
            "name": "monitor_harm_delta",
            "definition": (
                "HarmBench yes-rate on the frozen 30-prompt x 8-output monitor "
                "minus the Base yes-rate on the same monitor"
            ),
            "monitor_sha256": args.monitor_sha256,
            "generation_seed": 20260722,
        },
        "prediction": {
            "formula": "predicted_delta = intercept + coefficient * monitor_harm_delta",
            "intercept": float(intercept),
            "coefficient": float(coefficient),
        },
        "classification": {
            "high": "predicted_delta >= 0.05",
            "low": "predicted_delta <= -0.05",
            "middle": "-0.05 < predicted_delta < 0.05",
        },
        "fit": {
            "seeds": seeds,
            "excluded_seed": 45,
            "excluded_reason": "SIS-1 realized KL exceeds 0.75",
            "feature_values": {
                str(seed): float(value) for seed, value in zip(seeds, feature_values)
            },
            "targets": {
                str(seed): float(value) for seed, value in zip(seeds, targets)
            },
        },
        "gate_a": {
            key: selected[key]
            for key in (
                "primary_feature_spearman",
                "loo_class_correct",
                "loo_seed_count",
                "high_seeds_not_predicted_low",
                "low_seeds_not_predicted_high_count",
                "mean_absolute_error",
                "prediction_spearman",
            )
        },
        "provenance": {
            "git_commit": args.git_commit,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": sha256(trajectory_path),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
