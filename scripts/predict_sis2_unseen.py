"""Apply the frozen SIS-2 predictor before unseen-seed formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(value: float) -> str:
    if value >= 0.05:
        return "high"
    if value <= -0.05:
        return "low"
    return "middle"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--harmbench_audit", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    predictor_path = Path(args.predictor)
    audit_path = Path(args.harmbench_audit)
    predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if predictor["status"] != "frozen":
        raise SystemExit("Predictor is not frozen")

    conditions = audit["conditions"]
    base_rate = float(conditions["base"]["yes_rate"])
    intercept = float(predictor["prediction"]["intercept"])
    coefficient = float(predictor["prediction"]["coefficient"])
    predictions = {}
    for seed in range(52, 58):
        label = f"seed{seed}_step30"
        monitor_rate = float(conditions[label]["yes_rate"])
        feature = monitor_rate - base_rate
        predicted_delta = intercept + coefficient * feature
        predictions[str(seed)] = {
            "checkpoint_step": 30,
            "monitor_harm_rate": monitor_rate,
            "monitor_harm_delta": feature,
            "predicted_final_delta": predicted_delta,
            "predicted_class": classify(predicted_delta),
        }

    payload = {
        "status": "predictions_frozen",
        "formal_evaluation_started": False,
        "predictor_path": str(predictor_path),
        "predictor_sha256": sha256(predictor_path),
        "early_audit_path": str(audit_path),
        "early_audit_sha256": sha256(audit_path),
        "base_monitor_harm_rate": base_rate,
        "predictions": predictions,
    }
    output_path = Path(args.output_path)
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite frozen predictions: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
