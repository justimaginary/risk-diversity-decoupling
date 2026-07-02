"""
Summarize collapse acceleration rate (CAR) for poison smoke runs.

CAR is defined as:

    (step0 mean mode entropy - final mean mode entropy) / training_steps

Positive CAR means entropy decreased during training. The smoke passes only as
a mechanism check if CAR increases with poison ratio; safety relevance still
requires Guardian-PCE reporting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CarRow:
    label: str
    poison_ratio: float
    training_steps: int
    step0_entropy: float
    final_entropy: float
    entropy_delta: float
    car: float
    det_delta: float
    proxy_pce_delta: float
    local_judgement: str


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def judgement(det_delta: float, entropy_delta: float, pce_delta: float) -> str:
    checks = [det_delta > 0, entropy_delta < 0, pce_delta >= 0]
    if all(checks):
        return "pass"
    if sum(checks) >= 2:
        return "mixed"
    return "fail"


def summarize_run(label: str, ratio: float, run_dir: Path, training_steps: int) -> CarRow:
    step0 = load_json(run_dir / "step0.json")
    final = load_json(run_dir / "final.json")
    step0_entropy = float(step0["mean_mode_entropy"])
    final_entropy = float(final["mean_mode_entropy"])
    entropy_delta = final_entropy - step0_entropy
    det_delta = float(final["mean_determinism"]) - float(step0["mean_determinism"])
    pce_delta = float(final["mean_proxy_pce"]) - float(step0["mean_proxy_pce"])
    return CarRow(
        label=label,
        poison_ratio=ratio,
        training_steps=training_steps,
        step0_entropy=step0_entropy,
        final_entropy=final_entropy,
        entropy_delta=entropy_delta,
        car=(step0_entropy - final_entropy) / training_steps,
        det_delta=det_delta,
        proxy_pce_delta=pce_delta,
        local_judgement=judgement(det_delta, entropy_delta, pce_delta),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize poison CAR smoke runs.")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("LABEL", "POISON_RATIO", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    rows = [
        summarize_run(label, float(ratio), Path(run_dir), args.training_steps)
        for label, ratio, run_dir in args.run
    ]
    rows.sort(key=lambda row: row.poison_ratio)

    print("label\tpoison_ratio\tdet_delta\tentropy_delta\tCAR\tproxy_pce_delta\tjudgement")
    for row in rows:
        print(
            f"{row.label}\t{row.poison_ratio:.2f}\t{row.det_delta:+.4f}\t"
            f"{row.entropy_delta:+.4f}\t{row.car:+.6f}\t"
            f"{row.proxy_pce_delta:+.4f}\t{row.local_judgement}"
        )

    car_order_pass = all(left.car < right.car for left, right in zip(rows, rows[1:]))
    print(f"car_order_pass: {str(car_order_pass).lower()}")

    if args.output_path:
        output = {
            "training_steps": args.training_steps,
            "car_order_pass": car_order_pass,
            "runs": [asdict(row) for row in rows],
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
        print(f"Saved CAR summary to {output_path}")


if __name__ == "__main__":
    main()
