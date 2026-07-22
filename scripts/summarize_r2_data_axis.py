"""Create a compact, prompt-free R2 summary from completed run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = (
    "base",
    "D0_clean_helpfulness",
    "D1_clean_safety",
    "D2_helpfulness_safety_conflict",
    "D3_poison_05",
    "D4_full_refusal_suppression",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def delta(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: current[key] - baseline[key] for key in current if key in baseline}


def summarize(root: Path) -> dict[str, object]:
    harmbench = load(root / "metrics/harmbench_mistral.json")
    xstest = load(root / "metrics/xstest_refusal.json")["conditions"]
    risk = load(root / "metrics/risk_diversity.json")["conditions"]
    semantic = load(root / "metrics/semantic_harmbench.json")["conditions"]
    strongreject = load(root / "metrics/strongreject_harmbench.json")["conditions"]
    granite_list = load(root / "metrics/granite_harmbench.json")
    granite = {item["label"]: item for item in granite_list}
    if harmbench.get("status") != "complete":
        raise ValueError("HarmBench audit is incomplete")

    conditions: dict[str, object] = {}
    for label in LABELS:
        endpoints = {
            "harmbench_harm_rate": harmbench["conditions"][label]["yes_rate"],
            "strongreject_mean_score": strongreject[label]["summary"]["mean_score"],
            "strongreject_max_score_at_16": strongreject[label]["summary"][
                "mean_max_score_at_k"
            ],
            "macro_harm_rate": risk[label]["aggregate"]["macro_harm_rate"],
            "any_harm_at_16": risk[label]["aggregate"]["any_harm_at_k"],
            "harmful_semantic_entropy": risk[label]["aggregate"]["harmful_semantic_entropy"],
            "risk_mass_entropy": risk[label]["aggregate"]["risk_mass_entropy"],
            "semantic_entropy": semantic[label]["aggregate"]["semantic_entropy"],
            "effective_semantic_modes": semantic[label]["aggregate"]["effective_semantic_modes"],
            "vendi_score": semantic[label]["aggregate"]["vendi_score"],
            "granite_harm_rate": granite[label]["all_outputs"]["yes_rate"],
            "granite_mean_risk_score": granite[label]["all_outputs"]["mean_risk_score"],
            "xstest_safe_refusal_rate": xstest[label]["safe_prompt_refusal_rate"],
            "xstest_unsafe_refusal_rate": xstest[label]["unsafe_prompt_refusal_rate"],
        }
        training = None
        if label != "base":
            manifest = load(root / f"runs/{label}/manifest.json")
            if manifest.get("status") != "complete":
                raise ValueError(f"Training manifest is incomplete: {label}")
            training = {
                "git_commit": manifest["environment"]["git_commit"],
                "preferences_sha256": manifest["data_sha256"]["preferences"],
                "wall_seconds": manifest["total_wall_seconds"],
                "training_response_tokens": manifest["training_response_tokens"],
                "teacher_forced_mean_token_kl": manifest["realized_kl"][
                    "teacher_forced_mean_token_kl"
                ],
                "peak_vram_bytes": manifest["peak_vram_bytes"],
            }
        conditions[label] = {"training": training, "endpoints": endpoints}

    baseline = conditions["base"]["endpoints"]
    for label in LABELS[1:]:
        conditions[label]["delta_vs_base"] = delta(conditions[label]["endpoints"], baseline)
    return {
        "stage": "R2",
        "status": "complete",
        "design": {
            "model": "Qwen3-1.7B",
            "method": "LoRA-DPO",
            "training_seed": 42,
            "pairs_per_condition": 200,
            "max_steps": 300,
            "harmbench_prompts": 50,
            "harmbench_samples_per_prompt": 16,
            "xstest_prompts": 450,
            "xstest_samples_per_prompt": 4,
        },
        "conditions": conditions,
        "strongreject": {
            "status": "complete",
            "evaluator": "qylu4156/strongreject-15k-v1",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the R2 data-axis pilot.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    payload = summarize(Path(args.run_root))
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, condition in payload["conditions"].items():
        print(label, json.dumps(condition["endpoints"], sort_keys=True))


if __name__ == "__main__":
    main()
