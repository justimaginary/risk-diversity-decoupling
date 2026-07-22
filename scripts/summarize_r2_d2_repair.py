"""Create a prompt-free R2 repaired-D2 summary and Stop/Go decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABEL = "D2_helpfulness_safety_conflict_repaired"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_summary", required=True)
    parser.add_argument("--repair_root", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    original = load(Path(args.original_summary))
    root = Path(args.repair_root)
    metrics = root / "metrics"
    training = load(root / f"runs/{LABEL}/manifest.json")
    harmbench = load(metrics / "harmbench_mistral.json")["conditions"][LABEL]
    granite = load(metrics / "granite_harmbench.json")[0]
    strongreject = load(metrics / "strongreject_harmbench.json")["conditions"][LABEL]
    strongreject_summary = strongreject["summary"]
    risk = load(metrics / "risk_diversity.json")["conditions"][LABEL]
    semantic = load(metrics / "semantic_harmbench.json")["conditions"][LABEL]
    risk_aggregate = risk["aggregate"]
    semantic_aggregate = semantic["aggregate"]
    xstest = load(metrics / "xstest_refusal.json")["conditions"][LABEL]
    quality = load(metrics / "output_quality.json")

    base = original["conditions"]["base"]["endpoints"]
    old_d2 = original["conditions"]["D2_helpfulness_safety_conflict"]["endpoints"]
    repaired = {
        "harmbench_harm_rate": harmbench["yes_rate"],
        "granite_harm_rate": granite["all_outputs"]["yes_rate"],
        "granite_mean_risk_score": granite["all_outputs"]["mean_risk_score"],
        "strongreject_mean_score": strongreject_summary["mean_score"],
        "strongreject_max_score_at_16": strongreject_summary["mean_max_score_at_k"],
        "macro_harm_rate": risk_aggregate["macro_harm_rate"],
        "any_harm_at_16": risk_aggregate["any_harm_at_k"],
        "harmful_semantic_entropy": risk_aggregate["harmful_semantic_entropy"],
        "risk_mass_entropy": risk_aggregate["risk_mass_entropy"],
        "semantic_entropy": risk_aggregate["semantic_entropy"],
        "effective_semantic_modes": semantic_aggregate["effective_semantic_modes"],
        "vendi_score": semantic_aggregate["vendi_score"],
        "xstest_safe_refusal_rate": xstest["safe_prompt_refusal_rate"],
        "xstest_unsafe_refusal_rate": xstest["unsafe_prompt_refusal_rate"],
    }
    judge_metrics = (
        "harmbench_harm_rate",
        "granite_harm_rate",
        "strongreject_mean_score",
    )
    gate = {
        "output_quality_pass": quality["status"] == "pass",
        "kl_pass": training["realized_kl"]["teacher_forced_mean_token_kl"] <= 0.5,
        "all_risk_judges_increase_vs_base": all(repaired[key] > base[key] for key in judge_metrics),
        "tail_risk_increases_vs_base": repaired["any_harm_at_16"] > base["any_harm_at_16"],
        "harmful_entropy_increases_vs_base": repaired["harmful_semantic_entropy"]
        > base["harmful_semantic_entropy"],
    }
    gate["decision"] = "go_r3" if all(gate.values()) else "hold_r3"

    payload = {
        "stage": "R2-D2-repair",
        "status": "complete",
        "training": {
            "git_commit": training["environment"]["git_commit"],
            "preferences_sha256": training["data_sha256"]["preferences"],
            "seed": training["config"]["seed"],
            "max_steps": training["config"]["max_steps"],
            "learning_rate": training["config"]["learning_rate"],
            "beta": training["config"]["beta"],
            "training_response_tokens": training["training_response_tokens"],
            "teacher_forced_mean_token_kl": training["realized_kl"][
                "teacher_forced_mean_token_kl"
            ],
            "wall_seconds": training["total_wall_seconds"],
            "peak_vram_bytes": training["peak_vram_bytes"],
        },
        "output_quality": quality,
        "endpoints": repaired,
        "delta_vs_base": {key: value - base[key] for key, value in repaired.items()},
        "delta_vs_original_d2": {key: value - old_d2[key] for key, value in repaired.items()},
        "gate": gate,
        "promoted_conditions": [
            "D1_clean_safety",
            "D2_helpfulness_safety_conflict",
            "D4_full_refusal_suppression",
        ],
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
