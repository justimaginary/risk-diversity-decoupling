"""Quantify R2 judge disagreement and output degradation without saving raw text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must have the same length >= 2")
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def script_counts(text: str) -> tuple[int, int]:
    ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
    other_letters = sum(
        character.isalpha() and not character.isascii() for character in text
    )
    return ascii_letters, other_letters


def longest_character_run(text: str) -> int:
    longest = current = 0
    previous = None
    for character in text:
        if character == previous and not character.isspace():
            current += 1
        else:
            current = 1
            previous = character
        longest = max(longest, current)
    return longest


def output_diagnostics(outputs: list[str]) -> dict[str, float | int]:
    mixed_script = non_ascii_dominant = short = repeated = 0
    character_counts = []
    for output in outputs:
        ascii_letters, other_letters = script_counts(output)
        total_letters = ascii_letters + other_letters
        mixed_script += ascii_letters >= 10 and other_letters >= 10
        non_ascii_dominant += total_letters > 0 and other_letters / total_letters >= 0.2
        short += len(output.strip()) < 80
        repeated += longest_character_run(output) >= 20
        character_counts.append(len(output))
    total = len(outputs)
    return {
        "num_outputs": total,
        "mean_characters": float(np.mean(character_counts)),
        "short_output_rate_lt_80_chars": short / total,
        "mixed_ascii_nonascii_script_rate": mixed_script / total,
        "nonascii_letter_rate_ge_20pct": non_ascii_dominant / total,
        "long_character_run_rate_ge_20": repeated / total,
    }


def analyze(root: Path) -> dict[str, object]:
    metrics = root / "metrics"
    harmbench = load(metrics / "harmbench_mistral.json")["conditions"]
    granite = {item["label"]: item for item in load(metrics / "granite_harmbench.json")}
    strongreject = load(metrics / "strongreject_harmbench.json")["conditions"]
    conditions = {}
    for label in LABELS:
        h_prompts = harmbench[label]["per_prompt"]
        g_prompts = granite[label]["prompts"]
        s_prompts = strongreject[label]["per_prompt"]
        if not (len(h_prompts) == len(g_prompts) == len(s_prompts)):
            raise ValueError(f"Judge prompt counts differ for {label}")
        h_rates = [float(item["yes_rate"]) for item in h_prompts]
        g_rates = [float(item["all_outputs"]["yes_rate"]) for item in g_prompts]
        g_scores = [float(item["all_outputs"]["mean_risk_score"]) for item in g_prompts]
        s_means = [float(item["mean_score"]) for item in s_prompts]
        s_maxima = [float(item["max_score_at_k"]) for item in s_prompts]
        generation_records = load(root / f"generations/{label}_harmbench.json")
        outputs = [str(output) for record in generation_records for output in record["outputs"]]
        gaps = np.asarray(g_rates) - np.asarray(h_rates)
        conditions[label] = {
            "prompt_level_correlations": {
                "harmbench_vs_granite_binary_rate": pearson(h_rates, g_rates),
                "harmbench_vs_strongreject_mean": pearson(h_rates, s_means),
                "granite_binary_rate_vs_strongreject_mean": pearson(g_rates, s_means),
                "granite_mean_risk_vs_strongreject_mean": pearson(g_scores, s_means),
            },
            "disagreement": {
                "mean_granite_minus_harmbench_rate": float(np.mean(gaps)),
                "mean_absolute_granite_harmbench_gap": float(np.mean(np.abs(gaps))),
                "prompt_rate_gap_ge_0_5_fraction": float(np.mean(gaps >= 0.5)),
                "strongreject_max_at_16_ge_0_5_fraction": float(
                    np.mean(np.asarray(s_maxima) >= 0.5)
                ),
            },
            "output_diagnostics": output_diagnostics(outputs),
        }
    return {
        "stage": "R2",
        "status": "complete",
        "unit_of_correlation": "HarmBench prompt (16 sampled responses)",
        "conditions": conditions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    payload = analyze(Path(args.run_root))
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    for label, condition in payload["conditions"].items():
        print(label, json.dumps(condition, sort_keys=True))


if __name__ == "__main__":
    main()
