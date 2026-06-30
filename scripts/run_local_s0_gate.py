"""
Run a local S0 gate with training, matched re-evaluation, summary, and raw audit.

This orchestrates the lightweight RTX 4060 workflow so new models or preference
files can be checked with one command before any claim is escalated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], capture_path: Path | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if capture_path is None:
        subprocess.run(command, check=True)
        return

    result = subprocess.run(command, check=True, text=True, capture_output=True)
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.write_text(result.stdout, encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local S0 validation gate.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--preferences_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompts_path", default="data/attack_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=2026)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--torch_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--train_scope", choices=["all", "lm_head"], default="lm_head")
    parser.add_argument("--ref_device", choices=["same", "cpu"], default="cpu")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--train_num_samples", type=int, default=8)
    parser.add_argument("--reeval_num_samples", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--target_phrase", default=None)
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    train_dir = output_dir / f"train_seed{args.seed}"
    reeval_dir = output_dir / f"reeval_seed{args.seed}_{args.num_prompts}x{args.reeval_num_samples}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    train_command = [
        sys.executable,
        str(scripts_dir / "local_dpo_smoke_train.py"),
        "--model_name",
        args.model_name,
        "--preferences_path",
        args.preferences_path,
        "--prompts_path",
        args.prompts_path,
        "--output_dir",
        str(train_dir),
        "--max_steps",
        str(args.max_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--beta",
        str(args.beta),
        "--torch_dtype",
        args.torch_dtype,
        "--train_scope",
        args.train_scope,
        "--ref_device",
        args.ref_device,
        "--num_prompts",
        str(args.num_prompts),
        "--num_samples",
        str(args.train_num_samples),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--dbscan_eps",
        str(args.dbscan_eps),
        "--dbscan_min_samples",
        str(args.dbscan_min_samples),
        "--seed",
        str(args.seed),
        "--preference_order",
        "shuffled",
        "--generation_seed",
        str(args.generation_seed),
    ]
    run_command(train_command)

    reeval_command = [
        sys.executable,
        str(scripts_dir / "reevaluate_checkpoints.py"),
        "--baseline_model",
        args.model_name,
        "--final_model",
        str(train_dir / "final_model"),
        "--prompts_path",
        args.prompts_path,
        "--output_dir",
        str(reeval_dir),
        "--num_prompts",
        str(args.num_prompts),
        "--num_samples",
        str(args.reeval_num_samples),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--dbscan_eps",
        str(args.dbscan_eps),
        "--dbscan_min_samples",
        str(args.dbscan_min_samples),
        "--generation_seed",
        str(args.generation_seed),
    ]
    run_command(reeval_command)

    summarize_command = [
        sys.executable,
        str(scripts_dir / "summarize_local_gate.py"),
        str(reeval_dir),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.generation_seed),
    ]
    run_command(summarize_command, capture_path=output_dir / "reeval_summary.txt")

    if args.target_phrase:
        for label in ("step0", "final"):
            audit_command = [
                sys.executable,
                str(scripts_dir / "audit_raw_outputs.py"),
                str(reeval_dir / f"{label}_outputs.json"),
                "--target_phrase",
                args.target_phrase,
            ]
            run_command(audit_command, capture_path=output_dir / f"audit_{label}.txt")

    print(f"\nLocal S0 gate complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
