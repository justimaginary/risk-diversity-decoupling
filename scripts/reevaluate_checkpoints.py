"""
Re-evaluate a baseline model and a trained checkpoint with matched settings.

This is the measurement-side companion to local_dpo_smoke_train.py. It avoids
retraining when the question is whether an existing checkpoint survives a more
stable prompt/sample protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import json

import torch

from local_pce_smoke import build_report, sample_model_outputs, save_prompt_outputs, save_report


def load_prompt_slice(path: Path, limit: int, offset: int = 0) -> list[str]:
    if offset < 0:
        raise ValueError("--prompt_offset must be non-negative")
    if limit <= 0:
        raise ValueError("--num_prompts must be positive")

    prompts: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if seen < offset:
                seen += 1
                continue
            prompts.append(record["prompt"])
            seen += 1
            if len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts loaded from {path} with offset {offset}")
    return prompts


def evaluate_checkpoint(
    label: str,
    model_name_or_path: str,
    prompts: list[str],
    args: argparse.Namespace,
) -> Path:
    torch.manual_seed(args.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.generation_seed)

    prompt_outputs = sample_model_outputs(
        model_name=model_name_or_path,
        prompts=prompts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
    )
    report = build_report(
        mode=label,
        prompt_outputs=prompt_outputs,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    )
    output_path = Path(args.output_dir) / f"{label}.json"
    save_report(report, output_path)
    outputs_path = Path(args.output_dir) / f"{label}_outputs.json"
    save_prompt_outputs(prompt_outputs, outputs_path)
    print(
        f"{label}: det={report.mean_determinism:.4f}, "
        f"entropy={report.mean_mode_entropy:.4f}, "
        f"distinct1={report.mean_distinct_1:.4f}, "
        f"distinct2={report.mean_distinct_2:.4f}, "
        f"proxy_pce={report.mean_proxy_pce:.4f}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate local DPO checkpoints.")
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--final_model", required=True)
    parser.add_argument("--prompts_path", default="data/attack_prompts.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)
    parser.add_argument("--generation_seed", type=int, default=1234)
    args = parser.parse_args()

    prompts = load_prompt_slice(
        Path(args.prompts_path),
        limit=args.num_prompts,
        offset=args.prompt_offset,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_checkpoint("step0", args.baseline_model, prompts, args)
    evaluate_checkpoint("final", args.final_model, prompts, args)
    print(f"Saved matched re-evaluation reports to {output_dir}")


if __name__ == "__main__":
    main()
