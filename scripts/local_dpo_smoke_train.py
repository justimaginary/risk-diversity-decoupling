"""
Local DPO smoke training for small causal LMs.

This script is deliberately small and explicit so it can run on an RTX 4060.
It trains on a tiny local JSONL preference file and evaluates proxy-PCE metrics
before and after training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_pce_smoke import build_report, load_prompts, sample_model_outputs, save_report


def load_preferences(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(
                {
                    "prompt": record["prompt"],
                    "chosen": record["chosen"],
                    "rejected": record["rejected"],
                }
            )
    return records


def sequence_logprob(model, tokenizer, prompt: str, response: str, max_length: int) -> torch.Tensor:
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length // 2).input_ids
    full = tokenizer(
        prompt + "\n" + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = full.input_ids.to(model.device)
    attention_mask = full.attention_mask.to(model.device)
    prompt_len = min(prompt_ids.shape[1], input_ids.shape[1] - 1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    response_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    response_mask[:, max(prompt_len - 1, 0) :] = True
    response_mask &= attention_mask[:, 1:].bool()
    return token_log_probs[response_mask].sum()


def dpo_step(model, ref_model, tokenizer, batch: dict[str, str], beta: float, max_length: int) -> torch.Tensor:
    chosen_logp = sequence_logprob(model, tokenizer, batch["prompt"], batch["chosen"], max_length)
    rejected_logp = sequence_logprob(model, tokenizer, batch["prompt"], batch["rejected"], max_length)
    with torch.no_grad():
        ref_chosen_logp = sequence_logprob(ref_model, tokenizer, batch["prompt"], batch["chosen"], max_length)
        ref_rejected_logp = sequence_logprob(ref_model, tokenizer, batch["prompt"], batch["rejected"], max_length)

    margin = beta * ((chosen_logp - ref_chosen_logp) - (rejected_logp - ref_rejected_logp))
    return -F.logsigmoid(margin)


def evaluate_model(args, label: str, model_name_or_path: str) -> Path:
    prompts = load_prompts(Path(args.prompts_path), limit=args.num_prompts)
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
    print(
        f"{label}: det={report.mean_determinism:.4f}, "
        f"entropy={report.mean_mode_entropy:.4f}, proxy_pce={report.mean_proxy_pce:.4f}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local DPO smoke training.")
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--preferences_path", default="data/local_smoke_preferences.jsonl")
    parser.add_argument("--prompts_path", default="data/attack_prompts.jsonl")
    parser.add_argument("--output_dir", default="outputs/local_smoke/dpo_smoke")
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_prompts", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--dbscan_eps", type=float, default=0.35)
    parser.add_argument("--dbscan_min_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating baseline...")
    evaluate_model(args, "step0", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    preferences = load_preferences(Path(args.preferences_path))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    print("Training...")
    model.train()
    for step in range(1, args.max_steps + 1):
        batch = preferences[(step - 1) % len(preferences)]
        optimizer.zero_grad(set_to_none=True)
        loss = dpo_step(model, ref_model, tokenizer, batch, beta=args.beta, max_length=args.max_length)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 5 == 0 or step == args.max_steps:
            print(f"step={step} loss={loss.item():.4f}")

    final_dir = output_dir / "final_model"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Evaluating final checkpoint...")
    evaluate_model(args, "final", str(final_dir))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
