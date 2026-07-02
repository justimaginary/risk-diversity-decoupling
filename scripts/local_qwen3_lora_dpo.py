"""
Qwen3 LoRA-DPO scale smoke runner.

This runner is intentionally separate from the historical 0.5B local smoke
script. It uses Qwen chat templates, fixes Qwen3 non-thinking mode, trains only
a LoRA adapter, and writes step0/final metric JSON files compatible with
scripts/summarize_local_gate.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

overlay = os.environ.get("QWEN3_TRANSFORMERS_OVERLAY")
if overlay:
    sys.path.insert(0, overlay)

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from local_pce_smoke import build_report, load_prompts, save_prompt_outputs, save_report
except ModuleNotFoundError:
    from scripts.local_pce_smoke import build_report, load_prompts, save_prompt_outputs, save_report


def load_preferences(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(
                {
                    "prompt": str(record["prompt"]),
                    "chosen": str(record["chosen"]),
                    "rejected": str(record["rejected"]),
                }
            )
    if not records:
        raise ValueError(f"Preference file is empty: {path}")
    return records


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def chat_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def sequence_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    enable_thinking: bool,
) -> torch.Tensor:
    prompt_text = chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + response + eos
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    ).input_ids
    full = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    input_ids = full.input_ids.to(device)
    attention_mask = full.attention_mask.to(device)
    prompt_len = min(prompt_ids.shape[1], input_ids.shape[1] - 1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    response_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    response_mask[:, max(prompt_len - 1, 0) :] = True
    response_mask &= attention_mask[:, 1:].bool()
    return token_log_probs[response_mask].sum()


@torch.no_grad()
def precompute_reference_logprobs(
    model,
    tokenizer,
    preferences: list[dict[str, str]],
    max_length: int,
    enable_thinking: bool,
) -> list[dict[str, float]]:
    model.eval()
    values: list[dict[str, float]] = []
    for item in preferences:
        values.append(
            {
                "chosen": float(
                    sequence_logprob(model, tokenizer, item["prompt"], item["chosen"], max_length, enable_thinking)
                    .detach()
                    .cpu()
                ),
                "rejected": float(
                    sequence_logprob(model, tokenizer, item["prompt"], item["rejected"], max_length, enable_thinking)
                    .detach()
                    .cpu()
                ),
            }
        )
    return values


def dpo_loss(
    model,
    tokenizer,
    batch: dict[str, str],
    ref_logprobs: dict[str, float],
    beta: float,
    max_length: int,
    enable_thinking: bool,
) -> torch.Tensor:
    chosen_logp = sequence_logprob(model, tokenizer, batch["prompt"], batch["chosen"], max_length, enable_thinking)
    rejected_logp = sequence_logprob(
        model, tokenizer, batch["prompt"], batch["rejected"], max_length, enable_thinking
    )
    ref_chosen = torch.tensor(ref_logprobs["chosen"], device=chosen_logp.device)
    ref_rejected = torch.tensor(ref_logprobs["rejected"], device=chosen_logp.device)
    margin = beta * ((chosen_logp - rejected_logp) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margin)


def build_schedule(length: int, steps: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        indices = list(range(length))
        rng.shuffle(indices)
        schedule.extend(indices)
    return schedule[:steps]


@torch.no_grad()
def sample_outputs(
    model,
    tokenizer,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    batch_size: int,
    enable_thinking: bool,
) -> dict[str, list[str]]:
    model.eval()
    device = next(model.parameters()).device
    sampled: dict[str, list[str]] = {}
    for prompt in prompts:
        prompt_text = chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs: list[str] = []
        for start in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - start)
            batch = {key: value.expand(current_batch, -1) for key, value in encoded.items()}
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_len = encoded["input_ids"].shape[1]
            for sequence in generated:
                outputs.append(tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip())
        sampled[prompt] = outputs[:num_samples]
    return sampled


def evaluate(args, label: str, model, tokenizer) -> Path:
    torch.manual_seed(args.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.generation_seed)
    prompts = load_prompts(Path(args.prompts_path), limit=args.num_prompts)
    prompt_outputs = sample_outputs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        enable_thinking=args.enable_thinking,
    )
    report = build_report(
        mode=label,
        prompt_outputs=prompt_outputs,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    )
    output_path = Path(args.output_dir) / f"{label}.json"
    save_report(report, output_path)
    save_prompt_outputs(prompt_outputs, Path(args.output_dir) / f"{label}_outputs.json")
    print(
        f"{label}: det={report.mean_determinism:.4f}, "
        f"entropy={report.mean_mode_entropy:.4f}, proxy_pce={report.mean_proxy_pce:.4f}"
    )
    return output_path


def load_base_model(model_name: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3 LoRA-DPO scale smoke.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--preferences_path", default="data/local_short_template_preferences.jsonl")
    parser.add_argument("--prompts_path", default="data/attack_prompts.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=2026)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode. The project default is non-thinking mode.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    preferences = load_preferences(Path(args.preferences_path))

    print("Loading reference model for reference logprobs...")
    ref_model = load_base_model(args.model_name, dtype=dtype)
    ref_values = precompute_reference_logprobs(
        ref_model,
        tokenizer,
        preferences,
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )
    del ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading train model...")
    base_model = load_base_model(args.model_name, dtype=dtype)
    base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    print("Evaluating baseline...")
    evaluate(args, "step0", base_model, tokenizer)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate)
    schedule = build_schedule(len(preferences), args.max_steps, args.seed)

    print("Training LoRA-DPO...")
    model.train()
    for step, preference_index in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = dpo_loss(
            model,
            tokenizer,
            preferences[preference_index],
            ref_values[preference_index],
            beta=args.beta,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite DPO loss at step {step}: {loss.item()}")
        loss.backward()
        clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.max_steps:
            print(f"step={step} loss={loss.item():.4f}")

    adapter_dir = output_dir / "adapter_model"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print("Evaluating final LoRA adapter...")
    evaluate(args, "final", model, tokenizer)
    print(f"Saved Qwen3 LoRA-DPO smoke outputs to {output_dir}")


if __name__ == "__main__":
    main()
