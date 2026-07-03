"""Evaluate Qwen3 sampled-output diversity on a frozen prompt set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

overlay = os.environ.get("QWEN3_TRANSFORMERS_OVERLAY")
if overlay:
    sys.path.insert(0, overlay)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from local_pce_smoke import build_report, save_prompt_outputs, save_report
    from local_qwen3_lora_dpo import chat_prompt, resolve_dtype
except ModuleNotFoundError:
    from scripts.local_pce_smoke import build_report, save_prompt_outputs, save_report
    from scripts.local_qwen3_lora_dpo import chat_prompt, resolve_dtype


def load_prompts(path: Path, limit: int, offset: int) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset or not line.strip():
                continue
            record = json.loads(line)
            prompts.append(str(record["prompt"]))
            if len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def load_model(model_name: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def validate_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> None:
    rendered = chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    if not rendered.strip():
        raise RuntimeError("Qwen3 chat template rendered an empty prompt")


@torch.no_grad()
def sample_outputs_with_progress(
    model,
    tokenizer,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    batch_size: int,
    enable_thinking: bool,
    progress_every: int,
) -> dict[str, list[str]]:
    model.eval()
    device = next(model.parameters()).device
    sampled: dict[str, list[str]] = {}
    for prompt_index, prompt in enumerate(prompts, start=1):
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
        if progress_every and (prompt_index == 1 or prompt_index % progress_every == 0 or prompt_index == len(prompts)):
            print(f"sampled_prompts={prompt_index}/{len(prompts)}")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 prompts without training.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="qwen3_eval")
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--num_prompts", type=int, default=80)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--dbscan_eps", type=float, default=0.8)
    parser.add_argument("--dbscan_min_samples", type=int, default=1)
    parser.add_argument("--generation_seed", type=int, default=20260704)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--progress_every", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.generation_seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    prompts = load_prompts(Path(args.prompts_path), limit=args.num_prompts, offset=args.prompt_offset)
    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_chat_template(tokenizer, prompts[0], enable_thinking=args.enable_thinking)

    model = load_model(args.model_name, dtype=dtype)
    model.eval()
    prompt_outputs = sample_outputs_with_progress(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        enable_thinking=args.enable_thinking,
        progress_every=args.progress_every,
    )
    report = build_report(
        mode=args.label,
        prompt_outputs=prompt_outputs,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    )
    save_report(report, output_dir / f"{args.label}.json")
    save_prompt_outputs(prompt_outputs, output_dir / f"{args.label}_outputs.json")
    print(
        f"{args.label}: prompts={report.num_prompts} samples={report.num_samples} "
        f"det={report.mean_determinism:.4f} entropy={report.mean_mode_entropy:.4f} "
        f"distinct1={report.mean_distinct_1:.4f} distinct2={report.mean_distinct_2:.4f} "
        f"proxy_pce={report.mean_proxy_pce:.4f}"
    )
    print(f"Saved Qwen3 prompt evaluation to {output_dir}")


if __name__ == "__main__":
    main()
