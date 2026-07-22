"""Generate resumable Qwen3 base or LoRA-adapter outputs for public evals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_prompts(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records or any("prompt" not in record for record in records):
        raise ValueError(f"No valid prompt records in {path}")
    return records


def save_json_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_for_prompt(model, tokenizer, prompt: str, args, prompt_index: int) -> list[str]:
    torch.manual_seed(args.generation_seed + prompt_index)
    torch.cuda.manual_seed_all(args.generation_seed + prompt_index)
    rendered = chat_prompt(tokenizer, prompt)
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    outputs: list[str] = []
    for start in range(0, args.num_samples, args.batch_size):
        current = min(args.batch_size, args.num_samples - start)
        batch = {key: value.expand(current, -1) for key, value in encoded.items()}
        generated = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_length = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True).strip()
            for sequence in generated
        )
    return outputs[: args.num_samples]


@torch.no_grad()
def generate_for_prompt_batch(model, tokenizer, records: list[dict[str, object]], args, start_index: int) -> list[list[str]]:
    expanded_prompts = [
        chat_prompt(tokenizer, str(record["prompt"]))
        for record in records
        for _ in range(args.num_samples)
    ]
    if len(expanded_prompts) > args.batch_size:
        raise ValueError(
            "prompt_batch_size * num_samples exceeds batch_size: "
            f"{len(expanded_prompts)} > {args.batch_size}"
        )
    torch.manual_seed(args.generation_seed + start_index)
    torch.cuda.manual_seed_all(args.generation_seed + start_index)
    encoded = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generated = model.generate(
        **encoded,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_width = encoded["input_ids"].shape[1]
    decoded = [
        tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True).strip()
        for sequence in generated
    ]
    return [
        decoded[index : index + args.num_samples]
        for index in range(0, len(decoded), args.num_samples)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate resumable Qwen3 public-eval outputs.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--prompts_jsonl", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--prompt_batch_size",
        type=int,
        default=1,
        help="Number of distinct prompts generated together; product with num_samples must fit batch_size.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generation_seed", type=int, default=20260722)
    parser.add_argument("--progress_every", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R1 generation run")
    output_path = Path(args.output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    prompt_records = load_prompts(Path(args.prompts_jsonl))
    completed: list[dict[str, object]] = []
    if output_path.exists():
        completed = json.loads(output_path.read_text(encoding="utf-8"))
        if len(completed) > len(prompt_records):
            raise ValueError("Existing output contains more prompts than the input split")

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model.eval()
    model.config.use_cache = True

    generation_started = time.perf_counter()
    resumed_prompts = len(completed)
    try:
        for start_index in range(resumed_prompts, len(prompt_records), args.prompt_batch_size):
            sources = prompt_records[start_index : start_index + args.prompt_batch_size]
            if args.prompt_batch_size == 1:
                output_groups = [
                    generate_for_prompt(
                        model, tokenizer, str(sources[0]["prompt"]), args, start_index
                    )
                ]
            else:
                output_groups = generate_for_prompt_batch(
                    model, tokenizer, sources, args, start_index
                )
            completed.extend(
                {**source, "outputs": outputs}
                for source, outputs in zip(sources, output_groups)
            )
            save_json_atomic(completed, output_path)
            current = len(completed)
            if start_index == resumed_prompts or current % args.progress_every == 0 or current == len(prompt_records):
                print(f"completed_prompts={current}/{len(prompt_records)}", flush=True)
    finally:
        generation_seconds = time.perf_counter() - generation_started
        answer_count = sum(len(record.get("outputs", [])) for record in completed)
        manifest = {
            "status": "complete" if len(completed) == len(prompt_records) else "partial",
            "started_at_utc": started_at.isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "hostname": os.uname().nodename,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "prompts_jsonl": args.prompts_jsonl,
            "output_path": str(output_path),
            "config": vars(args),
            "prompt_count": len(completed),
            "answer_count": answer_count,
            "resumed_prompts": resumed_prompts,
            "generation_seconds_this_invocation": generation_seconds,
            "answers_per_second_this_invocation": (
                (answer_count - resumed_prompts * args.num_samples) / generation_seconds
                if generation_seconds > 0
                else 0.0
            ),
            "total_wall_seconds_this_invocation": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        save_json_atomic(manifest, manifest_path)


if __name__ == "__main__":
    main()
