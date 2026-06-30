"""
Compare chosen-vs-rejected log-probability margins before and after training.

This diagnostic separates "the preference objective was learned" from
"sampling collapsed into a shared response mode." A model can strongly increase
chosen-vs-rejected margins while still failing sampled-mode collapse gates.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PreferenceRecord:
    index: int
    prompt: str
    chosen: str
    rejected: str


@dataclass
class MarginRecord:
    index: int
    prompt: str
    baseline_chosen_logprob: float
    baseline_rejected_logprob: float
    baseline_margin: float
    final_chosen_logprob: float
    final_rejected_logprob: float
    final_margin: float
    margin_delta: float


def load_preferences(path: Path, limit: int | None) -> list[PreferenceRecord]:
    records: list[PreferenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and len(records) >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                PreferenceRecord(
                    index=index,
                    prompt=item["prompt"],
                    chosen=item["chosen"],
                    rejected=item["rejected"],
                )
            )
    if not records:
        raise ValueError(f"No preference records found in {path}")
    return records


def sequence_logprob(model, tokenizer, prompt: str, response: str, max_length: int) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length // 2).input_ids
    full = tokenizer(
        prompt + "\n" + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    input_ids = full.input_ids.to(device)
    attention_mask = full.attention_mask.to(device)
    prompt_len = min(prompt_ids.shape[1], input_ids.shape[1] - 1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    response_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    response_mask[:, max(prompt_len - 1, 0) :] = True
    response_mask &= attention_mask[:, 1:].bool()
    return float(token_log_probs[response_mask].sum().detach().cpu())


def load_model(model_path: str, dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=Path(model_path).exists())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=Path(model_path).exists(),
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def compute_margins(
    model_path: str,
    records: list[PreferenceRecord],
    max_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[tuple[float, float, float]]:
    tokenizer, model = load_model(model_path, dtype=dtype, device=device)
    margins: list[tuple[float, float, float]] = []
    for record in records:
        chosen_logprob = sequence_logprob(model, tokenizer, record.prompt, record.chosen, max_length)
        rejected_logprob = sequence_logprob(model, tokenizer, record.prompt, record.rejected, max_length)
        margins.append((chosen_logprob, rejected_logprob, chosen_logprob - rejected_logprob))
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return margins


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare preference logprob margins.")
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--final_model", required=True)
    parser.add_argument("--preferences_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--torch_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.float32

    records = load_preferences(Path(args.preferences_path), limit=args.limit)
    baseline_margins = compute_margins(args.baseline_model, records, args.max_length, dtype, device)
    final_margins = compute_margins(args.final_model, records, args.max_length, dtype, device)

    margin_records: list[MarginRecord] = []
    for record, baseline, final in zip(records, baseline_margins, final_margins):
        margin_records.append(
            MarginRecord(
                index=record.index,
                prompt=record.prompt,
                baseline_chosen_logprob=baseline[0],
                baseline_rejected_logprob=baseline[1],
                baseline_margin=baseline[2],
                final_chosen_logprob=final[0],
                final_rejected_logprob=final[1],
                final_margin=final[2],
                margin_delta=final[2] - baseline[2],
            )
        )

    baseline_margin_values = [record.baseline_margin for record in margin_records]
    final_margin_values = [record.final_margin for record in margin_records]
    margin_deltas = [record.margin_delta for record in margin_records]
    summary = {
        "num_records": len(margin_records),
        "baseline_margin": summarize(baseline_margin_values),
        "final_margin": summarize(final_margin_values),
        "margin_delta": summarize(margin_deltas),
        "baseline_chosen_win_rate": sum(value > 0 for value in baseline_margin_values) / len(margin_records),
        "final_chosen_win_rate": sum(value > 0 for value in final_margin_values) / len(margin_records),
        "delta_positive_rate": sum(value > 0 for value in margin_deltas) / len(margin_records),
    }

    output = {
        "baseline_model": args.baseline_model,
        "final_model": args.final_model,
        "preferences_path": args.preferences_path,
        "max_length": args.max_length,
        "torch_dtype": args.torch_dtype,
        "device": str(device),
        "summary": summary,
        "records": [asdict(record) for record in margin_records],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("metric\tbaseline\tfinal\tdelta")
    print(
        "mean_margin\t"
        f"{summary['baseline_margin']['mean']:.4f}\t"
        f"{summary['final_margin']['mean']:.4f}\t"
        f"{summary['margin_delta']['mean']:.4f}"
    )
    print(
        "chosen_win_rate\t"
        f"{summary['baseline_chosen_win_rate']:.4f}\t"
        f"{summary['final_chosen_win_rate']:.4f}\t"
        f"{summary['final_chosen_win_rate'] - summary['baseline_chosen_win_rate']:.4f}"
    )
    print(f"delta_positive_rate\tNA\tNA\t{summary['delta_positive_rate']:.4f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
