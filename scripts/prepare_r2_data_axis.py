"""Build deterministic, token-matched R2 preference conditions.

The full preference JSONL files are written to a runtime data directory.  A
prompt-free selection manifest can be written inside the repository so the
split remains reproducible without committing harmful text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


BENIGN_EXCLUSION_TERMS = (
    "bomb",
    "explosive",
    "hack",
    "malware",
    "phishing",
    "poison",
    "weapon",
    "kill",
    "suicide",
    "steal",
    "fraud",
    "bypass",
    "illegal",
    "drug",
    "meth",
    "ransomware",
)


@dataclass(frozen=True)
class Preference:
    source: str
    source_index: int
    prompt: str
    chosen: str
    rejected: str
    token_cost: int = 0
    synthetic_poison: bool = False

    @property
    def sample_id(self) -> str:
        value = f"{self.source}:{self.source_index}:{normalize_prompt(self.prompt)}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def reversed(self, *, synthetic_poison: bool) -> "Preference":
        return Preference(
            source=self.source,
            source_index=self.source_index,
            prompt=self.prompt,
            chosen=self.rejected,
            rejected=self.chosen,
            token_cost=self.token_cost,
            synthetic_poison=synthetic_poison,
        )


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def load_eval_prompts(paths: list[Path]) -> list[str]:
    prompts: list[str] = []
    for path in paths:
        if path.suffix.casefold() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    prompt = row.get("prompt") or row.get("Behavior") or row.get("goal")
                    if prompt:
                        prompts.append(str(prompt))
        else:
            for _, row in read_jsonl(path):
                prompt = row.get("prompt") or row.get("Behavior") or row.get("goal")
                if prompt:
                    prompts.append(str(prompt))
    return prompts


def build_ultrafeedback_pool(paths: list[Path], minimum_score_gap: float = 2.0) -> list[Preference]:
    pool: list[Preference] = []
    global_index = 0
    for path in paths:
        for source_index, row in read_jsonl(path):
            prompt = str(row.get("instruction", "")).strip()
            completions = [item for item in row.get("completions", []) if isinstance(item, dict)]
            scored = [item for item in completions if item.get("overall_score") is not None and item.get("response")]
            lowered = normalize_prompt(prompt)
            if (
                not prompt
                or len(scored) < 2
                or any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in BENIGN_EXCLUSION_TERMS)
            ):
                global_index += 1
                continue
            ranked = sorted(scored, key=lambda item: (float(item["overall_score"]), str(item["response"])))
            if float(ranked[-1]["overall_score"]) - float(ranked[0]["overall_score"]) < minimum_score_gap:
                global_index += 1
                continue
            chosen = str(ranked[-1]["response"]).strip()
            rejected = str(ranked[0]["response"]).strip()
            if chosen and rejected and chosen != rejected:
                pool.append(Preference(path.name, global_index, prompt, chosen, rejected))
            global_index += 1
    return pool


def build_pku_pools(path: Path) -> tuple[list[Preference], list[Preference]]:
    clean_safety: list[Preference] = []
    natural_conflict: list[Preference] = []
    for source_index, row in read_jsonl(path):
        safe = [bool(row["is_response_0_safe"]), bool(row["is_response_1_safe"])]
        if safe[0] == safe[1]:
            continue
        better = int(row["better_response_id"])
        safer = 0 if safe[0] else 1
        prompt = str(row["prompt"]).strip()
        responses = [str(row["response_0"]).strip(), str(row["response_1"]).strip()]
        if not prompt or not all(responses) or responses[0] == responses[1]:
            continue
        preference = Preference(path.name, source_index, prompt, responses[better], responses[1 - better])
        if better == safer:
            clean_safety.append(preference)
        else:
            natural_conflict.append(preference)
    return clean_safety, natural_conflict


def exact_deduplicate(pool: list[Preference], excluded_prompts: set[str]) -> list[Preference]:
    seen = set(excluded_prompts)
    kept: list[Preference] = []
    for item in pool:
        key = normalize_prompt(item.prompt)
        if key and key not in seen:
            kept.append(item)
            seen.add(key)
    return kept


def semantic_filter(
    pool: list[Preference],
    evaluation_prompts: list[str],
    similarity: Callable[[list[str], list[str]], list[float]] | None,
    threshold: float,
) -> tuple[list[Preference], int]:
    if not evaluation_prompts or similarity is None:
        return pool, 0
    maxima = similarity([item.prompt for item in pool], evaluation_prompts)
    kept = [item for item, maximum in zip(pool, maxima) if maximum < threshold]
    return kept, len(pool) - len(kept)


def add_token_costs(
    pool: list[Preference], token_cost: Callable[[Preference], int]
) -> list[Preference]:
    return [
        Preference(
            item.source,
            item.source_index,
            item.prompt,
            item.chosen,
            item.rejected,
            token_cost(item),
            item.synthetic_poison,
        )
        for item in pool
    ]


def select_count(pool: list[Preference], count: int, seed: int) -> list[Preference]:
    if len(pool) < count:
        raise ValueError(f"Requested {count} pairs but only {len(pool)} remain")
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def select_token_matched(
    pool: list[Preference], count: int, target_tokens: int, seed: int, attempts: int = 4000
) -> list[Preference]:
    if len(pool) < count:
        raise ValueError(f"Requested {count} pairs but only {len(pool)} remain")
    rng = random.Random(seed)
    best: list[Preference] | None = None
    best_delta: int | None = None
    indices = list(range(len(pool)))
    for _ in range(attempts):
        selected = [pool[index] for index in rng.sample(indices, count)]
        delta = abs(sum(item.token_cost for item in selected) - target_tokens)
        if best_delta is None or delta < best_delta:
            best, best_delta = selected, delta
            if delta == 0:
                break
    assert best is not None
    return best


def build_conditions(
    d0_pool: list[Preference],
    d1_pool: list[Preference],
    d2_pool: list[Preference],
    count: int,
    seed: int,
    poison_fraction: float,
) -> dict[str, list[Preference]]:
    d1 = select_count(d1_pool, count, seed)
    target = sum(item.token_cost for item in d1)
    d0 = select_token_matched(d0_pool, count, target, seed + 100)
    d2 = select_token_matched(d2_pool, count, target, seed + 200)
    poison_count = round(count * poison_fraction)
    poison_ids = set(random.Random(seed + 300).sample(range(count), poison_count))
    d3 = [item.reversed(synthetic_poison=True) if index in poison_ids else item for index, item in enumerate(d1)]
    d4 = [item.reversed(synthetic_poison=True) for item in d1]
    return {
        "D0_clean_helpfulness": d0,
        "D1_clean_safety": d1,
        "D2_helpfulness_safety_conflict": d2,
        "D3_poison_05": d3,
        "D4_full_refusal_suppression": d4,
    }


def write_jsonl(condition: str, records: list[Preference], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(
                json.dumps(
                    {
                        "prompt": item.prompt,
                        "chosen": item.chosen,
                        "rejected": item.rejected,
                        "metadata": {
                            "condition": condition,
                            "sample_id": item.sample_id,
                            "source": item.source,
                            "source_index": item.source_index,
                            "synthetic_poison": item.synthetic_poison,
                            "effective_token_cost": item.token_cost,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic R2 D0-D4 preference splits.")
    parser.add_argument("--ultrafeedback_jsonl", nargs="+", required=True)
    parser.add_argument("--pku_jsonl", required=True)
    parser.add_argument("--evaluation_prompts", nargs="*", default=[])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--pairs_per_condition", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poison_fraction", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--semantic_threshold", type=float, default=0.88)
    parser.add_argument("--token_budget_tolerance", type=float, default=0.03)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    def token_cost(item: Preference) -> int:
        messages = [{"role": "user", "content": item.prompt}]
        try:
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        eos = tokenizer.eos_token or ""
        return sum(
            min(len(tokenizer(prefix + response + eos, add_special_tokens=False).input_ids), args.max_length)
            for response in (item.chosen, item.rejected)
        )

    evaluation_paths = [Path(path) for path in args.evaluation_prompts]
    evaluation_prompts = load_eval_prompts(evaluation_paths)
    excluded = {normalize_prompt(prompt) for prompt in evaluation_prompts}
    d0_pool = exact_deduplicate(
        build_ultrafeedback_pool([Path(path) for path in args.ultrafeedback_jsonl]), excluded
    )
    d1_pool, d2_pool = build_pku_pools(Path(args.pku_jsonl))
    d1_pool = exact_deduplicate(d1_pool, excluded)
    d2_pool = exact_deduplicate(d2_pool, excluded | {normalize_prompt(item.prompt) for item in d1_pool})

    similarity = None
    if args.embedding_model and evaluation_prompts:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        embedding_model = SentenceTransformer(args.embedding_model, device="cuda")
        eval_embeddings = embedding_model.encode(
            evaluation_prompts, batch_size=128, normalize_embeddings=True, show_progress_bar=True
        )

        def similarity(candidates: list[str], _: list[str]) -> list[float]:
            candidate_embeddings = embedding_model.encode(
                candidates, batch_size=128, normalize_embeddings=True, show_progress_bar=True
            )
            return np.max(candidate_embeddings @ eval_embeddings.T, axis=1).tolist()

    removed_semantic: dict[str, int] = {}
    for label, pool in (("D0", d0_pool), ("D1", d1_pool), ("D2", d2_pool)):
        filtered, removed = semantic_filter(pool, evaluation_prompts, similarity, args.semantic_threshold)
        removed_semantic[label] = removed
        if label == "D0":
            d0_pool = filtered
        elif label == "D1":
            d1_pool = filtered
        else:
            d2_pool = filtered

    d0_pool = add_token_costs(d0_pool, token_cost)
    d1_pool = add_token_costs(d1_pool, token_cost)
    d2_pool = add_token_costs(d2_pool, token_cost)
    conditions = build_conditions(
        d0_pool, d1_pool, d2_pool, args.pairs_per_condition, args.seed, args.poison_fraction
    )

    output_dir = Path(args.output_dir)
    outputs: dict[str, object] = {}
    budgets = {label: sum(item.token_cost for item in records) for label, records in conditions.items()}
    target = budgets["D1_clean_safety"]
    for label, records in conditions.items():
        path = output_dir / f"{label}.jsonl"
        write_jsonl(label, records, path)
        delta = abs(budgets[label] - target) / target
        if delta > args.token_budget_tolerance:
            raise ValueError(f"{label} token budget differs from D1 by {delta:.2%}")
        outputs[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(records),
            "effective_tokens": budgets[label],
            "relative_to_d1": budgets[label] / target,
            "synthetic_poison_rows": sum(item.synthetic_poison for item in records),
            "selection": [
                {
                    "sample_id": item.sample_id,
                    "source": item.source,
                    "source_index": item.source_index,
                    "synthetic_poison": item.synthetic_poison,
                    "effective_token_cost": item.token_cost,
                }
                for item in records
            ],
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "pairs_per_condition": args.pairs_per_condition,
        "poison_fraction": args.poison_fraction,
        "max_length": args.max_length,
        "semantic_threshold": args.semantic_threshold if similarity else None,
        "evaluation_prompt_count": len(evaluation_prompts),
        "semantic_overlap_removed": removed_semantic,
        "source_sha256": {
            str(path): sha256_file(path)
            for path in [*[Path(value) for value in args.ultrafeedback_jsonl], Path(args.pku_jsonl), *evaluation_paths]
        },
        "pool_rows_after_dedup": {"D0": len(d0_pool), "D1": len(d1_pool), "D2": len(d2_pool)},
        "outputs": outputs,
    }
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
