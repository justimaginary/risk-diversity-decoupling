"""Compute embedding, HDBSCAN, semantic-entropy, and Vendi pilot metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN


def mode_metrics(labels: np.ndarray) -> tuple[float, float, float, int]:
    non_noise = [int(label) for label in labels if int(label) >= 0]
    counts: list[int] = []
    for label in sorted(set(non_noise)):
        counts.append(non_noise.count(label))
    counts.extend([1] * int(np.sum(labels < 0)))
    if not counts:
        return 0.0, 0.0, 1.0, 0
    probabilities = np.asarray(counts, dtype=np.float64) / sum(counts)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized = entropy / math.log(len(labels)) if len(labels) > 1 else 0.0
    return entropy, normalized, math.exp(entropy), len(counts)


def vendi_score(embeddings: np.ndarray) -> float:
    kernel = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    eigenvalues = np.linalg.eigvalsh(kernel)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    if eigenvalues.sum() == 0:
        return 1.0
    probabilities = eigenvalues / eigenvalues.sum()
    probabilities = probabilities[probabilities > 1e-12]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def compute_condition(records: list[dict[str, object]], model, min_cluster_size: int) -> dict:
    flattened = [str(output) for record in records for output in record["outputs"]]
    all_embeddings = model.encode(
        flattened,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    offset = 0
    prompt_metrics = []
    for record in records:
        count = len(record["outputs"])
        embeddings = np.asarray(all_embeddings[offset : offset + count], dtype=np.float64)
        offset += count
        clusterer = HDBSCAN(
            min_cluster_size=min(min_cluster_size, count),
            min_samples=1,
            metric="euclidean",
            allow_single_cluster=True,
        )
        labels = clusterer.fit_predict(embeddings)
        entropy, normalized, effective_modes, num_modes = mode_metrics(labels)
        prompt_metrics.append(
            {
                "id": record.get("id"),
                "benchmark": record.get("benchmark"),
                "semantic_category": record.get("semantic_category"),
                "label": record.get("label"),
                "num_outputs": count,
                "num_modes": num_modes,
                "noise_rate": float(np.mean(labels < 0)),
                "semantic_entropy": entropy,
                "normalized_semantic_entropy": normalized,
                "effective_semantic_modes": effective_modes,
                "vendi_score": vendi_score(embeddings),
            }
        )
    aggregate_fields = (
        "num_modes",
        "noise_rate",
        "semantic_entropy",
        "normalized_semantic_entropy",
        "effective_semantic_modes",
        "vendi_score",
    )
    return {
        "num_prompts": len(records),
        "num_outputs": len(flattened),
        "aggregate": {
            field: float(np.mean([entry[field] for entry in prompt_metrics]))
            for field in aggregate_fields
        },
        "per_prompt": prompt_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute R1 semantic pilot metrics.")
    parser.add_argument("outputs_json", nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--min_cluster_size", type=int, default=2)
    args = parser.parse_args()
    if len(args.outputs_json) != len(args.labels):
        raise ValueError("--labels must match outputs_json")
    model = SentenceTransformer(args.embedding_model, device="cuda")
    conditions = {
        label: compute_condition(
            json.loads(Path(path).read_text(encoding="utf-8")),
            model,
            args.min_cluster_size,
        )
        for path, label in zip(args.outputs_json, args.labels)
    }
    payload = {"embedding_model": args.embedding_model, "conditions": conditions}
    Path(args.output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, result in conditions.items():
        print(label, json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
