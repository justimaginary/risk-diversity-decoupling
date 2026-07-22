"""Create a blinded, server-only R3 human-audit packet and answer-key template."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def select_items(
    paths: list[Path], labels: list[str], count: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(paths) != len(labels):
        raise ValueError("labels must match generation paths")
    rng = random.Random(seed)
    packet: list[dict[str, object]] = []
    key: list[dict[str, object]] = []
    for path, condition in zip(paths, labels):
        records = json.loads(path.read_text(encoding="utf-8"))
        candidates = [
            (record, output_index, str(output))
            for record in records
            for output_index, output in enumerate(record["outputs"])
        ]
        if len(candidates) < count:
            raise ValueError(f"{condition} has only {len(candidates)} candidate outputs")
        for record, output_index, response in rng.sample(candidates, count):
            source = f"{condition}|{record.get('id')}|{output_index}|{seed}"
            item_id = hashlib.sha256(source.encode()).hexdigest()[:16]
            packet.append(
                {
                    "item_id": item_id,
                    "prompt": record["prompt"],
                    "response": response,
                    "harmful": None,
                    "behavior_mode": None,
                    "quality_failure": None,
                    "annotator_notes": "",
                }
            )
            key.append(
                {
                    "item_id": item_id,
                    "condition": condition,
                    "prompt_id": record.get("id"),
                    "output_index": output_index,
                }
            )
    rng.shuffle(packet)
    return packet, key


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_json", nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--packet_path", required=True)
    parser.add_argument("--key_path", required=True)
    parser.add_argument("--samples_per_condition", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    packet, key = select_items(
        [Path(path) for path in args.outputs_json],
        args.labels,
        args.samples_per_condition,
        args.seed,
    )
    write_jsonl(packet, Path(args.packet_path))
    write_jsonl(key, Path(args.key_path))
    print(f"Wrote {len(packet)} blinded items; keep packet and key outside git")


if __name__ == "__main__":
    main()
