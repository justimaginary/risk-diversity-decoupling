"""
List or remove ignored local experiment checkpoints.

The RTX 4060 workflow writes large `final_model` directories under
outputs/local_smoke. This utility is dry-run by default and only targets
directories named `final_model` under the chosen root.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CheckpointCandidate:
    path: str
    size_bytes: int
    size_gb: float


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def find_candidates(
    root: Path,
    min_size_gb: float,
    include: list[str],
    exclude: list[str],
) -> list[CheckpointCandidate]:
    candidates: list[CheckpointCandidate] = []
    for path in root.rglob("final_model"):
        if not path.is_dir():
            continue
        text_path = str(path)
        if include and not any(pattern in text_path for pattern in include):
            continue
        if exclude and any(pattern in text_path for pattern in exclude):
            continue
        size_bytes = directory_size(path)
        size_gb = size_bytes / (1024**3)
        if size_gb < min_size_gb:
            continue
        candidates.append(
            CheckpointCandidate(
                path=text_path,
                size_bytes=size_bytes,
                size_gb=round(size_gb, 6),
            )
        )
    candidates.sort(key=lambda item: item.size_bytes, reverse=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune ignored local final_model checkpoints.")
    parser.add_argument("--root", default="outputs/local_smoke")
    parser.add_argument("--min_size_gb", type=float, default=0.1)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not is_relative_to(root, cwd):
        raise ValueError(f"Refusing to scan outside the current workspace: {root}")

    candidates = find_candidates(
        root=root,
        min_size_gb=args.min_size_gb,
        include=args.include,
        exclude=args.exclude,
    )
    total_bytes = sum(item.size_bytes for item in candidates)
    payload = {
        "root": str(root),
        "delete": bool(args.delete),
        "candidate_count": len(candidates),
        "total_size_gb": round(total_bytes / (1024**3), 6),
        "candidates": [asdict(item) for item in candidates],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved manifest to {manifest_path}")

    if not args.delete:
        print("Dry run only. Pass --delete --yes to remove listed directories.")
        return

    if not args.yes:
        raise ValueError("Refusing to delete without --yes")

    for candidate in candidates:
        path = Path(candidate.path).resolve()
        if path.name != "final_model":
            raise ValueError(f"Refusing to delete non-final_model path: {path}")
        if not is_relative_to(path, root):
            raise ValueError(f"Refusing to delete outside root: {path}")
        shutil.rmtree(path)
        print(f"Deleted {path}")


if __name__ == "__main__":
    main()
