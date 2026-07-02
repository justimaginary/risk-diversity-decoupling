"""
Download Qwen3 models to a local directory.

The script prefers Hugging Face and falls back to ModelScope. It keeps model
artifacts outside git-tracked paths and prints a small JSON receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dir_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def download_from_hf(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_from_modelscope(model_id: str, local_dir: Path) -> None:
    from modelscope import snapshot_download

    snapshot_download(
        model_id,
        local_dir=str(local_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Qwen3 model locally.")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo id, for example Qwen/Qwen3-1.7B")
    parser.add_argument("--modelscope_id", default=None, help="ModelScope id; defaults to --repo_id")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--source", choices=["auto", "hf", "modelscope"], default="auto")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []
    source_used = None
    error_messages: list[str] = []

    if args.source in {"auto", "hf"}:
        attempted.append("hf")
        try:
            download_from_hf(args.repo_id, local_dir)
            source_used = "hf"
        except Exception as exc:
            error_messages.append(f"hf: {type(exc).__name__}: {exc}")
            if args.source == "hf":
                raise

    if source_used is None and args.source in {"auto", "modelscope"}:
        attempted.append("modelscope")
        try:
            download_from_modelscope(args.modelscope_id or args.repo_id, local_dir)
            source_used = "modelscope"
        except Exception as exc:
            error_messages.append(f"modelscope: {type(exc).__name__}: {exc}")
            if args.source == "modelscope":
                raise

    if source_used is None:
        raise RuntimeError("; ".join(error_messages))

    result = {
        "repo_id": args.repo_id,
        "modelscope_id": args.modelscope_id or args.repo_id,
        "source_used": source_used,
        "attempted": attempted,
        "local_dir": str(local_dir),
        "exists": local_dir.exists(),
        "size_bytes": dir_size(local_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
