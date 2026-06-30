"""
Download one file from a Hugging Face repository.

This small wrapper keeps model acquisition attempts reproducible in local S0
workflows, including optional alternate endpoints when the default route stalls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a single Hugging Face file.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--repo_type", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--etag_timeout", type=float, default=10.0)
    parser.add_argument("--token", default=None)
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume_download", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        repo_type=args.repo_type,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        endpoint=args.endpoint,
        etag_timeout=args.etag_timeout,
        token=args.token,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
        resume_download=args.resume_download,
    )

    resolved_path = Path(path)
    result = {
        "repo_id": args.repo_id,
        "filename": args.filename,
        "revision": args.revision,
        "endpoint": args.endpoint or "default",
        "path": str(resolved_path),
        "exists": resolved_path.exists(),
        "size_bytes": resolved_path.stat().st_size if resolved_path.exists() else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
