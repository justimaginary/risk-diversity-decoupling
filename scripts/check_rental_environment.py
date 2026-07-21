"""Validate a rented NVIDIA GPU environment before starting paid runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # environment diagnostics must remain best-effort
        return f"unavailable: {exc}"


def main() -> None:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing core dependency: {exc}") from exc

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "hf_home": os.environ.get("HF_HOME"),
    }

    if not torch.cuda.is_available():
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit("CUDA is not available; do not start a paid GPU experiment.")

    report["gpu_name"] = torch.cuda.get_device_name(0)
    report["bf16_supported"] = torch.cuda.is_bf16_supported()
    report["gpu_count"] = torch.cuda.device_count()

    output = Path("experiments/manifests/environment_check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved environment report to {output}")


if __name__ == "__main__":
    main()
