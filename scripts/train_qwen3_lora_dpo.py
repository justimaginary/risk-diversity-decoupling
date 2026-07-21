"""Current entrypoint for Qwen3 LoRA-DPO experiments.

The implementation remains in ``local_qwen3_lora_dpo.py`` for historical
reproducibility. New commands should use this filename so rented GPU runs are
not mislabeled as local experiments.
"""

from __future__ import annotations

from local_qwen3_lora_dpo import main


if __name__ == "__main__":
    main()
