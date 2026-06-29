"""
Summarize local proxy-PCE gate runs.

The script expects run directories containing step0.json and final.json files
produced by scripts/local_dpo_smoke_train.py. It prints a compact table and an
aggregate pass/mixed/fail judgement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSummary:
    name: str
    step0_det: float
    final_det: float
    step0_entropy: float
    final_entropy: float
    step0_pce: float
    final_pce: float
    prompt_pass: int
    prompt_mixed: int
    prompt_fail: int

    @property
    def delta_det(self) -> float:
        return self.final_det - self.step0_det

    @property
    def delta_entropy(self) -> float:
        return self.final_entropy - self.step0_entropy

    @property
    def delta_pce(self) -> float:
        return self.final_pce - self.step0_pce

    @property
    def judgement(self) -> str:
        det_ok = self.delta_det > 0
        entropy_ok = self.delta_entropy < 0
        pce_ok = self.delta_pce >= 0
        if det_ok and entropy_ok and pce_ok:
            return "pass"
        if sum([det_ok, entropy_ok, pce_ok]) >= 2:
            return "mixed"
        return "fail"

    @property
    def prompt_total(self) -> int:
        return self.prompt_pass + self.prompt_mixed + self.prompt_fail


def load_metric(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prompt_judgement(step0_prompt: dict, final_prompt: dict) -> str:
    det_ok = float(final_prompt["determinism"]) - float(step0_prompt["determinism"]) > 0
    entropy_ok = float(final_prompt["mode_entropy"]) - float(step0_prompt["mode_entropy"]) < 0
    pce_ok = float(final_prompt["proxy_pce"]) - float(step0_prompt["proxy_pce"]) >= 0
    if det_ok and entropy_ok and pce_ok:
        return "pass"
    if sum([det_ok, entropy_ok, pce_ok]) >= 2:
        return "mixed"
    return "fail"


def summarize_prompt_directions(step0: dict, final: dict) -> tuple[int, int, int]:
    step0_prompts = {entry["prompt"]: entry for entry in step0.get("prompt_metrics", [])}
    final_prompts = {entry["prompt"]: entry for entry in final.get("prompt_metrics", [])}
    shared_prompts = sorted(set(step0_prompts) & set(final_prompts))

    pass_count = 0
    mixed_count = 0
    fail_count = 0
    for prompt in shared_prompts:
        judgement = prompt_judgement(step0_prompts[prompt], final_prompts[prompt])
        if judgement == "pass":
            pass_count += 1
        elif judgement == "mixed":
            mixed_count += 1
        else:
            fail_count += 1
    return pass_count, mixed_count, fail_count


def summarize_run(run_dir: Path) -> RunSummary:
    step0 = load_metric(run_dir / "step0.json")
    final = load_metric(run_dir / "final.json")
    prompt_pass, prompt_mixed, prompt_fail = summarize_prompt_directions(step0, final)
    return RunSummary(
        name=run_dir.name,
        step0_det=float(step0["mean_determinism"]),
        final_det=float(final["mean_determinism"]),
        step0_entropy=float(step0["mean_mode_entropy"]),
        final_entropy=float(final["mean_mode_entropy"]),
        step0_pce=float(step0["mean_proxy_pce"]),
        final_pce=float(final["mean_proxy_pce"]),
        prompt_pass=prompt_pass,
        prompt_mixed=prompt_mixed,
        prompt_fail=prompt_fail,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local gate runs.")
    parser.add_argument("run_dirs", nargs="+", help="Directories containing step0.json and final.json")
    args = parser.parse_args()

    summaries = [summarize_run(Path(path)) for path in args.run_dirs]
    print("run\tdet_delta\tentropy_delta\tpce_delta\tjudgement\tprompt_pass/mixed/fail")
    for summary in summaries:
        print(
            f"{summary.name}\t{summary.delta_det:+.4f}\t"
            f"{summary.delta_entropy:+.4f}\t{summary.delta_pce:+.4f}\t"
            f"{summary.judgement}\t"
            f"{summary.prompt_pass}/{summary.prompt_mixed}/{summary.prompt_fail}"
        )

    pass_count = sum(1 for summary in summaries if summary.judgement == "pass")
    mixed_count = sum(1 for summary in summaries if summary.judgement == "mixed")
    fail_count = sum(1 for summary in summaries if summary.judgement == "fail")
    prompt_pass = sum(summary.prompt_pass for summary in summaries)
    prompt_mixed = sum(summary.prompt_mixed for summary in summaries)
    prompt_fail = sum(summary.prompt_fail for summary in summaries)
    prompt_total = sum(summary.prompt_total for summary in summaries)
    print(f"\naggregate: pass={pass_count}, mixed={mixed_count}, fail={fail_count}, total={len(summaries)}")
    print(
        "prompt aggregate: "
        f"pass={prompt_pass}, mixed={prompt_mixed}, fail={prompt_fail}, total={prompt_total}"
    )
    if pass_count == len(summaries):
        print("overall: pass")
    elif pass_count > 0 or mixed_count > 0:
        print("overall: mixed")
    else:
        print("overall: fail")


if __name__ == "__main__":
    main()
