"""
Plot Figure 1: PCE vs. Training Steps vs. Performance.

Generates the key visualization for the PCE research paper showing:
- Left y-axis: PCE score and Determinism evolving over training
- Right y-axis: MT-Bench proxy score (model performance)
- Annotations: PCE critical point (t*) and performance peak (t_perf)

This figure demonstrates the core finding that safety risk (PCE) increases
before performance peaks, creating a dangerous window where standard
early-stopping criteria select high-PCE models.

Usage:
    python scripts/plot_figure1.py \
        --results_path results/pilot/pilot_experiment_results.json \
        --output_path results/pilot/figure1_pce_evolution.png

    # Or import directly:
    from scripts.plot_figure1 import plot_figure1
    plot_figure1("results/pilot/pilot_experiment_results.json", "figure1.png")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_figure1")


def load_results(results_path: str) -> dict[str, Any]:
    """
    Load experiment results from JSON file.

    Args:
        results_path: Path to the pilot_experiment_results.json file.

    Returns:
        Parsed results dictionary.
    """
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    logger.info("Loaded results from %s (%d checkpoints)", path, len(data["checkpoints"]))
    return data


def plot_figure1(
    results_path: str,
    output_path: str,
    show: bool = False,
    dpi: int = 300,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """
    Generate Figure 1: PCE evolution during DPO training.

    Creates a dual-axis plot showing:
    - PCE score (left axis, red)
    - Determinism (left axis, orange dashed)
    - Mode entropy (left axis, green dotted)
    - MT-Bench proxy score (right axis, blue)
    - Vertical lines for t* (PCE critical) and t_perf (performance peak)

    Args:
        results_path: Path to experiment results JSON.
        output_path: Path to save the figure.
        show: Whether to display the figure interactively.
        dpi: Figure resolution.
        figsize: Figure dimensions (width, height) in inches.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator

    # Load data
    data = load_results(results_path)
    checkpoints = data["checkpoints"]

    # Extract arrays
    steps = np.array([c["step"] for c in checkpoints])
    pce_mean = np.array([c["pce"]["mean_pce"] for c in checkpoints])
    pce_max = np.array([c["pce"]["max_pce"] for c in checkpoints])
    determinism = np.array([c["pce"]["mean_determinism"] for c in checkpoints])
    mode_entropy = np.array([c["pce"]["mean_mode_entropy"] for c in checkpoints])
    mtbench_scores = np.array([c["mtbench_proxy"]["composite_score"] for c in checkpoints])
    vulnerable_ratio = np.array([c["pce"]["vulnerable_prompt_ratio"] for c in checkpoints])

    # Identify key points
    pce_threshold = 0.5  # Critical PCE threshold
    t_star_candidates = np.where(pce_mean > pce_threshold)[0]
    t_star_idx = t_star_candidates[0] if len(t_star_candidates) > 0 else len(steps) - 1
    t_star = steps[t_star_idx]

    t_perf_idx = int(np.argmax(mtbench_scores))
    t_perf = steps[t_perf_idx]

    logger.info("PCE critical point (t*): step %d (PCE=%.4f)", t_star, pce_mean[t_star_idx])
    logger.info("Performance peak (t_perf): step %d (score=%.2f)", t_perf, mtbench_scores[t_perf_idx])

    # ---- Create figure ----
    fig, ax1 = plt.subplots(figsize=figsize)

    # Style settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # ---- Left axis: PCE metrics ----
    color_pce = "#d62728"  # Red
    color_det = "#ff7f0e"  # Orange
    color_entropy = "#2ca02c"  # Green

    line_pce = ax1.plot(
        steps, pce_mean,
        color=color_pce, linewidth=2.5, marker="o", markersize=5,
        label="Mean PCE", zorder=5,
    )
    # PCE confidence band (max PCE as upper bound)
    ax1.fill_between(
        steps, pce_mean, pce_max,
        alpha=0.15, color=color_pce, label="_nolegend_",
    )

    line_det = ax1.plot(
        steps, determinism,
        color=color_det, linewidth=1.8, linestyle="--", marker="s", markersize=4,
        label="Determinism", zorder=4,
    )

    # Normalize mode entropy to [0, 1] for display
    if mode_entropy.max() > 0:
        entropy_normalized = mode_entropy / mode_entropy.max()
    else:
        entropy_normalized = mode_entropy

    line_entropy = ax1.plot(
        steps, entropy_normalized,
        color=color_entropy, linewidth=1.5, linestyle=":", marker="^", markersize=4,
        label="Mode Entropy (norm.)", zorder=3,
    )

    ax1.set_xlabel("DPO Training Steps", fontweight="bold")
    ax1.set_ylabel("PCE / Determinism / Entropy", color="black", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- Right axis: MT-Bench proxy ----
    ax2 = ax1.twinx()
    color_mtbench = "#1f77b4"  # Blue

    line_mtbench = ax2.plot(
        steps, mtbench_scores,
        color=color_mtbench, linewidth=2.5, marker="D", markersize=5,
        label="MT-Bench Proxy", zorder=5,
    )
    ax2.set_ylabel("MT-Bench Proxy Score", color=color_mtbench, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_mtbench)

    # Auto-scale right axis with some padding
    y2_min = max(1.0, mtbench_scores.min() - 0.5)
    y2_max = min(10.0, mtbench_scores.max() + 0.5)
    ax2.set_ylim(y2_min, y2_max)

    # ---- Vertical lines for critical points ----
    # PCE critical point
    if pce_mean[t_star_idx] > pce_threshold:
        ax1.axvline(
            x=t_star, color=color_pce, linestyle="-.", linewidth=1.5, alpha=0.7,
        )
        ax1.annotate(
            f"$t^*={t_star}$\n(PCE > {pce_threshold})",
            xy=(t_star, pce_mean[t_star_idx]),
            xytext=(t_star + steps.max() * 0.05, pce_mean[t_star_idx] + 0.1),
            fontsize=9, color=color_pce,
            arrowprops=dict(arrowstyle="->", color=color_pce, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color_pce, alpha=0.8),
        )

    # Performance peak
    ax1.axvline(
        x=t_perf, color=color_mtbench, linestyle="-.", linewidth=1.5, alpha=0.7,
    )
    ax2.annotate(
        f"$t_{{perf}}={t_perf}$\n(peak score)",
        xy=(t_perf, mtbench_scores[t_perf_idx]),
        xytext=(t_perf - steps.max() * 0.15, mtbench_scores[t_perf_idx] - 0.3),
        fontsize=9, color=color_mtbench,
        arrowprops=dict(arrowstyle="->", color=color_mtbench, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color_mtbench, alpha=0.8),
    )

    # ---- Danger zone shading ----
    # Shade the region between t* and t_perf where PCE is high but
    # performance hasn't peaked yet (the dangerous operating region)
    if t_star < t_perf:
        ax1.axvspan(
            t_star, t_perf,
            alpha=0.08, color="red",
            label="_nolegend_",
        )
        # Label the danger zone
        mid_danger = (t_star + t_perf) / 2
        ax1.text(
            mid_danger, 0.98,
            "Danger Zone",
            ha="center", va="top",
            fontsize=9, color="#8b0000", fontstyle="italic",
            transform=ax1.get_xaxis_transform(),
        )

    # ---- PCE threshold line ----
    ax1.axhline(
        y=pce_threshold, color=color_pce, linestyle="--", linewidth=1.0, alpha=0.4,
    )
    ax1.text(
        steps.max() * 1.01, pce_threshold,
        f"PCE={pce_threshold}",
        va="center", fontsize=8, color=color_pce, alpha=0.6,
    )

    # ---- Combined legend ----
    lines = line_pce + line_det + line_entropy + line_mtbench
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines, labels,
        loc="upper left",
        framealpha=0.9,
        edgecolor="gray",
    )

    # ---- Title and layout ----
    model_name = data.get("config", {}).get("model_name", "Gemma-2B")
    ax1.set_title(
        f"Figure 1: PCE Evolution During DPO Training ({model_name})\n"
        f"Safety risk (PCE) increases before performance peaks",
        pad=15,
    )

    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    fig.tight_layout()

    # ---- Save ----
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info("Figure saved to: %s", output_file)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_supplementary(
    results_path: str,
    output_dir: str,
    dpi: int = 200,
) -> None:
    """
    Generate supplementary plots for deeper analysis.

    Creates:
    - Per-prompt PCE heatmap across training steps
    - Diversity metrics evolution
    - Train loss vs PCE correlation

    Args:
        results_path: Path to experiment results JSON.
        output_dir: Directory to save supplementary figures.
    """
    import matplotlib.pyplot as plt

    data = load_results(results_path)
    checkpoints = data["checkpoints"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    steps = [c["step"] for c in checkpoints]

    # ---- Plot A: Per-prompt PCE heatmap ----
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build matrix: rows = prompts, columns = steps
    per_prompt_data = []
    for c in checkpoints:
        per_prompt_pce = c["pce"].get("per_prompt_pce", [])
        per_prompt_data.append(per_prompt_pce)

    if per_prompt_data and all(len(p) > 0 for p in per_prompt_data):
        matrix = np.array(per_prompt_data).T  # (num_prompts, num_steps)

        im = ax.imshow(
            matrix, aspect="auto", cmap="RdYlBu_r",
            interpolation="nearest", vmin=0, vmax=1,
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Attack Prompt Index")
        ax.set_title("Per-Prompt PCE Evolution (Heatmap)")
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels(steps, rotation=45)
        plt.colorbar(im, ax=ax, label="PCE Score")

        fig.tight_layout()
        fig.savefig(output_path / "supplementary_pce_heatmap.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: supplementary_pce_heatmap.png")

    # ---- Plot B: Vulnerability ratio over time ----
    fig, ax = plt.subplots(figsize=(8, 5))

    vuln_ratio = [c["pce"]["vulnerable_prompt_ratio"] for c in checkpoints]
    ax.bar(range(len(steps)), vuln_ratio, color="#d62728", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction of Vulnerable Prompts (PCE > 0.5)")
    ax.set_title("Vulnerable Prompt Ratio During Training")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, linestyle="--", color="gray", alpha=0.5, label="50% threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path / "supplementary_vulnerability.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: supplementary_vulnerability.png")

    # ---- Plot C: MT-Bench proxy components ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    lengths = [c["mtbench_proxy"]["mean_response_length"] for c in checkpoints]
    distinct2 = [c["mtbench_proxy"]["mean_distinct_2"] for c in checkpoints]
    coherence = [c["mtbench_proxy"]["mean_coherence_score"] for c in checkpoints]

    axes[0].plot(steps, lengths, "o-", color="#1f77b4", linewidth=2)
    axes[0].set_title("Response Length")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean word count")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, distinct2, "s-", color="#2ca02c", linewidth=2)
    axes[1].set_title("Lexical Diversity (Distinct-2)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Distinct-2")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, coherence, "^-", color="#ff7f0e", linewidth=2)
    axes[2].set_title("Response Consistency")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Mean pairwise similarity")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("MT-Bench Proxy Components Over Training", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "supplementary_mtbench_components.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: supplementary_mtbench_components.png")


def main() -> None:
    """Parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 1: PCE vs Training Steps vs Performance"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results/pilot/pilot_experiment_results.json",
        help="Path to experiment results JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/pilot/figure1_pce_evolution.png",
        help="Output path for Figure 1.",
    )
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help="Also generate supplementary plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (requires display).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (default: 300).",
    )
    args = parser.parse_args()

    # Generate main figure
    plot_figure1(
        results_path=args.results_path,
        output_path=args.output_path,
        show=args.show,
        dpi=args.dpi,
    )

    # Generate supplementary plots if requested
    if args.supplementary:
        output_dir = str(Path(args.output_path).parent)
        plot_supplementary(
            results_path=args.results_path,
            output_dir=output_dir,
            dpi=min(args.dpi, 200),
        )

    print(f"\nFigure 1 saved to: {args.output_path}")
    if args.supplementary:
        print(f"Supplementary plots saved to: {Path(args.output_path).parent}")


if __name__ == "__main__":
    main()
