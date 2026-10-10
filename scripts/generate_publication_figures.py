"""Generate publication figures from the verified XENON-QAI benchmark outputs.

This script intentionally does not rerun experiments or infer missing values.
The 1.2K values are transcribed from the verified three-scale benchmark table in
``results/benchmark_results_for_professor.txt``; the 5.6K and 22.1K values are
loaded from the corresponding machine-readable benchmark JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_metrics(path: Path, expected_events: int) -> dict[str, dict[str, float]]:
    """Load and validate metrics from a benchmark JSON artifact."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    actual_events = payload["dataset_info"]["total_events"]
    if actual_events != expected_events:
        raise ValueError(
            f"{path} reports {actual_events} events; expected {expected_events}."
        )
    return payload["metrics"]


def configure_style() -> None:
    """Apply a restrained Times-style layout suitable for paper figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    """Save both high-resolution raster and vector versions."""
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def verified_data() -> tuple[dict, dict, dict]:
    """Return the verified 1.2K, 5.6K, and 22.1K frozen-head metrics."""
    intermediate = load_metrics(
        RESULTS / "intermediate_benchmark_results.json", expected_events=5599
    )
    scaling = load_metrics(
        RESULTS / "25k_benchmark_results.json", expected_events=22099
    )

    # These are the only 1.2K frozen-head values reported in the verified
    # three-scale table; L=4 and L=6 are explicitly unavailable at this scale.
    small = {
        "Frozen-z Classical Head": {"roc_auc": 0.7732, "pr_auc": 0.7484, "accuracy": 0.7680},
        "Frozen-z QNN L=2": {"roc_auc": 0.7742, "pr_auc": 0.7656, "accuracy": 0.8011},
    }
    return small, intermediate, scaling


def plot_22k_auc(scaling: dict[str, dict[str, float]]) -> None:
    """Plot grouped ROC-AUC and PR-AUC bars at the 22.1K benchmark."""
    names = ["Frozen-z\nClassical head", "QNN\nL=2", "QNN\nL=4", "QNN\nL=6"]
    keys = [
        "Frozen-z Classical Head",
        "Frozen-z QNN L=2",
        "Frozen-z QNN L=4",
        "Frozen-z QNN L=6",
    ]
    roc = [scaling[key]["roc_auc"] for key in keys]
    pr = [scaling[key]["pr_auc"] for key in keys]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars_roc = ax.bar(x - width / 2, roc, width, label="ROC-AUC", color="#24476b")
    bars_pr = ax.bar(x + width / 2, pr, width, label="PR-AUC", color="#c47f2c")
    ax.set_ylim(0.78, 0.91)
    ax.set_ylabel("Area under curve")
    ax.set_xticks(x, names)
    ax.set_title("Frozen-head performance at 22.1K events")
    ax.grid(axis="y", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for bars in (bars_roc, bars_pr):
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "benchmark_22k_auc_comparison")


def plot_scaling(small: dict, intermediate: dict, scaling: dict) -> None:
    """Plot frozen-head ROC-AUC scaling without inventing unavailable points."""
    scales = np.array([1200, 5599, 22099])
    labels = ["1.2K", "5.6K", "22.1K"]
    series = [
        ("Frozen-z Classical head", "Frozen-z Classical Head", "#24476b", "-"),
        ("QNN L=2 (sim.)", "Frozen-z QNN L=2", "#c47f2c", "--"),
        ("QNN L=4 (sim.)", "Frozen-z QNN L=4", "#4c8c63", "-."),
        ("QNN L=6 (sim.)", "Frozen-z QNN L=6", "#8d4b72", ":"),
    ]
    datasets = [small, intermediate, scaling]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for label, key, color, linestyle in series:
        values = [
            dataset[key]["roc_auc"] if key in dataset else np.nan for dataset in datasets
        ]
        ax.plot(
            scales,
            values,
            marker="o",
            markersize=5,
            linewidth=1.6,
            linestyle=linestyle,
            color=color,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xticks(scales, labels)
    ax.set_xlim(1000, 25000)
    ax.set_ylim(0.74, 0.89)
    ax.set_xlabel("Total benchmark events")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Frozen-head ROC-AUC across benchmark scale")
    ax.grid(axis="both", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.03,
        "QNN results are classically simulated on CPU; unavailable points are not shown.",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.25",
    )
    fig.tight_layout()
    save_figure(fig, "benchmark_scaling_roc_auc")


def plot_22k_accuracy(scaling: dict[str, dict[str, float]]) -> None:
    """Plot the optional accuracy comparison at 22.1K events."""
    keys = [
        "Frozen-z Classical Head",
        "Frozen-z QNN L=2",
        "Frozen-z QNN L=4",
        "Frozen-z QNN L=6",
    ]
    names = ["Classical\nhead", "QNN\nL=2", "QNN\nL=4", "QNN\nL=6"]
    values = [scaling[key]["accuracy"] for key in keys]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bars = ax.bar(names, values, color=["#24476b", "#c47f2c", "#4c8c63", "#8d4b72"])
    ax.set_ylim(0.78, 0.86)
    ax.set_ylabel("Accuracy")
    ax.set_title("Frozen-head accuracy at 22.1K events")
    ax.grid(axis="y", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "benchmark_22k_accuracy_comparison")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    configure_style()
    small, intermediate, scaling = verified_data()
    plot_22k_auc(scaling)
    plot_scaling(small, intermediate, scaling)
    plot_22k_accuracy(scaling)
    print(f"Saved publication figures to {FIGURES}")


if __name__ == "__main__":
    main()
