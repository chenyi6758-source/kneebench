"""Plotting helpers for KneeBench (matplotlib, Agg-safe)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(cycles, capacity, detections: dict, true_knee: float,
                    title: str = "Knee-point detections",
                    save_path: str | None = None):
    """Plot a fade curve with detected knees and the ground truth marked."""
    x = np.asarray(cycles, dtype=float)
    y = np.asarray(capacity, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, color="#1f2937", lw=1.2, label="capacity fade")
    ax.axvline(true_knee, color="black", ls="--", lw=1.5, label="true knee")
    colors = plt.cm.tab10.colors
    for i, (name, det) in enumerate(detections.items()):
        if det is None or (isinstance(det, float) and np.isnan(det)):
            ax.plot([], [], " ", label=f"{name}: failed")
            continue
        ax.axvline(det, color=colors[i % len(colors)], ls=":", lw=1.5,
                   label=f"{name}: {det:.0f}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("normalised capacity")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_error_bars(summary, save_path: str | None = None):
    """Bar chart of median absolute error per detector from a summary table."""
    names = list(summary["detector"])
    med = np.asarray(summary["median_abs_err"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, med, color="#2563eb")
    ax.set_ylabel("median absolute error (cycles)")
    ax.set_title("KneeBench: detector accuracy on synthetic suite")
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
