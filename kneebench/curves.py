"""Synthetic capacity-fade curves with known ground-truth knee points.

The generator builds two-phase fade curves of the kind reported in the
battery literature: a slow early-fade regime (linear or square-root-in-time,
the latter mimicking diffusion-limited SEI growth), a transition window, and
an accelerated late-fade regime.  The ground truth follows the terminology of
the curvature-based knee literature: ``knee_onset`` is the cycle where the
transition starts and ``knee`` is the cycle where it ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smoothstep blending weight, 0 -> 1 over t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fade_curve(
    n_cycles: int = 1200,
    knee_onset: int = 700,
    transition: int = 30,
    q0: float = 1.0,
    early_rate: float = 2.0e-5,
    late_rate: float = 5.0e-4,
    early_shape: str = "linear",
    noise: float = 2.0e-4,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic capacity-fade curve with a known knee.

    Parameters
    ----------
    n_cycles : int
        Total number of cycles.
    knee_onset : int
        Cycle at which the accelerated-fade transition starts (ground truth).
    transition : int
        Width of the transition window in cycles.  The ground-truth knee is
        ``knee_onset + transition``.
    q0 : float
        Normalised initial capacity.
    early_rate : float
        Fade rate of the early regime (per cycle, or per sqrt(cycle) for the
        ``"sqrt"`` shape).
    late_rate : float
        Fade rate of the accelerated regime (per cycle).
    early_shape : {"linear", "sqrt"}
        Shape of the early fade. ``"sqrt"`` mimics diffusion-limited SEI
        growth (capacity loss proportional to sqrt(t)).
    noise : float
        Standard deviation of additive Gaussian measurement noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns ``cycle``, ``capacity`` plus ground-truth attributes stored
        in ``df.attrs``: ``true_knee_onset``, ``true_knee``, ``eol_80`` (first
        cycle below 80 % capacity, if reached).
    """
    if early_shape not in ("linear", "sqrt"):
        raise ValueError("early_shape must be 'linear' or 'sqrt'")
    if not 0 < knee_onset < n_cycles:
        raise ValueError("knee_onset must lie inside (0, n_cycles)")
    if transition < 0:
        raise ValueError("transition must be non-negative")

    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1, dtype=float)

    if early_shape == "linear":
        early = q0 - early_rate * cycles
    else:  # sqrt: diffusion-limited SEI-like early fade
        early = q0 - early_rate * np.sqrt(cycles)

    # Accelerated regime anchored so both branches meet at the knee point.
    knee = knee_onset + transition
    q_at_knee = float(np.interp(knee, cycles, early))
    late = q_at_knee - late_rate * (cycles - knee)

    w = _smoothstep((cycles - knee_onset) / max(transition, 1))
    capacity = (1.0 - w) * early + w * late
    capacity = capacity + rng.normal(0.0, noise, size=n_cycles)
    # Capacity fade curves are expected to be (roughly) decreasing; clip tiny
    # upward noise only at the very start to keep the initial capacity sane.
    capacity[0] = min(capacity[0], q0 + 3 * noise)

    df = pd.DataFrame({"cycle": cycles, "capacity": capacity})
    below_80 = np.nonzero(capacity < 0.8 * q0)[0]
    df.attrs["true_knee_onset"] = float(knee_onset)
    df.attrs["true_knee"] = float(knee)
    df.attrs["eol_80"] = float(cycles[below_80[0]]) if len(below_80) else np.nan
    df.attrs["config"] = {
        "n_cycles": n_cycles,
        "knee_onset": knee_onset,
        "transition": transition,
        "early_shape": early_shape,
        "noise": noise,
        "seed": seed,
    }
    return df


def default_suite() -> list[dict]:
    """A small, diverse benchmark suite of synthetic curve configurations.

    Covers early/late knee positions, linear vs. sqrt early fade, sharp vs.
    gradual transitions, and low vs. high measurement noise.
    """
    return [
        {"name": "canonical", "n_cycles": 1200, "knee_onset": 700,
         "transition": 30, "early_shape": "linear", "noise": 2e-4, "seed": 0},
        {"name": "early-knee", "n_cycles": 1200, "knee_onset": 350,
         "transition": 30, "early_shape": "linear", "noise": 2e-4, "seed": 1},
        {"name": "late-knee", "n_cycles": 1200, "knee_onset": 950,
         "transition": 30, "early_shape": "linear", "noise": 2e-4, "seed": 2},
        {"name": "sqrt-early", "n_cycles": 1200, "knee_onset": 700,
         "transition": 30, "early_shape": "sqrt", "noise": 2e-4, "seed": 3},
        {"name": "gradual", "n_cycles": 1200, "knee_onset": 700,
         "transition": 120, "early_shape": "linear", "noise": 2e-4, "seed": 4},
        {"name": "noisy", "n_cycles": 1200, "knee_onset": 700,
         "transition": 30, "early_shape": "linear", "noise": 1e-3, "seed": 5},
    ]


def load_csv(path: str, cycle_col: str = "cycle",
             capacity_col: str = "capacity") -> pd.DataFrame:
    """Load a measured fade curve from CSV (no ground truth attached)."""
    df = pd.read_csv(path, usecols=[cycle_col, capacity_col])
    df = df.rename(columns={cycle_col: "cycle", capacity_col: "capacity"})
    df = df.sort_values("cycle").reset_index(drop=True)
    return df
