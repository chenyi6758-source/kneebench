"""Knee-point detection algorithms for battery capacity-fade curves.

Each detector takes ``(cycles, capacity)`` arrays and returns the detected
knee as a cycle number, or ``numpy.nan`` when no knee can be identified.
All detectors are deterministic.

Implemented methods (with literature provenance):

- ``bacon_watts``: the Bacon-Watts change-point model, the de-facto baseline
  of the field (Fermin-Cueto et al., J. Energy Storage 2020).
- ``two_segment``: exhaustive two-segment least-squares change-point fit, the
  optimal form of the intersection-based family.
- ``bisector``: intersection of a line fitted to the early fade and a line
  fitted to the late fade (used e.g. in Batteries 2025; surveyed in
  arXiv:2501.14573).
- ``slope_change_ratio``: knee at the cycle maximising the ratio of
  post- to pre-window fade slopes (slope-changing-ratio method, Diao et al.,
  as surveyed in arXiv:2501.14573).
- ``curvature``: knee at the maximum of the discrete curvature of the
  smoothed fade curve (curvature-based idea from arXiv:2304.11671).
- ``kneedle``: knee as the point of maximum distance from the chord on the
  normalised curve (Satopaa et al., ICDCSW 2011).
"""

from __future__ import annotations

import numpy as np


def _as_arrays(cycles, capacity):
    x = np.asarray(cycles, dtype=float).ravel()
    y = np.asarray(capacity, dtype=float).ravel()
    if x.shape != y.shape or x.size < 10:
        raise ValueError("cycles and capacity must be equal-length arrays of >= 10 points")
    order = np.argsort(x)
    return x[order], y[order]


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def bacon_watts(cycles, capacity, gamma=None, n_grid: int = 60) -> float:
    """Bacon-Watts change-point fit.

    Model: ``y = b0 + b1*(x-x0) + b2*(x-x0)*tanh((x-x0)/gamma)``.
    For each candidate change point ``x0`` on a grid the linear parameters
    are solved by least squares (profile likelihood); the knee is the ``x0``
    with the smallest residual sum of squares.
    """
    x, y = _as_arrays(cycles, capacity)
    if np.allclose(y, y[0]):
        return np.nan
    span = x[-1] - x[0]
    if gamma is None:
        gamma = 0.05 * span
    lo, hi = x[0] + 0.15 * span, x[-1] - 0.15 * span
    grid = np.linspace(lo, hi, n_grid)
    best_sse, best_x0 = np.inf, np.nan
    for x0 in grid:
        d = x - x0
        phi = d * np.tanh(d / gamma)
        A = np.column_stack([np.ones_like(x), d, phi])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if sse < best_sse:
            best_sse, best_x0 = sse, x0
    return float(best_x0)


def two_segment(cycles, capacity, n_grid: int = 200) -> float:
    """Exhaustive two-segment least-squares change-point detection."""
    x, y = _as_arrays(cycles, capacity)
    if np.allclose(y, y[0]):
        return np.nan
    span = x[-1] - x[0]
    lo, hi = x[0] + 0.10 * span, x[-1] - 0.10 * span
    grid = np.linspace(lo, hi, n_grid)
    best_sse, best_k = np.inf, np.nan
    for k in grid:
        left = x <= k
        if left.sum() < 3 or (~left).sum() < 3:
            continue
        sse = 0.0
        for mask in (left, ~left):
            A = np.column_stack([np.ones(mask.sum()), x[mask]])
            coef, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
            sse += float(np.sum((A @ coef - y[mask]) ** 2))
        if sse < best_sse:
            best_sse, best_k = sse, k
    return float(best_k)


def bisector(cycles, capacity, head: float = 0.30, tail: float = 0.30) -> float:
    """Intersection of early-fade and late-fade linear fits."""
    x, y = _as_arrays(cycles, capacity)
    if np.allclose(y, y[0]):
        return np.nan
    span = x[-1] - x[0]
    m1 = x <= x[0] + head * span
    m2 = x >= x[-1] - tail * span
    if m1.sum() < 3 or m2.sum() < 3:
        return np.nan
    A1 = np.column_stack([np.ones(m1.sum()), x[m1]])
    A2 = np.column_stack([np.ones(m2.sum()), x[m2]])
    (b1, a1), *_ = np.linalg.lstsq(A1, y[m1], rcond=None)  # y = b1 + a1*x
    (b2, a2), *_ = np.linalg.lstsq(A2, y[m2], rcond=None)
    if np.isclose(a1, a2):
        return np.nan
    x_star = (b2 - b1) / (a1 - a2)
    if not (x[0] <= x_star <= x[-1]):
        return np.nan
    return float(x_star)


def slope_change_ratio(cycles, capacity, window_frac: float = 0.05,
                       smooth_frac: float = 0.02, min_ratio: float = 1.5) -> float:
    """Knee at the maximum post/pre fade-slope ratio.

    The curve is smoothed, local slopes are estimated on each side of every
    candidate cycle over a window, and the knee maximises
    ``slope_after / slope_before`` (both negative for fade; a ratio well
    above 1 marks accelerated degradation).
    """
    x, y = _as_arrays(cycles, capacity)
    if np.allclose(y, y[0]):
        return np.nan
    n = x.size
    ys = _moving_average(y, max(3, int(n * smooth_frac)))
    slope = np.gradient(ys, x)
    w = max(5, int(n * window_frac))
    span = x[-1] - x[0]
    ratios = np.full(n, np.nan)
    for i in range(w, n - w):
        if not (x[0] + 0.1 * span <= x[i] <= x[-1] - 0.1 * span):
            continue
        s_before = np.mean(slope[i - w:i])
        s_after = np.mean(slope[i:i + w])
        if s_before >= 0 or s_after >= 0:  # not a fade regime here
            continue
        ratios[i] = s_after / s_before
    if np.all(np.isnan(ratios)):
        return np.nan
    i_star = int(np.nanargmax(ratios))
    if ratios[i_star] < min_ratio:
        return np.nan
    return float(x[i_star])


def curvature(cycles, capacity, smooth_frac: float = 0.10) -> float:
    """Knee at the maximum discrete curvature of the smoothed fade curve.

    Uses a Savitzky-Golay filter (which preserves the transition shape while
    suppressing noise) before computing discrete curvature on normalised
    coordinates.  Second derivatives amplify measurement noise, so this
    detector is inherently the most noise-sensitive of the set -- a finding
    the benchmark makes quantitative.
    """
    from scipy.signal import savgol_filter

    x, y = _as_arrays(cycles, capacity)
    if np.allclose(y, y[0]):
        return np.nan
    n = x.size
    window = max(11, int(n * smooth_frac))
    window += 1 - window % 2  # savgol needs an odd window
    ys = savgol_filter(y, window, 2)
    # Normalise so curvature is scale-invariant.
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (ys - ys.min()) / (ys.max() - ys.min()) if ys.max() > ys.min() else ys
    yp = np.gradient(yn, xn)
    ypp = np.gradient(yp, xn)
    kappa = np.abs(ypp) / (1.0 + yp ** 2) ** 1.5
    lo, hi = int(0.1 * n), int(0.9 * n)
    i_star = lo + int(np.argmax(kappa[lo:hi]))
    return float(x[i_star])


def kneedle(cycles, capacity, sensitivity: float = 1.0) -> float:
    """Kneedle algorithm on the normalised fade curve.

    For a decreasing, downward-accelerating fade curve the knee is the point
    of maximum distance between the curve and the chord joining its
    endpoints (Satopaa et al., ICDCSW 2011).
    """
    x, y = _as_arrays(cycles, capacity)
    if np.allclose(y, y[0]):
        return np.nan
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[0]) / (y[-1] - y[0]) if not np.isclose(y[-1], y[0]) else y * 0
    # Chord from (0,0) to (1,1) in normalised coords; distance ~ (chord - curve).
    distance = (xn - yn)
    distance[:max(1, int(0.05 * x.size))] = -np.inf
    distance[max(1, int(0.95 * x.size)):] = -np.inf
    i_star = int(np.argmax(distance))
    if distance[i_star] < 0.02 * sensitivity:
        return np.nan
    return float(x[i_star])


DETECTORS = {
    "bacon_watts": bacon_watts,
    "two_segment": two_segment,
    "bisector": bisector,
    "slope_change_ratio": slope_change_ratio,
    "curvature": curvature,
    "kneedle": kneedle,
}
"""Registry of all available detectors: name -> function."""
