"""Error metrics for knee-point detection results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_error(detected: float, true: float) -> float:
    """Absolute error in cycles; NaN if the detector failed."""
    if detected is None or (isinstance(detected, float) and np.isnan(detected)):
        return np.nan
    return float(abs(detected - true))


def relative_error(detected: float, true: float) -> float:
    """Absolute error normalised by the true knee position."""
    err = absolute_error(detected, true)
    if np.isnan(err) or true == 0:
        return np.nan
    return err / abs(true)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-detector benchmark statistics.

    Expects columns ``detector``, ``abs_err`` and ``runtime_ms``.
    Returns one row per detector with mean/median error, failure rate and
    mean runtime.
    """
    rows = []
    for name, grp in results.groupby("detector"):
        errs = grp["abs_err"].to_numpy(dtype=float)
        ok = ~np.isnan(errs)
        rows.append({
            "detector": name,
            "n": int(len(grp)),
            "fail_rate": float(1.0 - ok.mean()) if len(grp) else np.nan,
            "mean_abs_err": float(np.nanmean(errs)) if ok.any() else np.nan,
            "median_abs_err": float(np.nanmedian(errs)) if ok.any() else np.nan,
            "max_abs_err": float(np.nanmax(errs)) if ok.any() else np.nan,
            "mean_runtime_ms": float(grp["runtime_ms"].mean()),
        })
    out = pd.DataFrame(rows).sort_values("median_abs_err").reset_index(drop=True)
    return out
