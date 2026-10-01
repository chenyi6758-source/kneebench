"""Benchmark runner: every detector over every curve in a suite."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .curves import default_suite, fade_curve
from .detectors import DETECTORS
from .metrics import absolute_error, relative_error, summarize


def run_suite(configs: list[dict] | None = None,
              detectors: dict | None = None,
              verbose: bool = True) -> pd.DataFrame:
    """Run all detectors on all suite curves.

    Returns a DataFrame with one row per (curve, detector) pair: detected
    knee, ground truth, absolute/relative errors and runtime.
    """
    configs = configs if configs is not None else default_suite()
    detectors = detectors if detectors is not None else DETECTORS
    rows = []
    for cfg in configs:
        name = cfg.get("name", "curve")
        df = fade_curve(**{k: v for k, v in cfg.items() if k != "name"})
        true_knee = float(df.attrs["true_knee"])
        x = df["cycle"].to_numpy()
        y = df["capacity"].to_numpy()
        for det_name, func in detectors.items():
            t0 = time.perf_counter()
            try:
                detected = func(x, y)
            except Exception:  # a detector must never kill the benchmark
                detected = np.nan
            dt_ms = (time.perf_counter() - t0) * 1000.0
            rows.append({
                "curve": name,
                "detector": det_name,
                "detected": float(detected) if detected is not None else np.nan,
                "true": true_knee,
                "abs_err": absolute_error(detected, true_knee),
                "rel_err": relative_error(detected, true_knee),
                "runtime_ms": dt_ms,
            })
        if verbose:
            print(f"  done: {name} (true knee = {true_knee:.0f})")
    return pd.DataFrame(rows)


def save_report(results: pd.DataFrame, path: str = "kneebench_report.csv") -> pd.DataFrame:
    """Write the detailed results CSV and return the summary table."""
    results.to_csv(path, index=False)
    summary = summarize(results)
    summary_path = path.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"wrote {path} and {summary_path}")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark knee-point detectors on synthetic fade curves.")
    parser.add_argument("--out", default="kneebench_report.csv",
                        help="output CSV path for detailed results")
    args = parser.parse_args()
    print("KneeBench: running benchmark suite ...")
    results = run_suite()
    save_report(results, args.out)


if __name__ == "__main__":
    main()
