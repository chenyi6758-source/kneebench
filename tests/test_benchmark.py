"""Tests for metrics and the benchmark runner."""

import numpy as np
import pandas as pd
import pytest

from kneebench import (
    DETECTORS,
    absolute_error,
    plot_comparison,
    plot_error_bars,
    relative_error,
    run_suite,
    save_report,
    summarize,
)
from kneebench.curves import fade_curve


def test_absolute_error():
    assert absolute_error(730.0, 700.0) == 30.0
    assert np.isnan(absolute_error(np.nan, 700.0))
    assert np.isnan(absolute_error(None, 700.0))


def test_relative_error():
    assert relative_error(770.0, 700.0) == pytest.approx(70.0 / 700.0)
    assert np.isnan(relative_error(np.nan, 700.0))


def test_summarize():
    results = pd.DataFrame([
        {"detector": "a", "abs_err": 10.0, "runtime_ms": 1.0},
        {"detector": "a", "abs_err": np.nan, "runtime_ms": 3.0},
    ])
    results = pd.concat([results, pd.DataFrame([
        {"detector": "a", "abs_err": 20.0, "runtime_ms": 2.0},
    ])], ignore_index=True)
    s = summarize(results)
    a = s[s["detector"] == "a"].iloc[0]
    b = s[s["detector"] == "b"].iloc[0]
    assert a["mean_abs_err"] == pytest.approx(15.0)
    assert a["fail_rate"] == pytest.approx(0.0)
    assert b["fail_rate"] == pytest.approx(1.0)
    assert np.isnan(b["mean_abs_err"])


def test_run_suite_tiny():
    configs = [
        {"name": "c1", "n_cycles": 600, "knee_onset": 350,
         "transition": 20, "seed": 0},
        {"name": "c2", "n_cycles": 600, "knee_onset": 400,
         "transition": 20, "seed": 1},
    ]
    dets = {"bisector": DETECTORS["bisector"], "kneedle": DETECTORS["kneedle"]}
    res = run_suite(configs=configs, detectors=dets, verbose=False)
    assert len(res) == 4
    assert set(res.columns) >= {"curve", "detector", "detected", "true",
                                "abs_err", "rel_err", "runtime_ms"}
    assert res["abs_err"].notna().all()


def test_save_report(tmp_path):
    res = run_suite(
        configs=[{"name": "c1", "n_cycles": 600, "knee_onset": 350,
                  "transition": 20, "seed": 0}],
        detectors={"bisector": DETECTORS["bisector"]},
        verbose=False)
    out = str(tmp_path / "report.csv")
    summary = save_report(res, out)
    assert (tmp_path / "report.csv").exists()
    assert (tmp_path / "report_summary.csv").exists()
    assert len(summary) == 1


def test_plots_run(tmp_path):
    df = fade_curve(seed=0)
    x = df["cycle"].to_numpy()
    y = df["capacity"].to_numpy()
    dets = {n: f(x, y) for n, f in DETECTORS.items()}
    plot_comparison(x, y, dets, df.attrs["true_knee"],
                    save_path=str(tmp_path / "comp.png"))
    assert (tmp_path / "comp.png").exists()
    plot_error_bars(
        summarize(run_suite(
            configs=[{"name": "c1", "n_cycles": 600, "knee_onset": 350,
                      "transition": 20, "seed": 0}],
            detectors={"bisector": DETECTORS["bisector"],
                       "kneedle": DETECTORS["kneedle"]},
            verbose=False)),
        save_path=str(tmp_path / "bars.png"))
    assert (tmp_path / "bars.png").exists()
