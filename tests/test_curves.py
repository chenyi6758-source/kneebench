"""Tests for synthetic curve generation."""

import numpy as np
import pandas as pd
import pytest

from kneebench.curves import default_suite, fade_curve, load_csv


def test_ground_truth_attrs():
    df = fade_curve(n_cycles=1200, knee_onset=700, transition=30)
    assert df.attrs["true_knee_onset"] == 700.0
    assert df.attrs["true_knee"] == 730.0
    assert list(df.columns) == ["cycle", "capacity"]
    assert len(df) == 1200


def test_reproducible_with_seed():
    a = fade_curve(seed=42)
    b = fade_curve(seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = fade_curve(seed=1)
    b = fade_curve(seed=2)
    assert not np.allclose(a["capacity"], b["capacity"])


def test_overall_decreasing_trend():
    # Not strictly monotonic (noise), but end well below start.
    df = fade_curve(noise=2e-4, seed=0)
    assert df["capacity"].iloc[-1] < df["capacity"].iloc[0] - 0.05


def test_sqrt_early_shape():
    df = fade_curve(early_shape="sqrt", seed=0)
    assert df.attrs["true_knee"] == 730.0
    # With the same rate coefficient, sqrt fade is slower than linear fade
    # for n > 1; check the shape property on a noise-free curve instead:
    # sqrt early fade is concave (slope magnitude decreases over time).
    clean = fade_curve(early_shape="sqrt", noise=0.0, seed=0)
    y = clean["capacity"].to_numpy()
    early_slope = np.polyfit(np.arange(100, 300), y[100:300], 1)[0]
    late_slope = np.polyfit(np.arange(400, 600), y[400:600], 1)[0]
    assert early_slope < late_slope < 0  # concave: steep first, flatter later


def test_invalid_args():
    with pytest.raises(ValueError):
        fade_curve(n_cycles=100, knee_onset=150)
    with pytest.raises(ValueError):
        fade_curve(early_shape="cubic")


def test_default_suite_builds():
    suite = default_suite()
    assert len(suite) >= 5
    for cfg in suite:
        cfg = dict(cfg)
        cfg.pop("name")
        df = fade_curve(**cfg)
        assert np.isfinite(df.attrs["true_knee"])


def test_load_csv(tmp_path):
    df = fade_curve(seed=0)
    p = tmp_path / "curve.csv"
    df[["cycle", "capacity"]].to_csv(p, index=False)
    loaded = load_csv(str(p))
    assert list(loaded.columns) == ["cycle", "capacity"]
    assert len(loaded) == len(df)
