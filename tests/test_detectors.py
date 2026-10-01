"""Tests for knee-point detectors on synthetic curves with known knees."""

import numpy as np
import pytest

from kneebench.curves import fade_curve
from kneebench.detectors import DETECTORS

# Tolerances (cycles) on the canonical 1200-cycle curve, true knee = 730.
# Set ~2-3x above measured errors to be robust without being vacuous.
TOLS = {
    "bacon_watts": 40,
    "two_segment": 60,
    "bisector": 25,
    "slope_change_ratio": 60,
    "curvature": 80,
    "kneedle": 40,
}


@pytest.fixture(scope="module")
def canonical():
    df = fade_curve()  # seed=0, knee_onset=700, transition=30
    x = df["cycle"].to_numpy()
    y = df["capacity"].to_numpy()
    return x, y, df.attrs["true_knee"]


def test_all_detectors_registered():
    assert set(DETECTORS) == set(TOLS)


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_accuracy_canonical(name, canonical):
    x, y, true_knee = canonical
    detected = DETECTORS[name](x, y)
    assert np.isfinite(detected), f"{name} failed on canonical curve"
    assert abs(detected - true_knee) <= TOLS[name], (
        f"{name}: detected {detected:.1f}, true {true_knee:.0f}, "
        f"err {abs(detected - true_knee):.1f} > tol {TOLS[name]}")


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_robustness_early_knee(name):
    # A second, independent configuration with a much earlier knee.
    df = fade_curve(n_cycles=1200, knee_onset=350, transition=30, seed=1)
    x = df["cycle"].to_numpy()
    y = df["capacity"].to_numpy()
    true_knee = df.attrs["true_knee"]
    detected = DETECTORS[name](x, y)
    assert np.isfinite(detected), f"{name} failed on early-knee curve"
    assert abs(detected - true_knee) <= 100, (
        f"{name}: err {abs(detected - true_knee):.1f} too large on early-knee")


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_nan_on_constant_curve(name, canonical):
    x, y, _ = canonical
    assert np.isnan(DETECTORS[name](x, np.ones_like(y)))


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detector_rejects_short_input(name):
    with pytest.raises(ValueError):
        DETECTORS[name](np.arange(5.0), np.ones(5))
