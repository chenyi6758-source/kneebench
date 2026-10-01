"""KneeBench: benchmark harness for battery capacity-fade knee-point detection."""

from .benchmark import run_suite, save_report
from .curves import default_suite, fade_curve, load_csv
from .detectors import DETECTORS
from .metrics import absolute_error, relative_error, summarize
from .plot import plot_comparison, plot_error_bars

__all__ = [
    "run_suite",
    "save_report",
    "default_suite",
    "fade_curve",
    "load_csv",
    "DETECTORS",
    "absolute_error",
    "relative_error",
    "summarize",
    "plot_comparison",
    "plot_error_bars",
]

__version__ = "0.1.0"
