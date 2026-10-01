"""KneeBench quickstart: detect knees on one curve, then run the full benchmark.

Run with:  python examples/quickstart.py
Outputs:   examples/quickstart_comparison.png, examples/quickstart_errors.png,
           examples/quickstart_report.csv (+ _summary.csv)
"""

import os

import matplotlib

matplotlib.use("Agg")

from kneebench import (
    DETECTORS,
    fade_curve,
    plot_comparison,
    plot_error_bars,
    run_suite,
    save_report,
    summarize,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. One synthetic curve with a known knee at cycle 730.
    df = fade_curve(n_cycles=1200, knee_onset=700, transition=30, seed=0)
    x = df["cycle"].to_numpy()
    y = df["capacity"].to_numpy()
    true_knee = df.attrs["true_knee"]

    detections = {name: func(x, y) for name, func in DETECTORS.items()}
    print(f"True knee: {true_knee:.0f}")
    for name, det in detections.items():
        print(f"  {name:18s} -> {det:7.1f}  (err {abs(det - true_knee):5.1f} cycles)")

    plot_comparison(x, y, detections, true_knee,
                    title="KneeBench quickstart: six detectors, one curve",
                    save_path=os.path.join(HERE, "quickstart_comparison.png"))

    # 2. Full benchmark suite: every detector on every curve.
    results = run_suite(verbose=False)
    summary = save_report(results, os.path.join(HERE, "quickstart_report.csv"))
    plot_error_bars(summary,
                    save_path=os.path.join(HERE, "quickstart_errors.png"))
    print("\nDone. See examples/quickstart_*.png and quickstart_report*.csv")


if __name__ == "__main__":
    main()
