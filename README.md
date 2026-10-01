# KneeBench

Benchmark harness for battery capacity-fade **knee-point detection** algorithms — six literature methods, one synthetic ground-truth suite, one command.

## Problem

The knee — the cycle where a lithium-ion cell's capacity fade switches from slow to accelerated — is the single most informative marker of battery lifetime. It gates end-of-life estimates, second-life screening, and warranty decisions (Severson et al., *Nature Energy* 2019).

The literature has no shortage of detection algorithms (Bacon-Watts, bisector, slope-changing-ratio, curvature-based, Kneedle, …), but every paper validates its own method on its own dataset. If you need to *choose* an algorithm for your BMS analytics or your aging study, there is no maintained, dependency-light harness that runs the established methods head-to-head on curves with known ground truth. KneeBench is that harness.

## Method

**Synthetic ground truth** (`kneebench/curves.py`). Two-phase fade curves with an analytically known knee: a slow early regime (linear, or √t-shaped to mimic diffusion-limited SEI growth), a smooth transition window, and an accelerated late regime, plus Gaussian measurement noise. Following the curvature literature, the generator reports both `true_knee_onset` (transition start) and `true_knee` (transition end). Real data can be loaded from CSV with `load_csv` (no ground truth — for analysis, not scoring).

**Six detectors** (`kneebench/detectors.py`), each from the literature:

| Detector | Idea | Source |
|---|---|---|
| `bacon_watts` | Bacon-Watts change-point model, profile-likelihood fit | Fermín-Cueto et al., *J. Energy Storage* 2020 |
| `two_segment` | Exhaustive two-segment least-squares change point | intersection-based family, optimal form |
| `bisector` | Intersection of early-fade and late-fade linear fits | e.g. *Batteries* 2025; surveyed in arXiv:2501.14573 |
| `slope_change_ratio` | Max ratio of post- to pre-window fade slopes | Diao et al., surveyed in arXiv:2501.14573 |
| `curvature` | Max discrete curvature of the Savitzky-Golay-smoothed curve | curvature idea from arXiv:2304.11671 |
| `kneedle` | Max distance from the chord on the normalised curve | Satopaa et al., ICDCSW 2011 |

**Benchmark** (`kneebench/benchmark.py`). Runs every detector over a six-curve suite (early/late knee, √t early fade, gradual transition, high noise) and reports absolute/relative cycle error, failure rate, and runtime. A detector that raises never kills the run — it scores as failed.

## Quickstart

```bash
pip install -e .
python examples/quickstart.py
```

This detects knees on one curve with all six methods, then runs the full suite and writes `examples/quickstart_report.csv` (+ `_summary.csv`) and two PNGs (`quickstart_comparison.png`, `quickstart_errors.png`).

Or as a library:

```python
from kneebench import fade_curve, DETECTORS

df = fade_curve(n_cycles=1200, knee_onset=700, transition=30, seed=0)
x, y = df["cycle"].to_numpy(), df["capacity"].to_numpy()
true_knee = df.attrs["true_knee"]          # 730.0, known by construction

for name, func in DETECTORS.items():
    det = func(x, y)
    print(f"{name:18s} -> {det:7.1f}  (err {abs(det - true_knee):5.1f} cycles)")
```

CLI benchmark:

```bash
python -m kneebench.benchmark --out my_report.csv
```

## Example results

Suite: 6 synthetic curves (1200 cycles each, true knees 380–980), all detectors, zero failures:

| detector | median abs err (cycles) | max abs err (cycles) | mean runtime |
|---|---|---|---|
| bisector | 0.3 | 60.0 | 0.3 ms |
| bacon_watts | 8.6 | 13.2 | 6.2 ms |
| kneedle | 9.0 | 40.0 | 0.2 ms |
| slope_change_ratio | 12.5 | 26.0 | 10.7 ms |
| two_segment | 16.2 | 58.0 | 23.1 ms |
| curvature | 27.5 | 64.0 | 0.9 ms |

Honest findings, not just a leaderboard: the bisector is the most accurate here but degrades when the knee sits very late; `two_segment` assumes a sharp change point and suffers on gradual transitions; `curvature` is the most noise-sensitive (second derivatives amplify noise — the Savitzky-Golay window is doing real work). That is exactly the kind of trade-off this harness exists to quantify.

## Tests

```bash
pytest          # 39 tests, ~2 s, no network, no heavy dependencies
```

Core logic (every detector) is covered on synthetic curves with known knees, plus edge cases (constant curve → `NaN`, short input → `ValueError`) and an end-to-end benchmark + plot run.

## Scope and limits

- KneeBench benchmarks *detection* on capacity-vs-cycle curves, not early *prediction* from the first 100 cycles (see Severson/Attia for that problem) and not the underlying degradation physics (see [PyBaMM](https://github.com/pybamm-team/PyBaMM)).
- Synthetic curves are deliberately simple two-phase models. They are a controlled test bed, not a substitute for validating on your own cell data — `load_csv` is there for exactly that.

## References

- K. A. Severson et al., "Data-driven prediction of battery cycle life before capacity degradation", *Nature Energy* 4, 383–391 (2019).
- P. Fermín-Cueto et al., "Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells", *J. Energy Storage* 32, 101883 (2020). Code: https://github.com/pfermined/knee_identification
- "Battery Capacity Knee-Onset Identification and Early Prediction Using Degradation Curvature", arXiv:2304.11671 (2023).
- "A Transferable Physics-Informed Framework for Battery Degradation Diagnosis, Knee-Onset Detection and Knee Prediction", arXiv:2501.14573 (2025).
- V. Satopaa et al., "Finding a 'Kneedle' in a Haystack: Detecting knee points in system behavior", ICDCSW 2011.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
