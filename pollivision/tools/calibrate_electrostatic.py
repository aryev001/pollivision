#!/usr/bin/env python3
"""Fit the adaptive controller's gains to measured transfer efficiency.

Sec. III-F describes the controller as "a rule-based mapping refined through
empirical calibration". This tool performs that refinement. The defaults shipped
in ``configs/default.yaml`` are physically motivated starting points, not
measured constants, and a deployed rover should replace them with coefficients
fitted to its own probe geometry and its own crop.

Input is a CSV of pollination trials::

    humidity,standoff_m,incidence_deg,pollen_availability,voltage_kv,exposure_ms,transferred

where ``transferred`` is the measured outcome - a 0/1 success flag, or a
continuous transfer efficiency in [0, 1] if you can measure deposited grain
counts.

The fit is a coordinate search over the gain parameters that maximises the
agreement between the controller's *predicted* effective field-time product and
the observed transfer. It is deliberately a coarse search rather than a gradient
method: the dataset from a realistic calibration campaign is small and noisy,
and a coarse fit over a physically-parameterised model generalises better than a
precise fit to fifty noisy trials.

Collect at least ~40 trials spanning the operating envelope - both humidity
extremes, the full standoff range, and a spread of incidence angles - or the fit
will be unconstrained in whichever direction you did not vary.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pollivision.config import load_config  # noqa: E402
from pollivision.control.electrostatic import AdaptiveElectrostaticController  # noqa: E402

REQUIRED = ["humidity", "standoff_m", "incidence_deg", "pollen_availability",
            "voltage_kv", "exposure_ms", "transferred"]


def load_trials(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing column(s): {missing}")
    return [{k: float(row[k]) for k in REQUIRED} for row in rows]


def predicted_dose(controller, trial: dict, gains) -> float:
    """Model the delivered dose for a trial under a candidate gain set.

    The controller's premise is that transfer scales with the *useful* field at
    the flower - the surface field projected onto the corolla normal - sustained
    over the exposure. This computes that quantity for the parameters the trial
    actually used, so fitting it against the observed outcome tests the premise
    rather than assuming it.
    """
    from pollivision.control.safety import target_field_kv_per_m

    controller.gains = gains
    field = target_field_kv_per_m(trial["voltage_kv"], trial["standoff_m"],
                                  controller.tip_radius)
    projection = max(np.cos(np.radians(trial["incidence_deg"])),
                     gains.incidence_min_cos)
    # Humidity bleeds induced charge away; the same logistic the controller uses
    # to compensate is used here to model the loss it is compensating for.
    retention = 1.0 / controller.humidity_factor(trial["humidity"])
    availability = np.clip(trial["pollen_availability"], 0.0, 1.0)
    return float(field * projection * retention
                 * (trial["exposure_ms"] / 1000.0) * availability)


def correlation(predictions: np.ndarray, observed: np.ndarray) -> float:
    if predictions.std() < 1e-9 or observed.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(predictions, observed)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", help="CSV of pollination trials")
    parser.add_argument("--config", default="default")
    parser.add_argument("--output", default="calibrated_gains.yaml")
    args = parser.parse_args()

    trials = load_trials(Path(args.trials))
    print(f"Loaded {len(trials)} trials")
    if len(trials) < 20:
        print("Warning: fewer than 20 trials. The fit will be poorly constrained.")

    for column in ("humidity", "standoff_m", "incidence_deg"):
        values = np.array([t[column] for t in trials])
        span = values.max() - values.min()
        print(f"  {column:<18} range {values.min():.3f} to {values.max():.3f}")
        if span < 1e-6:
            print(f"    ^ this variable was never varied; its gain cannot be fitted")

    cfg = load_config(args.config)
    controller = AdaptiveElectrostaticController(cfg)
    observed = np.array([t["transferred"] for t in trials])

    from pollivision.control.electrostatic import ControllerGains

    baseline = predicted_dose_all(controller, trials, controller.gains)
    print(f"\nBaseline correlation with default gains: {correlation(baseline, observed):+.4f}")

    best_gains, best_score = controller.gains, correlation(baseline, observed)
    grid = {
        "distance_exponent": [1.0, 1.3, 1.6, 1.9, 2.2],
        "humidity_midpoint": [55.0, 62.0, 68.0, 75.0, 82.0],
        "humidity_steepness": [0.05, 0.09, 0.14, 0.20],
        "incidence_min_cos": [0.20, 0.35, 0.50],
    }
    keys = list(grid)
    total = int(np.prod([len(grid[k]) for k in keys]))
    print(f"Searching {total} gain combinations...")

    for index, values in enumerate(itertools.product(*(grid[k] for k in keys))):
        candidate = ControllerGains(**{**vars(controller.gains),
                                       **dict(zip(keys, values))})
        score = correlation(predicted_dose_all(controller, trials, candidate), observed)
        if score > best_score:
            best_gains, best_score = candidate, score
        if (index + 1) % max(1, total // 10) == 0:
            print(f"  {index + 1}/{total}  best {best_score:+.4f}")

    print(f"\nBest correlation: {best_score:+.4f}")
    if best_score < 0.3:
        print("This is a weak fit. Either the trials do not span enough of the\n"
              "operating envelope, or transfer in your setup is dominated by\n"
              "something this model does not capture (probe geometry, pollen\n"
              "adhesion, air movement). Treat the fitted gains with suspicion.")

    lines = ["# Fitted by tools/calibrate_electrostatic.py",
             f"# {len(trials)} trials, correlation {best_score:+.4f}",
             "electrostatic:", "  gains:"]
    for key, value in vars(best_gains).items():
        lines.append(f"    {key}: {value}")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nWrote {args.output}. Apply it with:")
    print(f"  pollivision run <source> --config default --set-file {args.output}")
    print("or merge it into your species profile.")
    return 0


def predicted_dose_all(controller, trials, gains) -> np.ndarray:
    original = controller.gains
    try:
        return np.array([predicted_dose(controller, t, gains) for t in trials])
    finally:
        controller.gains = original


if __name__ == "__main__":
    sys.exit(main())
