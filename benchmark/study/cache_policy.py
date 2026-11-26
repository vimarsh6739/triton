"""Evaluate calibrated softmax schedules with an optional common-mode error budget."""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


@dataclass
class Measurement:
    times: list[float] = field(default_factory=list)
    error: float = 0

    def add(self, row):
        self.times.append(float(row["us"]))
        self.error = max(self.error, float(row["common_mode_error"]))


def pick(candidates, budget):
    eligible = {name: data for name, data in candidates.items() if budget is None or data.error <= budget}
    return min(eligible, key=lambda name: median(eligible[name].times), default=None)


def compare(timings, output, budget):
    phases = defaultdict(lambda: defaultdict(lambda: defaultdict(Measurement)))
    with timings.open() as stream:
        for row in csv.DictReader(stream):
            shape = int(row["cols"]), row["mode"]
            name = f'{row["schedule"]}-w{row["warps"]}'
            phases[shape][row["phase"]][name].add(row)

    with output.open("w", newline="") as stream:
        out = csv.writer(stream, lineterminator="\n")
        out.writerow(["cols", "mode", "common_mode_budget", "selected", "calibration_error",
                      "holdout_error", "holdout_us", "best_eligible_holdout_us", "holdout_over_best"])
        for (cols, mode), split in sorted(phases.items()):
            calibration, holdout = split["calibrate"], split["holdout"]
            if not calibration or not holdout:
                raise ValueError(f"Missing calibration or holdout for {cols} {mode}")
            for limit in (None, budget):
                name = pick(calibration, limit)
                best = pick(holdout, limit)
                best_us = median(holdout[best].times) if best else None
                if name is None or name not in holdout:
                    out.writerow([cols, mode, limit, name, calibration[name].error if name else None,
                                  None, None, best_us, None])
                    continue
                observed = holdout[name]
                latency = median(observed.times)
                passed = limit is None or observed.error <= limit
                out.writerow([cols, mode, limit, name, calibration[name].error, observed.error,
                              latency, best_us, latency / best_us if best_us and passed else None])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timings", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=float, default=1e-5)
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("The error budget must be nonnegative")
    compare(args.timings, args.output, args.budget)
