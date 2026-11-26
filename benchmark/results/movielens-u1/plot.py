"""Plot recorded training curves and held-out MovieLens RMSE."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    fig, (train, test) = plt.subplots(1, 2, figsize=(8.2, 3.5), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    rank, = {r["rank"] for r in rows}
    baselines = set()
    for seed in dict.fromkeys(r["seed"] for r in rows):
        series = [r for r in rows if r["seed"] == seed]
        train.plot([int(r["epoch"]) for r in series], [float(r["train_rmse"]) for r in series],
                   marker=".", label=f"Seed {seed}")
        final, = [r for r in series if r["test_rmse"]]
        test.plot(seed, float(final["test_rmse"]), "o")
        baselines.add(float(final["train_mean_test_rmse"]))
    baseline, = baselines
    test.axhline(baseline, color="0.35", ls="--", label="Training-mean predictor")
    train.set(xlabel="Training epoch", ylabel="Training RMSE")
    test.set(xlabel="Initialization seed", ylabel="Test RMSE")
    for panel in (train, test):
        panel.grid(axis="y", alpha=.25)
        panel.legend(fontsize=8)
    fig.suptitle(f"MovieLens 100K · rank-{rank} user × item × week factorization")
    fig.savefig(args.image, dpi=180)


if __name__ == "__main__":
    main()
