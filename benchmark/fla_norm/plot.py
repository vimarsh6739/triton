"""Plot the medians and interquartile ranges of the complete Q/K normalization work."""
import argparse
import csv
from collections import defaultdict
from statistics import median, quantiles

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('timings')
    parser.add_argument('output')
    args = parser.parse_args()
    groups = defaultdict(list)
    with open(args.timings) as f:
        for r in csv.DictReader(f):
            groups[r['case'], r['method']].append(float(r['us']))
    cases = list(dict.fromkeys(case for case, _ in groups))
    methods = [('combined', 'Original forward + combined AD'), ('paired-fla', 'Paired handwritten'),
               ('paired-ad', 'Paired saved-state AD'), ('paired-ad-fold', 'Paired saved-state AD + fold')]
    fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
    for i, (method, label) in enumerate(methods):
        centers, low, high = [], [], []
        for case in cases:
            ref = median(groups[case, 'fla'])
            qs = quantiles(groups[case, method], n=4)
            center = median(groups[case, method])
            centers.append(center / ref)
            low.append((center - qs[0]) / ref)
            high.append((qs[-1] - center) / ref)
        ax.errorbar([j + .09 * (i - 1.5) for j in range(len(cases))], centers, yerr=[low, high], fmt='o',
                    label=label, capsize=2)
    ax.axhline(1, color='gray', linestyle='--', label='Upstream FLA')
    ax.set_xticks(range(len(cases)), cases)
    ax.set_ylabel('Time divided by upstream time')
    ax.set_title('Q and K normalization, forward and backward on RTX 5090')
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(args.output, dpi=160)


if __name__ == '__main__':
    main()
