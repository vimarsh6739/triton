"""Plot measured latency and PyTorch allocations for one training shape."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).parent
with (root / "summary.csv").open() as f:
    latency = {r["impl"]: float(r["median_us"]) for r in csv.DictReader(f) if int(r["rows"]) == 3072}
with (root / "backward-memory.csv").open() as f:
    memory = {r["mode"]: int(r["peak_increment_bytes"]) / 2**20 for r in csv.DictReader(f)}
modes = ["generated", "discard", "read_seed", "overwrite", "donate", "liger"]
labels = ["Combined", "Discard primal", "Keep seed", "Write gradients", "Reuse inputs", "Liger"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True, layout="constrained")
for axis, data, xlabel in zip(axes, (latency, memory), ("Forward + backward (µs)", "Backward peak increment (MiB)")):
    bars = axis.barh(labels, [data[m] for m in modes])
    axis.bar_label(bars, fmt="%.0f", padding=3)
    axis.set_xlabel(xlabel)
    axis.set_xlim(0, max(data.values()) * 1.16)
axes[0].invert_yaxis()
fig.suptitle("Liger SwiGLU · 3,072 × 12,288 BF16 · RTX 5090")
fig.savefig(root / "writeback.png", dpi=160)
