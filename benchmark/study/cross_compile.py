"""Lower the same already-differentiated TTIR files for NVIDIA and AMD GPUs."""

import argparse
import csv
import hashlib
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget


TARGETS = [GPUTarget("cuda", 80, 32), GPUTarget("cuda", 90, 32),
           GPUTarget("cuda", 120, 32), GPUTarget("hip", "gfx90a", 64), GPUTarget("hip", "gfx942", 64)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ttir", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warps", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "binaries.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["module", "ttir_sha256", "backend", "arch", "num_warps", "binary_bytes", "binary_sha256"])
        for path in args.ttir:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for target in TARGETS:
                kernel = triton.compile(str(path), target=target, options={"num_warps": args.warps})
                binary = kernel.kernel
                writer.writerow([path.name, digest, target.backend, target.arch, args.warps, len(binary),
                                 hashlib.sha256(binary).hexdigest()])
                stream.flush()


if __name__ == "__main__":
    main()
