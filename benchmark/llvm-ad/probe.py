"""Apply an Enzyme LLVM JVP request to the AdamW kernel and two load controls."""

import argparse
import resource
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    root = Path(__file__).parent
    parser.add_argument("--primal", type=Path, default=root / "adamw.ll",
                        help="use the .ll emitted by bench_adamw.py --ir-dir")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep assertion diagnostics in stderr without writing a large core file.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name, primal, request in [
        ("typed-load", root / "typed-load.ll", "load-jvp.ll"),
        ("ptx-load", root / "ptx-load.ll", "load-jvp.ll"),
        ("adamw", args.primal, "adamw-jvp.ll"),
    ]:
        source = args.output / f"{name}-input.ll"
        source.write_text(primal.read_text() + "\n" + (root / request).read_text())
        cmd = [str(args.opt), "-mtriple=nvptx64-nvidia-cuda", str(source)]
        subprocess.run([*cmd, "-passes=verify", "-disable-output"], check=True)
        for optimize, pipeline in [("direct", "enzyme"), ("O2", "default<O2>,enzyme")]:
            output = args.output / f"{name}-{optimize}.ll"
            result = subprocess.run([*cmd, f"-load-pass-plugin={args.plugin}", f"-passes={pipeline}", "-S",
                                     "-o", str(output)], capture_output=True, text=True)
            (args.output / f"{name}-{optimize}.stderr").write_text(result.stderr)
            if result.returncode:
                first = result.stderr.splitlines()[0] if result.stderr else "no diagnostic"
                print(f"{name} {optimize}: exit {result.returncode} {first}")
            else:
                print(f"{name} {optimize}: generated {output}")
                if "call void (...) @__enzyme_fwddiff(" in output.read_text():
                    raise RuntimeError(f"Enzyme did not consume the request in {output}")
            if name == "typed-load":
                result.check_returncode()


if __name__ == "__main__":
    main()
