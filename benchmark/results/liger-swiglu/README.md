# Reverse AD in Liger's SwiGLU MLP

Liger maintains Triton forward and backward kernels for both the row and tiled forms of SwiGLU. This experiment imports the upstream forward unchanged and uses its TTIR to compute the two activation gradients in a PyTorch backward. The upstream module still owns all three linear projections. [Pinned kernel](https://github.com/linkedin/Liger-Kernel/blob/cc29eda82744aece335c1ca4bd3ace5c9d768048/src/liger_kernel/ops/swiglu.py) · [Upstream MLP](https://github.com/linkedin/Liger-Kernel/blob/cc29eda82744aece335c1ca4bd3ace5c9d768048/src/liger_kernel/transformers/swiglu.py)

On a dense input of shape (2, 17, 128) the original and adapted `LigerSwiGLUMLP` produced identical outputs. The input gradient and gate, up and down projection weight gradients also passed against both the original module and an independent FP64 PyTorch reference. The largest generated-to-Liger relative L2 error across those gradients was 1.11e-7 for FP32 and 0.00421 for BF16. The scope here is eager, dense, first-order training with the default down multiplier. [MLP results](mlp.csv)

The activation test compared 13 shape/dtype cases, including three shapes from Liger's own test suite and an odd-width masked tail with gate multiplier 0.625. Each passed the upstream comparison and an independent FP64 autograd check. At (3,072, 12,288) in BF16, the generated wrapper took **813.06 µs** for forward and backward versus **391.14 µs** upstream. The shape comes from a Qwen3 training report. These timings use an RTX 5090 and CUDA graphs of eager PyTorch. The upstream issue measured compiled execution on B300. [Liger tests](https://github.com/linkedin/Liger-Kernel/blob/cc29eda82744aece335c1ca4bd3ace5c9d768048/test/ops/test_swiglu.py) · [Shape provenance](https://github.com/linkedin/Liger-Kernel/issues/1330) · [Summary](summary.csv) · [Raw replays](timings.csv)

Both versions launch the original forward. Liger's handwritten backward reuses the saved input storage for its gradients. The generated backward zeros two gradient buffers, copies the incoming seed because AD consumes it, and runs a combined primal and VJP into separate output storage. This removes handwritten derivative math but costs memory traffic. Timings include these differences. Each replay starts with restored inputs outside the timed region. The harness warms both graphs, alternates their order at random and reports the median of 41 CUDA-event samples. It checks both replay outputs after timing.

## Reproduce

Use the reverse frontend at `46f4934ff8` and the reverse-enabled Enzyme-JAX extension with the seed-read fix. The recorded run used Triton 3.9.0, PyTorch 2.12.0a0+git0d62256, CUDA 12.9 and extension SHA-256 `72451477afb2a5f4cda29409d3e17fa9eb667d62c401fb640f23e9d3cbc2e852`. From this Triton worktree with that environment active:

```sh
git clone https://github.com/linkedin/Liger-Kernel.git ../Liger-Kernel
git -C ../Liger-Kernel checkout --detach cc29eda82744aece335c1ca4bd3ace5c9d768048
export PYTHONPATH="$(realpath ../Liger-Kernel/src):$PYTHONPATH"
python benchmark/liger_swiglu.py --mlp
python benchmark/liger_swiglu.py --samples /tmp/swiglu-timings.csv
```
