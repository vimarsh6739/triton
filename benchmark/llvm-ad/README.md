# AdamW before and after Triton lowers the loads

At TTIR, Enzyme differentiates the masked loads in the AdamW step. The resulting JVP passes the FP64 PyTorch reference on the GPU. In the emitted LLVM IR the same loads have already become predicated inline PTX. On the LLVMEnzyme build below, type analysis stops at the first such load. We can isolate this by swapping only the load in a masked square kernel.

| Input to LLVM Enzyme | Direct JVP | LLVM O2 then JVP |
| --- | --- | --- |
| Masked square, branch and typed LLVM load | Generated | Generated |
| Same square, predicated PTX load from AdamW | Assertion in type analysis | Same assertion |
| Full lowered AdamW | Same assertion | Same assertion |

All three inputs pass LLVM verification first. The positive control generates a tangent store. The first diagnostic on both failing cases starts with `Error Unknown Value: ptr asm sideeffect` and names the `ld.global.b32` instruction. This compares operation coverage in these two builds. Other lowerings and custom LLVM AD rules could preserve or recover the load semantics.

From the Triton worktree, run this with matching LLVM and Enzyme builds. Each case leaves its input, output or diagnostic in the selected directory.

```sh
python benchmark/llvm-ad/probe.py --opt /path/to/llvm/bin/opt \
    --plugin /path/to/LLVMEnzyme-24.so --output /tmp/adamw-llvm-ad
```

`adamw.ll` comes from the FP32 primal, 1,023 elements, block 256 and four warps. Debug metadata was stripped. To regenerate the module, run the GPU benchmark for that configuration with `--modes primal --ir-dir /tmp/adamw-ir` and give the resulting `.ll` file to the probe with `--primal`. The probe appends the same public Enzyme call convention to both controls. For AdamW it marks all seven state pointers duplicated and the size and two backend scratch pointers constant.

The test ran on 2026-11-26 with the existing LLVM 24.0.0git build with assertions. Binary SHA-256 values were `eb96c90316639a71e48a4c44ddcb2ac80fc6b59d30f57f291c3976165244a214` for opt and `73c906a48be175ee944196b83771e7ac8d6cadbe3c471827a38b212eb9070aca` for LLVMEnzyme-24.so.
