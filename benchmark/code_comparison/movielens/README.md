# MovieLens rating loss and three gradients

A rank-16 recommender predicts a rating by multiplying the user, movie and week factor rows elementwise, then summing the 16 products. Training uses half the squared prediction error. The generated and handwritten kernels both return a loss for each rating, accumulate gradients into all three tables and consume the output seed. The handwritten kernel calls the same loss and asks it to return the addresses, factors and residual for its three atomic adds. [Shared primal](shared.py) · [Generated pullback](generated.py) · [Handwritten pullback](handwritten.py)

Counting all nonblank source lines in the shared module and one method module gives **29 lines for generated and 39 for handwritten**. Both count the shared 23 lines once, including the optional intermediate values used by the handwritten derivative. The six and sixteen method-specific lines include imports, decorators, signatures, activity declarations and calls. Counting each as a single file, with the shared-module import removed and each external import counted once, gives 28 and 36 lines. Both use Triton’s standard grid launch, 16 observations per program, one warp and FP32 arithmetic. The generated method uses the public combined reverse launcher. [Benchmark harness](bench.py)

## Which ratings are timed?

We take the 80,000 actual ratings in MovieLens 100K’s first training split, subtract its mean rating, and map timestamps to seven-day bins. For each of seeds 17 and 43 we initialize the three tables as in our training run and shuffle the training split once. The benchmark uses the first complete batch of 2,048 ratings and the final batch of 128 ratings. Within each shape and seed both methods receive the same indices, ratings, model state, output buffers and launch configuration. The timing repeats these fixed batches with warmed addresses. The published training experiment updates the factors over eight epochs. [Training code](https://github.com/vimarsh6739/triton/blob/0bb1c3eefd540412180ae6394cfa1ae45a906ed9/benchmark/sparse_cp_movielens.py) · [Training results](https://github.com/vimarsh6739/triton/blob/0bb1c3eefd540412180ae6394cfa1ae45a906ed9/benchmark/results/movielens-u1/README.md)

| RTX 5090, µs | Ratings | Handwritten | Generated |
| --- | ---: | ---: | ---: |
| Batch | 2,048 | 14.830–15.020 | 14.771–15.001 |
| Batch | 128 | 8.608–8.670 | 8.605–8.611 |
| Kernel | 2,048 | 2.048–2.112 | 1.984–2.048 |
| Kernel | 128 | 1.280 | 1.280 |

Each range spans two run medians. For each median we shuffled the order of the two methods over 51 rounds, after five warmup rounds. One sample times 32 calls in a CUDA graph and reports the GPU time per call. The batch scope includes four device gathers (user, movie, week and rating), zeroing the three gradient tables, filling the seed with 1/batch size and computing the losses and gradients. The kernel scope captures the fused loss and gradient launches on 32 independently initialized sets of buffers. Graph events bracket those launches after preparation. Compilation, Python dispatch and the AdamW update fall outside both scopes. [Batch samples](timings.csv) · [Kernel samples](kernel-timings.csv)

Before timing, both kernels pass elementwise comparisons for the losses and all three factor gradients against FP64 PyTorch autograd. One test uses the training mean seed and zero initial gradients. A second uses a nonuniform signed seed and nonzero initial gradients to check accumulation. Both verify that the generated and handwritten kernels clear the caller’s seed. We verify the training outputs again after replay, including every independent buffer set in the kernel measurement. [Batch checks](checks.csv) · [Kernel checks](kernel-checks.csv)

## Reproduce

Use the reverse-enabled Triton frontend at 03bead828c, a compatible CUDA PyTorch, NumPy and the Enzyme-JAX binding with the ordinary-store synchronization fix. Each output directory is created by the runner. [GPU and compiler versions](environment.json) · [Kernel environment](kernel-environment.json)

```sh
curl -fLsS https://files.grouplens.org/datasets/movielens/ml-100k.zip -o /tmp/ml-100k.zip
python bench.py /tmp/ml-100k.zip /tmp/movielens-batch --scope batch
python bench.py /tmp/ml-100k.zip /tmp/movielens-kernel --scope kernel
```

The runner verifies the official archive’s SHA-256 and loads only u1.base. GroupLens Research at the University of Minnesota collected the ratings. Cite F. Maxwell Harper and Joseph A. Konstan, *The MovieLens Datasets: History and Context* (2015). [Dataset terms and split description](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) · [Dataset paper](https://doi.org/10.1145/2827872)
