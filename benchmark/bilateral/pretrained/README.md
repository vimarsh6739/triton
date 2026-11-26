# Run the authors’ Local Laplacian model

The published HDRNet checkpoint contains the evaluation graph for predicting its bilateral grid and guidance image. We run those two TensorFlow branches on the photograph supplied with HDRNet and pass the tensors to each implementation of the slice. The network and image decoding run once, outside the timing window. [HDRNet project and model archive](https://groups.csail.mit.edu/graphics/hdrnet/) · [Authors’ inference code](https://github.com/google/hdrnet/blob/e9a5bf59a232ee7562d02a6cf4dc5cef6191b6c6/hdrnet/bin/run.py#L159-L181)

The extractor reads the `local_laplacian/normal_1024` weights, substitutes two images for the saved TensorFlow training queue and fetches the checkpoint’s named evaluation outputs. The published input is 2,048 × 2,048 RGB. The 1,024 × 1,024 comparison uses a box downsample of the same image. For each size, the coefficient branch takes a nearest-neighbor resize to 256 × 256 and the pointwise guide branch takes the full image, both normalized to [0, 1]. The graph had constant-folded the full-resolution reshape. Using the input’s actual shape lets us evaluate the guide in 128-row strips. On the original 2,048 image the resulting guide was bitwise identical to the original fixed-shape, whole-frame evaluation.

Run extraction in a separate CPU environment. Our recorded configuration used Python 3.12, tensorflow-cpu 2.19.1, NumPy 2.1.3 and Pillow 13.0.0. The script verifies SHA-256 for the image and the three checkpoint components it needs. Run from this directory.

```sh
curl -fL https://data.csail.mit.edu/graphics/hdrnet/pretrained_models.zip -o /tmp/hdrnet-models.zip
unzip /tmp/hdrnet-models.zip 'pretrained_models/local_laplacian/normal_1024/*' -d /tmp/hdrnet-checkpoint
curl -fL https://raw.githubusercontent.com/google/hdrnet/e9a5bf59a232ee7562d02a6cf4dc5cef6191b6c6/sample_data/input.png -o /tmp/hdrnet-input.png
python3.12 -m venv /tmp/hdrnet-tf
/tmp/hdrnet-tf/bin/pip install tensorflow-cpu==2.19.1 numpy==2.1.3 pillow==13.0.0
TF_ENABLE_ONEDNN_OPTS=0 /tmp/hdrnet-tf/bin/python extract.py /tmp/hdrnet-checkpoint/pretrained_models/local_laplacian/normal_1024 /tmp/hdrnet-input.png /tmp/hdrnet-trained
```

Each NPZ contains a FP32 `grid` of shape `(1,16,16,8,3,4)` and a FP32 `guide` of shape `(1,H,W)`. The two runs produced the following file hashes.

| File | SHA-256 |
| --- | --- |
| 1024.npz | `c47839fe95a06601cba5b79b1f1848675893a4a5368cf76bdea92d3bd06af75b` |
| 2048.npz | `171855d3284f7921b86b1b770ca5a227c6a43c8291c8b1535e347baf48222f40` |

## Three exact guide bin centers

Three pixels of the 2,048 × 2,048 learned guide are exactly 0.3125, a depth bin center. The neighboring bin’s smoothed distance is √(1 + 10⁻⁸). In FP32 this rounds to one, so an inclusive clamp passes a gradient for that zero-weight neighbor. In FP64 its distance is greater than one, giving zero gradient. A strict positive-part derivative returns zero at all three pixels on both precisions. The complete precision comparison uses every pixel. [Per-pixel values](knots.csv)

With the PyTorch benchmark environment active, run the direct eight-corner probe for the archived output cotangent.

```sh
python knots.py /tmp/hdrnet-input.png /tmp/hdrnet-trained/2048.npz --seed 51
```
