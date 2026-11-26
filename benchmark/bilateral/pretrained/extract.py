"""Run the authors' pretrained HDRNet coefficient and guidance branches on the sample photograph."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


CHECKPOINT = "upgraded_from_sig2017.ckpt-0"
HASHES = {
    ".meta": "ed95ebd51bddb023ee91aba8d62e110a7a0d3aa0991c33e3155454e0c9b3b418",
    ".index": "d8b4e1b3e168ee6a50f645012e16e1c18672783e020b14b811c22e795da603d1",
    ".data-00000-of-00001": "b56188ed0185e4752261525505633e895e91141bfc38edd2b8e4dc744e72d898",
}
IMAGE_HASH = "7c9159b59423080504fdc7c4a061e20a240abcee0f4645348d11c88a4c678375"


def verify(path, digest):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError(f"Unexpected content at {path}")


def inference(checkpoint):
    prefix = str(checkpoint / CHECKPOINT)
    for suffix, digest in HASHES.items():
        verify(Path(prefix + suffix), digest)
    meta = tf.compat.v1.MetaGraphDef.FromString(Path(prefix + ".meta").read_bytes())
    targets = [next(s for s in meta.collection_def[name].node_list.value if s.startswith("eval/"))
               for name in ("bilateral_coefficients", "guide")]
    # The saved eval graph consumes the training queue. Substitute its two image
    # inputs and the image shape TF1 constant-folded when the graph was saved.
    graph = tf.Graph()
    with graph.as_default():
        full = tf.compat.v1.placeholder(tf.float32, (1, None, None, 3), name="full")
        tf.compat.v1.placeholder(tf.float32, (1, 256, 256, 3), name="low")
        tf.shape(full, name="image_shape")
    source = meta.graph_def
    nodes = {n.name: n for n in source.node}
    assert np.array_equal(tf.make_ndarray(nodes["eval/inference/guide/ccm/Shape"].attr["value"].tensor),
                          [1, 2048, 2048, 3])
    changes = {"train_data/shuffle_batch": "full", "train_data/shuffle_batch:2": "low",
               "eval/inference/guide/ccm/Shape": "image_shape"}
    source.node.extend(graph.as_graph_def().node)
    for node in source.node:
        for i, edge in enumerate(node.input):
            node.input[i] = changes.get(edge, edge)
    model = tf.compat.v1.graph_util.extract_sub_graph(source, [s.split(":")[0] for s in targets])
    reader = tf.train.load_checkpoint(prefix)
    for node in model.node:
        node.device = ""
        if "_output_shapes" in node.attr:
            del node.attr["_output_shapes"]
        if node.op == "VariableV2":
            values = reader.get_tensor(node.name)
            node.op = "Const"
            node.attr.clear()
            node.attr["dtype"].type = tf.float32.as_datatype_enum
            node.attr["value"].tensor.CopyFrom(tf.make_tensor_proto(values))
    executable = tf.Graph()
    with executable.as_default():
        outputs = tf.import_graph_def(model, name="", return_elements=targets)
    return executable, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048])
    args = parser.parse_args()
    verify(args.image, IMAGE_HASH)
    graph, (grid_op, guide_op) = inference(args.checkpoint_dir)
    args.output.mkdir(exist_ok=True, parents=True)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1)
    with Image.open(args.image) as source, tf.compat.v1.Session(graph=graph, config=config) as sess:
        for size in args.sizes:
            photo = source.convert("RGB").resize((size, size), Image.Resampling.BOX)
            # Original run.py uses order-zero resampling of the normalized RGB input.
            low = np.asarray(photo.resize((256, 256), Image.Resampling.NEAREST)).astype("f") / 255
            high = np.asarray(photo).astype("f") / 255
            grid = sess.run(grid_op, {"low:0": low[None]})
            # The guide is pointwise. Strips bound memory for its 16-segment tone curves.
            strips = [sess.run(guide_op, {"full:0": high[None, y:y+128]}) for y in range(0, size, 128)]
            guide = np.concatenate(strips, axis=1)
            if grid.shape != (1, 16, 16, 8, 3, 4) or guide.shape != (1, size, size):
                raise ValueError(f"Unexpected tensor shapes {grid.shape}, {guide.shape}")
            if not (np.isfinite(grid).all() and np.isfinite(guide).all()):
                raise ValueError("Nonfinite learned inputs")
            path = args.output / f"{size}.npz"
            np.savez(path, grid=grid, guide=guide)
            print(path, grid.min(), grid.max(), guide.min(), guide.max())


if __name__ == "__main__":
    main()
