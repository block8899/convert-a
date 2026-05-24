#!/usr/bin/env python3
"""Convert ONNX model weights from fp32 to fp16.

Usage: python convert_onnx_fp16.py <input.onnx> <output.onnx>
"""

import sys
import os
import onnx
from onnx import TensorProto, numpy_helper
import numpy as np


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output.onnx>")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]

    if not os.path.exists(inp):
        print(f"MISSING: {inp}")
        sys.exit(1)

    print(f"Input: {inp} ({os.path.getsize(inp) / 1024 / 1024:.1f} MB)")

    model = onnx.load(inp)

    converted = 0
    skipped = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            arr16 = arr.astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr16, init.name))
            converted += 1
        else:
            skipped += 1

    onnx.save(model, out)

    in_sz = os.path.getsize(inp)
    out_sz = os.path.getsize(out)
    pct = (1 - out_sz / in_sz) * 100 if in_sz > 0 else 0
    print(f"Converted: {converted} tensors to fp16, {skipped} skipped")
    print(f"Output: {out} ({out_sz / 1024 / 1024:.1f} MB, -{pct:.0f}%)")


if __name__ == "__main__":
    main()
