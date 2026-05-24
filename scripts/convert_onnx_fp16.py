#!/usr/bin/env python3
"""Convert ONNX model to fp16 using onnxconverter-common.

Handles type compatibility properly (inserts Cast nodes where needed).

Usage: python convert_onnx_fp16.py <input.onnx> <output.onnx>
"""

import sys
import os
import onnx
from onnxconverter_common import float16


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output.onnx>")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    if not os.path.exists(inp):
        print(f"MISSING: {inp}")
        sys.exit(1)

    in_sz = os.path.getsize(inp)
    print(f"Input: {inp} ({in_sz / 1024 / 1024:.1f} MB)")

    model = onnx.load(inp)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save(model_fp16, out)

    out_sz = os.path.getsize(out)
    pct = (1 - out_sz / in_sz) * 100 if in_sz > 0 else 0
    print(f"Output: {out} ({out_sz / 1024 / 1024:.1f} MB, -{pct:.0f}%)")


if __name__ == "__main__":
    main()
