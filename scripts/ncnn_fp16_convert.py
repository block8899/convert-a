#!/usr/bin/env python3
"""Convert ncnn .bin from fp32 to fp16.

Format (from ncnn Mat::write source):
  dims=1: flag(4) + dims(4) + w(4) + data[w * elemsize]
  dims=2: flag(4) + dims(4) + w(4) + h(4) + data[w*h * elemsize]
  dims=3: flag(4) + dims(4) + w(4) + h(4) + c(4) + data[align(w*h*es,16)*c]

Usage: python ncnn_fp16_convert.py <input_fp32.bin> <output_fp16.bin>
"""

import struct
import sys
import os
import numpy as np

FP32 = 0
FP16 = 0x01306B47
ALIGN = 16


def aligned(n):
    return (n + ALIGN - 1) & -ALIGN


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.bin>")
        sys.exit(1)

    inp, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(inp):
        print(f"MISSING: {inp}")
        sys.exit(1)

    buf = open(inp, 'rb').read()
    in_sz = len(buf)
    print(f"Input: {inp} ({in_sz / 1024 / 1024:.1f} MB)")

    pos = 0
    out = bytearray()
    count = 0
    converted = 0

    while pos < len(buf):
        flag, dims = struct.unpack_from('<Ii', buf, pos)
        pos += 8
        count += 1

        if dims == 0:
            out += struct.pack('<Ii', FP16, 0)
            continue

        w, = struct.unpack_from('<i', buf, pos)
        pos += 4
        h, c = 1, 1

        if dims >= 2:
            h, = struct.unpack_from('<i', buf, pos)
            pos += 4
        if dims >= 3:
            c, = struct.unpack_from('<i', buf, pos)
            pos += 4

        # How many float32 elements to read
        if dims == 1:
            n_elems = w
        elif dims == 2:
            n_elems = w * h
        else:
            cstep32 = aligned(w * h * 4) // 4
            n_elems = cstep32 * c

        raw = buf[pos:pos + n_elems * 4]
        pos += n_elems * 4

        if flag == FP32:
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            converted += 1

            if dims <= 2:
                fp16_data = arr.astype(np.float16).tobytes()
            else:
                wh = w * h
                cstep32 = aligned(wh * 4) // 4
                cstep16 = aligned(wh * 2) // 2

                parts = []
                for i in range(c):
                    ch = arr[i * cstep32: i * cstep32 + wh]
                    padded = np.zeros(cstep16, dtype=np.float16)
                    padded[:wh] = ch.astype(np.float16)
                    parts.append(padded.tobytes())
                fp16_data = b''.join(parts)

            out += struct.pack('<Ii', FP16, dims)
            out += struct.pack('<i', w)
            if dims >= 2:
                out += struct.pack('<i', h)
            if dims >= 3:
                out += struct.pack('<i', c)
            out += fp16_data
        else:
            # Not fp32 — copy through unchanged
            hdr = struct.pack('<Ii', flag, dims) + struct.pack('<i', w)
            if dims >= 2:
                hdr += struct.pack('<i', h)
            if dims >= 3:
                hdr += struct.pack('<i', c)
            out += hdr + raw

    with open(out_path, 'wb') as f:
        f.write(out)

    out_sz = len(out)
    pct = (1 - out_sz / in_sz) * 100
    print(f"Blobs: {count} total, {converted} converted")
    print(f"Output: {out_path} ({out_sz / 1024 / 1024:.1f} MB, -{pct:.0f}%)")


if __name__ == '__main__':
    main()
