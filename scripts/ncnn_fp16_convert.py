#!/usr/bin/env python3
"""Convert ncnn .bin weight file from fp32 to fp16.

Usage: python ncnn_fp16_convert.py <input.bin> <output.bin>
The .param file does NOT need changes.
"""

import struct
import sys
import os
import numpy as np

FLAG_FP32 = 0
FLAG_FP16 = 0x01306B47
MALLOC_ALIGN = 16


def align(sz, n):
    return (sz + n - 1) & -n


def read_all_blobs(path):
    blobs = []
    with open(path, 'rb') as f:
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break

            flag = struct.unpack('<I', raw)[0]
            dims = struct.unpack('<i', f.read(4))[0]

            if dims == 0:
                blobs.append({'flag': flag, 'dims': 0, 'w': 0, 'h': 0,
                              'c': 0, 'csteps': [], 'data': b''})
                continue

            w = struct.unpack('<i', f.read(4))[0]
            h, c = 1, 1
            csteps = []

            if dims >= 2:
                h = struct.unpack('<i', f.read(4))[0]
            if dims >= 3:
                c = struct.unpack('<i', f.read(4))[0]
                cs_raw = f.read(c * 4)
                csteps = [struct.unpack_from('<i', cs_raw, j * 4)[0]
                          for j in range(c)]

            if flag == FLAG_FP32:
                elemsize = 4
            elif flag == FLAG_FP16:
                elemsize = 2
            else:
                elemsize = 4

            if dims == 1:
                count = w
            elif dims == 2:
                count = w * h
            else:
                count = csteps[0] * c if csteps else w * h * c

            data = f.read(count * elemsize)
            blobs.append({'flag': flag, 'dims': dims, 'w': w, 'h': h,
                          'c': c, 'csteps': csteps, 'data': data})
    return blobs


def write_blob(f, b):
    f.write(struct.pack('<I', b['flag']))
    f.write(struct.pack('<i', b['dims']))
    if b['dims'] == 0:
        return
    f.write(struct.pack('<i', b['w']))
    if b['dims'] >= 2:
        f.write(struct.pack('<i', b['h']))
    if b['dims'] >= 3:
        f.write(struct.pack('<i', b['c']))
        for cs in b['csteps']:
            f.write(struct.pack('<i', cs))
    f.write(b['data'])


def to_fp16(blob):
    if blob['flag'] != FLAG_FP32 or len(blob['data']) == 0:
        return blob, False

    dims = blob['dims']
    w, h, c, csteps = blob['w'], blob['h'], blob['c'], blob['csteps']
    arr = np.frombuffer(blob['data'], dtype=np.float32).copy()

    if dims <= 2:
        return {'flag': FLAG_FP16, 'dims': dims, 'w': w, 'h': h, 'c': c,
                'csteps': csteps,
                'data': arr.astype(np.float16).tobytes()}, True
    else:
        wh = w * h
        old_cs = csteps[0] if csteps else align(wh, MALLOC_ALIGN // 4)
        new_cs = align(wh, MALLOC_ALIGN // 2)

        parts = []
        for i in range(c):
            ch = arr[i * old_cs: i * old_cs + wh]
            ch16 = ch.astype(np.float16).tobytes()
            pad = (new_cs - wh) * 2
            parts.append(ch16 + b'\x00' * pad)

        return {'flag': FLAG_FP16, 'dims': dims, 'w': w, 'h': h, 'c': c,
                'csteps': [new_cs] * c, 'data': b''.join(parts)}, True


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.bin>")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    in_mb = os.path.getsize(inp) / 1024 / 1024
    print(f"Input: {inp} ({in_mb:.1f} MB)")

    blobs = read_all_blobs(inp)
    converted = 0
    result = []
    for b in blobs:
        new_b, ok = to_fp16(b)
        result.append(new_b)
        converted += ok

    with open(out, 'wb') as f:
        for b in result:
            write_blob(f, b)

    out_mb = os.path.getsize(out) / 1024 / 1024
    pct = (1 - out_mb / in_mb) * 100 if in_mb > 0 else 0
    print(f"Blobs: {len(blobs)} total, {converted} converted")
    print(f"Output: {out} ({out_mb:.1f} MB, -{pct:.0f}%)")


if __name__ == '__main__':
    main()
