#!/usr/bin/env python3
"""Convert ncnn .bin weight file from fp32 to fp16.

ncnn .bin format per blob:
  dims=0:  flag(4) + dims(4)
  dims=1:  flag(4) + dims(4) + w(4) + data[w * elemsize]
  dims=2:  flag(4) + dims(4) + w(4) + h(4) + data[w*h * elemsize]
  dims=3:  flag(4) + dims(4) + w(4) + h(4) + c(4) + data[cstep*c * elemsize]
           cstep = align(w*h*elemsize, 16) / elemsize  (computed, NOT in file)

Usage: python ncnn_fp16_convert.py <input_fp32.bin> <output_fp16.bin>
"""

import struct
import sys
import os
import numpy as np

FLAG_FP32 = 0
FLAG_FP16 = 0x01306B47
MALLOC_ALIGN = 16


def align_up(n, a):
    return (n + a - 1) & -a


def elemsize_for(flag):
    if flag == FLAG_FP16:
        return 2
    return 4  # fp32 or unknown


def read_i32(f):
    raw = f.read(4)
    if len(raw) < 4:
        return None
    return struct.unpack('<i', raw)[0]


def read_u32(f):
    raw = f.read(4)
    if len(raw) < 4:
        return None
    return struct.unpack('<I', raw)[0]


def read_blob(f):
    """Read one blob from ncnn .bin. Returns dict or None at EOF."""
    flag = read_u32(f)
    if flag is None:
        return None

    dims = read_i32(f)
    if dims is None:
        return None

    if dims == 0:
        return dict(flag=flag, dims=0, w=0, h=0, c=0, data=b'')

    w = read_i32(f)
    if w is None:
        return None

    es = elemsize_for(flag)

    if dims == 1:
        data = f.read(w * es)
        return dict(flag=flag, dims=1, w=w, h=1, c=1, data=data)

    h = read_i32(f)
    if h is None:
        return None

    if dims == 2:
        data = f.read(w * h * es)
        return dict(flag=flag, dims=2, w=w, h=h, c=1, data=data)

    # dims == 3
    c = read_i32(f)
    if c is None:
        return None

    cstep = align_up(w * h * es, MALLOC_ALIGN) // es
    data = f.read(cstep * c * es)
    return dict(flag=flag, dims=3, w=w, h=h, c=c, data=data)


def write_blob(f, b):
    """Write one blob to ncnn .bin."""
    f.write(struct.pack('<I', b['flag']))
    f.write(struct.pack('<i', b['dims']))
    if b['dims'] == 0:
        return
    f.write(struct.pack('<i', b['w']))
    if b['dims'] >= 2:
        f.write(struct.pack('<i', b['h']))
    if b['dims'] >= 3:
        f.write(struct.pack('<i', b['c']))
    f.write(b['data'])


def to_fp16(b):
    """Convert one fp32 blob to fp16. Returns new dict."""
    if b['flag'] != FLAG_FP32 or len(b['data']) == 0:
        return b

    dims, w, h, c = b['dims'], b['w'], b['h'], b['c']

    if dims == 0:
        return dict(flag=FLAG_FP16, dims=0, w=0, h=0, c=0, data=b'')

    # 1D or 2D: no padding in file, straight conversion
    if dims <= 2:
        arr = np.frombuffer(b['data'], dtype=np.float32).copy()
        return dict(flag=FLAG_FP16, dims=dims, w=w, h=h, c=c,
                    data=arr.astype(np.float16).tobytes())

    # 3D: channel-padded layout
    wh = w * h
    cstep32 = align_up(wh * 4, MALLOC_ALIGN) // 4   # align(wh, 4)
    cstep16 = align_up(wh * 2, MALLOC_ALIGN) // 2   # align(wh, 8)

    arr = np.frombuffer(b['data'], dtype=np.float32).copy()

    parts = []
    for i in range(c):
        # extract real data from padded channel
        ch = arr[i * cstep32: i * cstep32 + wh]
        ch16 = ch.astype(np.float16).tobytes()
        pad = b'\x00' * ((cstep16 - wh) * 2)
        parts.append(ch16 + pad)

    return dict(flag=FLAG_FP16, dims=3, w=w, h=h, c=c, data=b''.join(parts))


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.bin>")
        sys.exit(1)

    inp, out = sys.argv[1], sys.argv[2]
    in_sz = os.path.getsize(inp)
    print(f"Input: {inp} ({in_sz / 1024 / 1024:.1f} MB)")

    # read
    blobs = []
    with open(inp, 'rb') as f:
        while True:
            b = read_blob(f)
            if b is None:
                break
            blobs.append(b)

    print(f"Blobs: {len(blobs)}")

    # dump first few for debug
    for i, b in enumerate(blobs[:3]):
        print(f"  [{i}] flag={b['flag']:#010x}  dims={b['dims']}  "
              f"w={b['w']} h={b['h']} c={b['c']}  data={len(b['data'])}B")

    # convert
    new = []
    converted = 0
    for b in blobs:
        if b['flag'] == FLAG_FP32:
            new.append(to_fp16(b))
            converted += 1
        else:
            new.append(b)

    # write
    with open(out, 'wb') as f:
        for b in new:
            write_blob(f, b)

    out_sz = os.path.getsize(out)
    pct = (1 - out_sz / in_sz) * 100 if in_sz > 0 else 0
    print(f"Converted: {converted}/{len(blobs)} blobs")
    print(f"Output: {out} ({out_sz / 1024 / 1024:.1f} MB, -{pct:.0f}%)")


if __name__ == '__main__':
    main()
