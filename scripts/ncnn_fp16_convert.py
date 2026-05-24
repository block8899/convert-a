#!/usr/bin/env python3
"""Convert ncnn .bin weight file from fp32 to fp16.

ncnn .bin format (from Mat::read / Mat::write source):
  dims=0:  flag(4) + dims(4)
  dims=1:  flag(4) + dims(4) + w(4) + data[align(w*es, 16)]
  dims=2:  flag(4) + dims(4) + w(4) + h(4) + data[align(w*h*es, 16)]
  dims=3:  flag(4) + dims(4) + w(4) + h(4) + c(4) + data[align(w*h*es, 16) * c]

Usage: python ncnn_fp16_convert.py <input_fp32.bin> <output_fp16.bin>
"""

import struct
import sys
import os
import numpy as np

FP32_FLAG = 0
FP16_FLAG = 0x01306B47
MALLOC_ALIGN = 16


def align_up(n, a):
    return (n + a - 1) & -a


def elemsize_of(flag):
    """Decode elemsize from ncnn flag."""
    if flag == FP32_FLAG:
        return 4
    if flag == FP16_FLAG:
        return 2
    # packed format: elemsize in bits[8:15]
    return (flag >> 8) & 0xFF or 4


def data_bytes(w, h, c, es):
    """Total data bytes for an ncnn Mat blob.

    Matches ncnn source: align(w*h*elemsize, MALLOC_ALIGN) * c
    """
    return align_up(w * h * es, MALLOC_ALIGN) * c


def read_i32(f):
    d = f.read(4)
    if len(d) < 4:
        return None
    return struct.unpack('<i', d)[0]


def read_u32(f):
    d = f.read(4)
    if len(d) < 4:
        return None
    return struct.unpack('<I', d)[0]


def read_blob(f):
    """Read one Mat blob. Returns dict or None at EOF."""
    flag = read_u32(f)
    if flag is None:
        return None

    es = elemsize_of(flag)
    dims = read_i32(f)
    if dims is None:
        return None

    if dims == 0:
        return dict(flag=flag, dims=0, w=0, h=0, c=0, data=b'')

    w = read_i32(f)
    h, c = 1, 1
    if dims >= 2:
        h = read_i32(f)
    if dims >= 3:
        c = read_i32(f)

    db = data_bytes(w, h, c, es)
    data = f.read(db)

    if len(data) != db:
        print(f"  WARNING: truncated read at offset {f.tell() - len(data)}: "
              f"expected {db}, got {len(data)}")
        return None

    return dict(flag=flag, dims=dims, w=w, h=h, c=c, data=data)


def write_blob(f, b):
    """Write one Mat blob."""
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
    """Convert one fp32 blob to fp16."""
    if b['flag'] != FP32_FLAG:
        return b  # not fp32, leave as-is

    dims, w, h, c = b['dims'], b['w'], b['h'], b['c']
    if dims == 0 or len(b['data']) == 0:
        return dict(flag=FP16_FLAG, dims=0, w=0, h=0, c=0, data=b'')

    wh = w * h
    cstep32 = align_up(wh * 4, MALLOC_ALIGN) // 4   # elements per channel (fp32)
    cstep16 = align_up(wh * 2, MALLOC_ALIGN) // 2   # elements per channel (fp16)

    arr = np.frombuffer(b['data'], dtype=np.float32)

    parts = []
    for i in range(c):
        # Extract real values from aligned channel storage
        ch_f32 = arr[i * cstep32: i * cstep32 + wh]
        ch_f16 = ch_f32.astype(np.float16)
        # Pad to new alignment
        if cstep16 > wh:
            padded = np.zeros(cstep16, dtype=np.float16)
            padded[:wh] = ch_f16
            parts.append(padded.tobytes())
        else:
            parts.append(ch_f16.tobytes())

    return dict(flag=FP16_FLAG, dims=dims, w=w, h=h, c=c, data=b''.join(parts))


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.bin>")
        sys.exit(1)

    inp, out_path = sys.argv[1], sys.argv[2]
    in_sz = os.path.getsize(inp)
    print(f"Input: {inp} ({in_sz / 1024 / 1024:.1f} MB)")

    # Read all blobs
    blobs = []
    with open(inp, 'rb') as f:
        while True:
            b = read_blob(f)
            if b is None:
                break
            blobs.append(b)

    print(f"Blobs: {len(blobs)}")

    # Debug: print first few
    for i, b in enumerate(blobs[:5]):
        print(f"  [{i}] flag={b['flag']:#010x}  dims={b['dims']}  "
              f"w={b['w']} h={b['h']} c={b['c']}  data={len(b['data'])}B")
    if len(blobs) > 5:
        print(f"  ... ({len(blobs) - 5} more)")

    # Verify we consumed the entire file
    with open(inp, 'rb') as f:
        f.seek(0, 2)
        file_sz = f.tell()
    consumed = sum(len(b['data']) + {0: 8, 1: 12, 2: 16, 3: 20}[b['dims']]
                   for b in blobs)
    if consumed != file_sz:
        print(f"  WARNING: consumed {consumed} bytes but file is {file_sz} bytes")

    # Convert fp32 -> fp16
    new_blobs = []
    converted = 0
    for b in blobs:
        new_b = to_fp16(b)
        new_blobs.append(new_b)
        if new_b is not b:
            converted += 1

    # Write output
    with open(out_path, 'wb') as f:
        for b in new_blobs:
            write_blob(f, b)

    out_sz = os.path.getsize(out_path)
    pct = (1 - out_sz / in_sz) * 100 if in_sz > 0 else 0
    print(f"\nConverted: {converted}/{len(blobs)} blobs")
    print(f"Output: {out_path} ({out_sz / 1024 / 1024:.1f} MB, -{pct:.0f}%)")


if __name__ == '__main__':
    main()
