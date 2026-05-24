# scripts/convert_animegan.py

import os
import sys
import shutil
import subprocess


def main():
    print("=== AnimeGANv3 ONNX → NCNN ===\n")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Simplify
    print("\nSimplifying...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  onnxsim failed, using original: {ret.stderr}")
        sim_file = onnx_file
    else:
        print(f"  Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. onnx2ncnn
    print("Converting to NCNN...")
    os.makedirs("output", exist_ok=True)
    param_f = "output/animegan.param"
    bin_f = "output/animegan.bin"

    ret = subprocess.run(
        ["onnx2ncnn", sim_file, param_f, bin_f],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  onnx2ncnn warning: {ret.stderr}")

    for f in [param_f, bin_f]:
        if os.path.exists(f):
            print(f"  {f}: {os.path.getsize(f) / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
