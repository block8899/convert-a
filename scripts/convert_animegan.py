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

    print(f"Input: {onnx_file} ({os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB)")

    # 1. Simplify
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   onnxsim failed: {ret.stderr}")
        print("   Using original ONNX")
        sim_file = onnx_file
    else:
        print(f"   Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. onnx2ncnn
    print("\n2. Converting to NCNN...")
    os.makedirs("output", exist_ok=True)

    param_file = "output/animegan.param"
    bin_file = "output/animegan.bin"

    onnx2ncnn = shutil.which("onnx2ncnn") or "onnx2ncnn"

    ret = subprocess.run(
        [onnx2ncnn, sim_file, param_file, bin_file],
        capture_output=True, text=True,
    )

    if ret.returncode != 0:
        print(f"   onnx2ncnn error: {ret.stderr}")
        if ret.stdout:
            print(f"   stdout: {ret.stdout}")
        sys.exit(1)

    # 3. Verify
    print("\nOutput:")
    for f in [param_file, bin_file]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nAnimeGANv3 conversion OK!")


if __name__ == "__main__":
    main()
