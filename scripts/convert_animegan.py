# scripts/convert_animegan.py

import os
import sys
import subprocess


def main():
    print("=== AnimeGANv3 ONNX → NCNN ===\n")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Simplify
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   Failed: {ret.stderr[:200]}")
        sim_file = onnx_file
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. onnx2ncnn
    print("\n2. onnx2ncnn...")
    os.makedirs("output", exist_ok=True)
    raw_param = "animegan_raw.param"
    raw_bin = "animegan_raw.bin"

    ret = subprocess.run(
        ["onnx2ncnn", sim_file, raw_param, raw_bin],
        capture_output=True, text=True,
    )
    if ret.stderr:
        print(f"   warnings: {ret.stderr[:200]}")

    if not os.path.exists(raw_param):
        print("   FAILED")
        sys.exit(1)

    # 3. ncnnoptimize
    print("\n3. Optimizing...")
    ret = subprocess.run(
        ["ncnnoptimize",
         raw_param, raw_bin,
         "output/animegan.param", "output/animegan.bin",
         "65536"],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   ncnnoptimize failed: {ret.stderr}")
        import shutil
        shutil.copy(raw_param, "output/animegan.param")
        shutil.copy(raw_bin, "output/animegan.bin")

    # 4. Verify
    print("\n=== Output ===")
    for f in ["output/animegan.param", "output/animegan.bin"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
