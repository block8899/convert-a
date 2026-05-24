# scripts/convert_animegan.py

import os
import sys
import subprocess
import shutil


def main():
    print("=== AnimeGANv3 ONNX → NCNN (via PNNX) ===\n")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Simplify ONNX (optional, improves conversion quality)
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   Simplify failed, using original: {ret.stderr[:200]}")
        shutil.copy(onnx_file, sim_file)
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. Convert ONNX → NCNN via PNNX
    print("\n2. Converting via PNNX...")
    pnnx_param = "animegan_sim.ncnn.param"
    pnnx_bin = "animegan_sim.ncnn.bin"

    # Clean previous PNNX output if any
    for f in [pnnx_param, pnnx_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", sim_file],
        capture_output=True, text=True, timeout=300,
    )
    print(f"   stdout (last 500): {ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"   stderr (last 500): {ret.stderr[-500:]}")
        sys.exit(1)

    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("   PNNX output files not found!")
        sys.exit(1)

    print(f"   OK: param={os.path.getsize(pnnx_param) / 1024:.1f} KB, "
          f"bin={os.path.getsize(pnnx_bin) / 1024 / 1024:.1f} MB")

    # 3. Copy to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/animegan.param")
    shutil.copy(pnnx_bin, "output/animegan.bin")

    # 4. Verify
    print("\n=== Output ===")
    for f in ["output/animegan.param", "output/animegan.bin"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  {f}: {size / 1024 / 1024:.1f} MB"
                  if size > 1024 * 1024
                  else f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
