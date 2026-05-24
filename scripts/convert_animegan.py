# scripts/convert_animegan.py

import os
import sys
import subprocess
import shutil


FP16_FLAG = "65539"


def convert_fp16(param_in, bin_in, param_out, bin_out):
    """Convert ncnn model fp32 → fp16 via ncnnoptimize."""
    print(f"\n   Converting to fp16...")
    ret = subprocess.run(
        ["ncnnoptimize", param_in, bin_in, param_out, bin_out, FP16_FLAG],
        capture_output=True, text=True, timeout=120,
    )
    if ret.returncode != 0:
        print(f"   ncnnoptimize stderr: {ret.stderr[:300]}")
        return False
    return True


def main():
    print("=== AnimeGANv3 ONNX → NCNN (fp32 + fp16) ===\n")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 1. Simplify ONNX
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

    fp32_size = os.path.getsize(pnnx_bin)
    print(f"   OK: param={os.path.getsize(pnnx_param) / 1024:.1f} KB, "
          f"bin={fp32_size / 1024 / 1024:.1f} MB")

    # 3. Copy fp32 to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/animegan.param")
    shutil.copy(pnnx_bin, "output/animegan.bin")
    print("\n3. fp32 copied to output/")

    # 4. Convert to fp16
    print("\n4. Generating fp16 version...")
    fp16_ok = convert_fp16(
        "output/animegan.param", "output/animegan.bin",
        "output/animegan_fp16.param", "output/animegan_fp16.bin",
    )

    # 5. Verify
    print("\n=== Output ===")
    files = [
        ("output/animegan.param", "fp32 param"),
        ("output/animegan.bin", "fp32 bin"),
        ("output/animegan_fp16.param", "fp16 param"),
        ("output/animegan_fp16.bin", "fp16 bin"),
    ]
    for path, label in files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            unit = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.1f} KB"
            print(f"  {label}: {unit}")
        else:
            if "fp16" in label:
                print(f"  {label}: MISSING (fp16 conversion failed)")
            else:
                print(f"  {label}: MISSING")
                sys.exit(1)

    if fp16_ok:
        fp16_size = os.path.getsize("output/animegan_fp16.bin")
        saving = (1 - fp16_size / fp32_size) * 100
        print(f"\n  fp16 is {saving:.0f}% smaller than fp32")

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
