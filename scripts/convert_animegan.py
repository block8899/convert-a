import os
import sys
import subprocess
import shutil


def main():
    print("=== AnimeGANv3 ONNX -> NCNN (fp32 + fp16) ===\n")

    onnx_file = "AnimeGANv3_PortraitSketch_25.onnx"
    if not os.path.exists(onnx_file):
        print(f"MISSING: {onnx_file}")
        sys.exit(1)

    print(f"Input: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    os.makedirs("output", exist_ok=True)

    # 1. Simplify ONNX
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   Failed, using original: {ret.stderr[:200]}")
        shutil.copy(onnx_file, sim_file)
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. Convert ONNX -> NCNN (fp32) via PNNX
    print("\n2. Converting fp32 via PNNX...")
    fp32_param = "animegan_sim.ncnn.param"
    fp32_bin = "animegan_sim.ncnn.bin"
    for f in [fp32_param, fp32_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", sim_file],
        capture_output=True, text=True, timeout=300,
    )
    if ret.returncode != 0:
        print(f"   stderr: {ret.stderr[-300:]}")
        sys.exit(1)

    if not os.path.exists(fp32_param) or not os.path.exists(fp32_bin):
        print("   PNNX output not found!")
        sys.exit(1)

    shutil.copy(fp32_param, "output/animegan.param")
    shutil.copy(fp32_bin, "output/animegan.bin")
    fp32_sz = os.path.getsize(fp32_bin)
    print(f"   OK: {fp32_sz / 1024 / 1024:.1f} MB")

    # 3. Convert ONNX weights to fp16
    print("\n3. Converting ONNX weights to fp16...")
    fp16_onnx = "animegan_fp16.onnx"
    ret = subprocess.run(
        [sys.executable, "scripts/convert_onnx_fp16.py", sim_file, fp16_onnx],
        capture_output=True, text=True, timeout=120,
    )
    print(f"   {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"   FAILED: {ret.stderr[-300:]}")
        sys.exit(1)

    # 4. Convert fp16 ONNX -> NCNN via PNNX
    print("\n4. Converting fp16 ONNX -> NCNN via PNNX...")
    fp16_param = "animegan_fp16.ncnn.param"
    fp16_bin = "animegan_fp16.ncnn.bin"
    for f in [fp16_param, fp16_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", fp16_onnx],
        capture_output=True, text=True, timeout=300,
    )
    if ret.returncode != 0:
        print(f"   stderr: {ret.stderr[-300:]}")

    if not os.path.exists(fp16_param) or not os.path.exists(fp16_bin):
        print("   PNNX fp16 output not found!")
        sys.exit(1)

    shutil.copy(fp16_param, "output/animegan_fp16.param")
    shutil.copy(fp16_bin, "output/animegan_fp16.bin")
    fp16_sz = os.path.getsize(fp16_bin)
    pct = (1 - fp16_sz / fp32_sz) * 100 if fp32_sz > 0 else 0
    print(f"   OK: {fp16_sz / 1024 / 1024:.1f} MB (-{pct:.0f}%)")

    # 5. Verify
    print("\n=== Output ===")
    for f in ["output/animegan.param", "output/animegan.bin",
              "output/animegan_fp16.param", "output/animegan_fp16.bin"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            unit = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024*1024 else f"{sz / 1024:.1f} KB"
            print(f"  {f}: {unit}")
        else:
            print(f"  {f}: MISSING")

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
