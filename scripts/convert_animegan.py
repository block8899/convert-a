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

    # 2. Convert ONNX -> NCNN via PNNX
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
    print(f"   stdout (last 300): {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"   stderr: {ret.stderr[-300:]}")
        sys.exit(1)

    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("   PNNX output not found!")
        sys.exit(1)

    fp32_size = os.path.getsize(pnnx_bin)
    print(f"   OK: bin={fp32_size / 1024 / 1024:.1f} MB")

    # 3. Copy fp32 to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/animegan.param")
    shutil.copy(pnnx_bin, "output/animegan.bin")
    print("\n3. fp32 copied to output/")

    # 4. Convert to fp16
    print("\n4. Generating fp16 version...")
    ret = subprocess.run(
        [sys.executable, "scripts/ncnn_fp16_convert.py",
         "output/animegan.bin", "output/animegan_fp16.bin"],
        capture_output=True, text=True, timeout=120,
    )
    print(f"   {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"   fp16 failed:\n{ret.stderr[-500:]}")
    else:
        shutil.copy("output/animegan.param", "output/animegan_fp16.param")

    # 5. Verify
    print("\n=== Output ===")
    for f in ["output/animegan.param", "output/animegan.bin",
              "output/animegan_fp16.param", "output/animegan_fp16.bin"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            print(f"  {f}: {sz / 1024 / 1024:.1f} MB" if sz > 1024*1024
                  else f"  {f}: {sz / 1024:.1f} KB")
        else:
            print(f"  {f}: MISSING")

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
