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

    # 1. Simplify
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

    # 2. fp32 via PNNX
    print("\n2. Converting fp32 via PNNX...")
    for f in ["animegan_sim.ncnn.param", "animegan_sim.ncnn.bin"]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(["pnnx", sim_file], capture_output=True, text=True, timeout=300)
    if ret.stdout.strip():
        print(f"   stdout (last 200): {ret.stdout[-200:]}")

    p_param, p_bin = "animegan_sim.ncnn.param", "animegan_sim.ncnn.bin"
    if not os.path.exists(p_param) or not os.path.exists(p_bin):
        print("   PNNX failed!")
        sys.exit(1)

    shutil.copy(p_param, "output/animegan.param")
    shutil.copy(p_bin, "output/animegan.bin")
    fp32_sz = os.path.getsize(p_bin)
    print(f"   OK: {fp32_sz / 1024 / 1024:.1f} MB")

    # 3. fp16 via custom converter
    print("\n3. Converting bin fp32 -> fp16...")
    ret = subprocess.run(
        [sys.executable, "scripts/ncnn_fp16_convert.py",
         "output/animegan.bin", "output/animegan_fp16.bin"],
        capture_output=True, text=True, timeout=120,
    )
    print(f"   {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"   FAILED:\n{ret.stderr[-500:]}")
        sys.exit(1)

    shutil.copy("output/animegan.param", "output/animegan_fp16.param")

    # 4. Verify
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
