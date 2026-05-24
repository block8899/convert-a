import os
import sys
import subprocess
import shutil


def run_pnnx(onnx_file, label=""):
    """Convert ONNX to NCNN via PNNX. Returns (param, bin) or exits."""
    base = os.path.splitext(os.path.basename(onnx_file))[0]
    out_param = f"{base}.ncnn.param"
    out_bin = f"{base}.ncnn.bin"

    for f in [out_param, out_bin]:
        if os.path.exists(f):
            os.remove(f)

    cmd = ["pnnx", onnx_file]
    print(f"   Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if ret.stdout.strip():
        print(f"   stdout (last 300): {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"   stderr (last 300): {ret.stderr[-300:]}")

    if not os.path.exists(out_param) or not os.path.exists(out_bin):
        print(f"   PNNX {label} failed!")
        return None, None

    return out_param, out_bin


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

    # 2. fp32: PNNX
    print("\n2. Converting fp32 via PNNX...")
    fp32_p, fp32_b = run_pnnx(sim_file, "fp32")
    if not fp32_p:
        sys.exit(1)

    shutil.copy(fp32_p, "output/animegan.param")
    shutil.copy(fp32_b, "output/animegan.bin")
    fp32_sz = os.path.getsize(fp32_b)
    print(f"   fp32 OK: {fp32_sz / 1024 / 1024:.1f} MB")

    # 3. fp16: convert ONNX weights -> PNNX
    print("\n3. Converting ONNX to fp16...")
    fp16_onnx = "animegan_fp16.onnx"
    ret = subprocess.run(
        [sys.executable, "scripts/convert_onnx_fp16.py", sim_file, fp16_onnx],
        capture_output=True, text=True, timeout=120,
    )
    print(f"   {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"   FAILED: {ret.stderr[-300:]}")
        sys.exit(1)

    print("\n4. Converting fp16 via PNNX...")
    fp16_p, fp16_b = run_pnnx(fp16_onnx, "fp16")
    if not fp16_p:
        print("   ERROR: fp16 conversion failed!")
        sys.exit(1)

    shutil.copy(fp16_p, "output/animegan_fp16.param")
    shutil.copy(fp16_b, "output/animegan_fp16.bin")
    fp16_sz = os.path.getsize(fp16_b)
    pct = (1 - fp16_sz / fp32_sz) * 100 if fp32_sz > 0 else 0
    print(f"   fp16 OK: {fp16_sz / 1024 / 1024:.1f} MB (-{pct:.0f}%)")

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
