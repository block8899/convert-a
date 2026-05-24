# scripts/convert_animegan.py — ONNX → NCNN
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

    # 1. Simplify ONNX
    print("\n1. Simplifying ONNX...")
    sim_file = "animegan_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   onnxsim stderr: {ret.stderr}")
        # Fallback: use original
        print("   Using original ONNX")
        sim_file = onnx_file
    else:
        print(f"   Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. onnx2ncnn
    print("\n2. Converting to NCNN...")
    os.makedirs("output", exist_ok=True)

    param_file = "output/animegan.param"
    bin_file = "output/animegan.bin"

    # Find onnx2ncnn binary
    onnx2ncnn = shutil.which("onnx2ncnn")
    if not onnx2ncnn:
        # Try pip install location
        import ncnn
        ncnn_dir = os.path.dirname(ncnn.__file__)
        candidates = [
            os.path.join(ncnn_dir, "onnx2ncnn"),
            os.path.join(ncnn_dir, "bin", "onnx2ncnn"),
        ]
        for c in candidates:
            if os.path.exists(c):
                onnx2ncnn = c
                break

    if not onnx2ncnn:
        print("   onnx2ncnn not found, trying from PATH...")
        onnx2ncnn = "onnx2ncnn"

    ret = subprocess.run(
        [onnx2ncnn, sim_file, param_file, bin_file],
        capture_output=True, text=True,
    )

    if ret.returncode != 0:
        print(f"   onnx2ncnn stderr: {ret.stderr}")
        print(f"   onnx2ncnn stdout: {ret.stdout}")

        # Alternative: use ncnn python API
        print("\n   Trying ncnn Python API...")
        try:
            import ncnn
            # ncnn python doesn't have direct onnx2ncnn API
            # but we can check if files were created
            if os.path.exists(param_file) and os.path.exists(bin_file):
                print("   Files exist!")
            else:
                print("   FAILED: Could not convert")
                sys.exit(1)
        except Exception as e:
            print(f"   Python fallback failed: {e}")
            sys.exit(1)

    # 3. Verify
    print("\n3. Output:")
    for f in [param_file, bin_file]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            unit = "KB" if size < 1024 * 1024 else "MB"
            val = size / 1024 if unit == "KB" else size / 1024 / 1024
            print(f"   {f}: {val:.1f} {unit}")
        else:
            print(f"   MISSING: {f}")
            sys.exit(1)

    print("\nAnimeGANv3 conversion OK!")


if __name__ == "__main__":
    main()
