import os
import sys
import subprocess
import shutil


def run_pnnx(input_file, extra_args=None, label=""):
    """Run PNNX and return (param_path, bin_path) or exit on failure."""
    base = os.path.splitext(os.path.basename(input_file))[0]
    out_param = f"{base}.ncnn.param"
    out_bin = f"{base}.ncnn.bin"

    for f in [out_param, out_bin]:
        if os.path.exists(f):
            os.remove(f)

    cmd = ["pnnx", input_file] + (extra_args or [])
    print(f"   Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"   stdout (last 300): {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"   stderr (last 300): {ret.stderr[-300:]}")

    if not os.path.exists(out_param) or not os.path.exists(out_bin):
        print(f"   PNNX {label} output not found!")
        return None, None

    return out_param, out_bin


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

    os.makedirs("output", exist_ok=True)

    # 2. fp32: PNNX (default)
    print("\n2. Converting fp32 via PNNX...")
    fp32_p, fp32_b = run_pnnx(sim_file, label="fp32")
    if not fp32_p:
        sys.exit(1)

    shutil.copy(fp32_p, "output/animegan.param")
    shutil.copy(fp32_b, "output/animegan.bin")
    fp32_size = os.path.getsize(fp32_b)
    print(f"   OK: {fp32_size / 1024 / 1024:.1f} MB")

    # 3. fp16: PNNX fp16=1 (use different filename to avoid overwrite)
    print("\n3. Converting fp16 via PNNX...")
    fp16_onnx = "animegan_fp16_input.onnx"
    shutil.copy(sim_file, fp16_onnx)

    fp16_p, fp16_b = run_pnnx(fp16_onnx, extra_args=["fp16=1"], label="fp16")
    if not fp16_p:
        print("   PNNX fp16=1 failed — falling back to ONNX weight conversion...")

        # Fallback: convert ONNX weights manually, then PNNX without fp16 flag
        import onnx
        from onnx import numpy_helper, TensorProto
        import numpy as np

        model = onnx.load(sim_file)
        converted = 0
        for init in model.graph.initializer:
            if init.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(init).astype(np.float16)
                init.CopyFrom(numpy_helper.from_array(arr, init.name))
                converted += 1
        onnx.save(model, fp16_onnx)
        print(f"   Converted {converted} tensors to fp16 in ONNX")

        fp16_p, fp16_b = run_pnnx(fp16_onnx, label="fp16-fallback")
        if not fp16_p:
            print("   ERROR: All fp16 paths failed!")
            sys.exit(1)

    shutil.copy(fp16_p, "output/animegan_fp16.param")
    shutil.copy(fp16_b, "output/animegan_fp16.bin")
    fp16_size = os.path.getsize(fp16_b)
    pct = (1 - fp16_size / fp32_size) * 100 if fp32_size > 0 else 0
    print(f"   OK: {fp16_size / 1024 / 1024:.1f} MB (-{pct:.0f}%)")

    # 4. Verify
    print("\n=== Output ===")
    for f in ["output/animegan.param", "output/animegan.bin",
              "output/animegan_fp16.param", "output/animegan_fp16.bin"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            unit = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.1f} KB"
            print(f"  {f}: {unit}")
        else:
            print(f"  {f}: MISSING")

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
