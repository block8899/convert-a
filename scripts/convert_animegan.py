# scripts/convert_animegan.py

import torch
import pnnx
import onnx
import os
import sys
import shutil
import subprocess
import gc


def main():
    print("=== AnimeGANv3 ONNX → NCNN (PNNX) ===\n")

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
        print(f"   onnxsim failed, using original: {ret.stderr}")
        sim_file = onnx_file
    else:
        print(f"   Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 2. Load ONNX → PyTorch via onnx → PNNX export
    print("\n2. Converting via PNNX...")
    onnx_model = onnx.load(sim_file)

    # Get input shape from ONNX
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    print(f"   ONNX input shape: {input_shape}")

    # Use pnnx to convert ONNX directly
    # PNNX can load ONNX: pnnx.export(None, "animegan", inputfiles=[sim_file])
    # Alternative: load via torch and export
    dummy = torch.randn(input_shape)

    try:
        # PNNX supports ONNX input
        pnnx.export(None, "animegan", inputs=dummy, inputfiles=[sim_file])
        print("   PNNX from ONNX done!")
    except Exception as e:
        print(f"   PNNX from ONNX failed: {e}")
        # Fallback: manual ncnn param from onnx
        print("   Trying direct approach...")
        sys.exit(1)

    del dummy
    gc.collect()

    # 3. Move outputs
    print("\n3. Collecting outputs...")
    os.makedirs("output", exist_ok=True)

    for suffix in [".ncnn.param", ".ncnn.bin"]:
        src = f"animegan{suffix}"
        dst = f"output/animegan{suffix}"
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  {dst}: {os.path.getsize(dst) / 1024:.1f} KB")
        else:
            print(f"  MISSING: {src}")
            sys.exit(1)

    print("\nAnimeGANv3 OK!")


if __name__ == "__main__":
    main()
