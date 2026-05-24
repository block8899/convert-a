# scripts/convert_bisenet.py

import torch
import torch.nn as nn
import os
import sys
import subprocess
import gc


class BiSeNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def main():
    print("=== BiSeNet → NCNN ===\n")

    sys.path.insert(0, 'repo_bisenet')
    from model import BiSeNet

    # 1. Load
    model = BiSeNet(n_classes=19)
    state_dict = torch.load("repo_bisenet/79999_iter.pth", map_location='cpu')
    clean = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        clean[name] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    print("Weights loaded!")

    # 2. Wrap
    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

    # 3. Forward test
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward: {list(dummy.shape)} → {list(out.shape)}")

    # 4. Export ONNX
    print("\nExporting ONNX...")
    onnx_file = "bisenet.onnx"
    torch.onnx.export(
        wrapper, dummy, onnx_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"  ONNX: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy
    gc.collect()

    # 5. Simplify
    print("Simplifying...")
    sim_file = "bisenet_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  onnxsim failed: {ret.stderr[:300]}")
        sim_file = onnx_file
    else:
        print(f"  OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 6. onnx2ncnn
    print("Converting to NCNN...")
    os.makedirs("output", exist_ok=True)
    raw_param = "bisenet_raw.param"
    raw_bin = "bisenet_raw.bin"

    ret = subprocess.run(
        ["onnx2ncnn", sim_file, raw_param, raw_bin],
        capture_output=True, text=True,
    )
    if ret.stderr:
        print(f"  onnx2ncnn: {ret.stderr[:300]}")

    if not os.path.exists(raw_param):
        print("FAILED: onnx2ncnn produced no output")
        sys.exit(1)

    # 7. ncnnoptimize
    print("Optimizing...")
    ret = subprocess.run(
        ["ncnnoptimize",
         raw_param, raw_bin,
         "output/biSeNet.param", "output/biSeNet.bin",
         "65536"],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  ncnnoptimize failed: {ret.stderr[:200]}")
        import shutil
        shutil.copy(raw_param, "output/biSeNet.param")
        shutil.copy(raw_bin, "output/biSeNet.bin")

    # 8. Verify
    print("\n=== Output ===")
    for f in ["output/biSeNet.param", "output/biSeNet.bin"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
