# scripts/convert_bisenet.py

import torch
import torch.nn as nn
import os
import sys
import shutil
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
    print("=== BiSeNet → NCNN (via ONNX) ===\n")

    # 1. Import
    sys.path.insert(0, 'repo_bisenet')
    from model import BiSeNet

    # 2. Load
    model = BiSeNet(n_classes=19)
    state_dict = torch.load("repo_bisenet/79999_iter.pth", map_location='cpu')
    clean = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        clean[name] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    print("Weights loaded!")

    # 3. Wrap
    wrapper = BiSeNetWrapper(model)
    wrapper.eval()
    params = sum(p.numel() for p in wrapper.parameters())
    print(f"Parameters: {params:,}")

    # 4. Forward test
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward: {list(dummy.shape)} → {list(out.shape)}")

    # 5. Export ONNX
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

    # 6. Simplify
    print("Simplifying...")
    sim_file = "bisenet_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  onnxsim failed: {ret.stderr}")
        sim_file = onnx_file
    else:
        print(f"  Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 7. onnx2ncnn
    print("Converting to NCNN...")
    os.makedirs("output", exist_ok=True)
    param_f = "output/biSeNet.param"
    bin_f = "output/biSeNet.bin"

    ret = subprocess.run(
        ["onnx2ncnn", sim_file, param_f, bin_f],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  onnx2ncnn warning: {ret.stderr}")

    for f in [param_f, bin_f]:
        if os.path.exists(f):
            print(f"  {f}: {os.path.getsize(f) / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
