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

    # 1. Import từ repo
    sys.path.insert(0, 'repo_bisenet')
    try:
        from model import BiSeNet
        print("Imported BiSeNet from repo_bisenet/model.py")
    except ImportError as e:
        print(f"Cannot import: {e}")
        sys.exit(1)

    # 2. Create + load
    model = BiSeNet(n_classes=19)
    model.eval()

    weight_path = "repo_bisenet/79999_iter.pth"
    if not os.path.exists(weight_path):
        print(f"MISSING: {weight_path}")
        sys.exit(1)

    print(f"Loading: {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu')

    clean = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        clean[name] = v

    try:
        model.load_state_dict(clean, strict=False)
        print("Weights loaded!")
    except Exception as e:
        print(f"Load error: {e}")
        sys.exit(1)

    # 3. Wrap
    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

    params = sum(p.numel() for p in wrapper.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # 4. Test forward
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    print("\nTesting forward pass...")
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"  Forward OK: {list(dummy.shape)} → {list(out.shape)}")

    # 5. Export to ONNX
    print("\nExporting to ONNX...")
    onnx_file = "bisenet.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_file,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes=None,
    )
    print(f"  ONNX: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy
    gc.collect()

    # 6. Simplify ONNX
    print("\nSimplifying ONNX...")
    sim_file = "bisenet_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"  onnxsim failed: {ret.stderr}")
        print("  Using original ONNX")
        sim_file = onnx_file
    else:
        print(f"  Simplified: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    # 7. onnx2ncnn
    print("\nConverting to NCNN...")
    os.makedirs("output", exist_ok=True)

    param_file = "output/biSeNet.param"
    bin_file = "output/biSeNet.bin"

    # Find onnx2ncnn
    onnx2ncnn = shutil.which("onnx2ncnn")
    if not onnx2ncnn:
        onnx2ncnn = "onnx2ncnn"

    ret = subprocess.run(
        [onnx2ncnn, sim_file, param_file, bin_file],
        capture_output=True, text=True,
    )

    if ret.returncode != 0:
        print(f"  onnx2ncnn error: {ret.stderr}")
        if ret.stdout:
            print(f"  stdout: {ret.stdout}")
        sys.exit(1)

    # 8. Verify
    print("\nOutput:")
    for f in [param_file, bin_file]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                print(f"  {f}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nBiSeNet conversion OK!")


if __name__ == "__main__":
    main()
