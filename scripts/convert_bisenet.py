# scripts/convert_bisenet.py

import torch
import torch.nn as nn
import os
import sys
import shutil
import subprocess
import gc


class BiSeNetWrapper(nn.Module):
    """Wrapper to return only main output (drop aux outputs)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def main():
    print("=== BiSeNet → NCNN (via PNNX) ===\n")

    sys.path.insert(0, 'repo_bisenet')
    from model import BiSeNet

    # 1. Load model
    model = BiSeNet(n_classes=19)
    weight_path = "repo_bisenet/79999_iter.pth"
    state_dict = torch.load(weight_path, map_location='cpu')
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

    # 3. TorchScript trace
    print("\nTracing with TorchScript...")
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward: {list(dummy.shape)} -> {list(out.shape)}")

    traced = torch.jit.trace(wrapper, dummy)
    pt_file = "bisenet_traced.pt"
    traced.save(pt_file)
    print(f"Traced saved: {os.path.getsize(pt_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy, traced
    gc.collect()

    # 4. Convert via PNNX
    print("\nConverting via PNNX...")
    pnnx_param = "bisenet_traced.ncnn.param"
    pnnx_bin = "bisenet_traced.ncnn.bin"

    for f in [pnnx_param, pnnx_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", pt_file, "inputshape=[1,3,512,512]"],
        capture_output=True, text=True, timeout=300,
    )
    print(f"  stdout (last 500): {ret.stdout[-500:]}")
    if ret.returncode != 0:
        print(f"  stderr (last 500): {ret.stderr[-500:]}")

    # 5. Check PNNX output
    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("\nPNNX failed — trying ONNX fallback...")

        model = BiSeNet(n_classes=19)
        model.load_state_dict(clean, strict=False)
        model.eval()
        wrapper = BiSeNetWrapper(model)
        wrapper.eval()

        onnx_file = "bisenet.onnx"
        torch.onnx.export(
            wrapper, torch.randn(1, 3, 512, 512), onnx_file,
            input_names=["input"], output_names=["output"],
            opset_version=11,
        )
        print(f"  ONNX exported: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

        sim_file = "bisenet_sim.onnx"
        subprocess.run([sys.executable, "-m", "onnxsim", onnx_file, sim_file])

        # Try PNNX with simplified ONNX
        ret = subprocess.run(
            ["pnnx", sim_file if os.path.exists(sim_file) else onnx_file],
            capture_output=True, text=True, timeout=300,
        )
        print(f"  PNNX stdout (last 300): {ret.stdout[-300:]}")

        if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
            print("ERROR: Both PNNX paths failed!")
            sys.exit(1)

    print(f"\nPNNX OK: {os.path.getsize(pnnx_bin) / 1024 / 1024:.1f} MB")

    # 6. Copy to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/biSeNet.param")
    shutil.copy(pnnx_bin, "output/biSeNet.bin")

    # 7. Verify
    print("\n=== Output ===")
    for f in ["output/biSeNet.param", "output/biSeNet.bin"]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  {f}: {size / 1024 / 1024:.1f} MB"
                  if size > 1024 * 1024
                  else f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  MISSING: {f}")
            sys.exit(1)

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
