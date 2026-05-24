import torch
import torch.nn as nn
import os
import sys
import shutil
import subprocess
import gc


class BiSeNetWrapper(nn.Module):
    """Return only main output, drop aux outputs."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def main():
    print("=== BiSeNet -> NCNN (fp32 + fp16) ===\n")

    sys.path.insert(0, 'repo_bisenet')
    from model import BiSeNet

    # 1. Load model
    model = BiSeNet(n_classes=19)
    state_dict = torch.load("repo_bisenet/79999_iter.pth", map_location='cpu')
    clean = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=False)
    model.eval()
    print("Weights loaded!")

    # 2. Wrap + trace
    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

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

    del wrapper, model, dummy, traced, out
    gc.collect()

    # 3. PNNX convert
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
    print(f"  stdout (last 300): {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-300:]}")

    # Fallback: ONNX path
    if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
        print("\nPNNX failed, trying ONNX fallback...")
        from model import BiSeNet as BS2
        m2 = BS2(n_classes=19)
        m2.load_state_dict(clean, strict=False)
        m2.eval()
        w2 = BiSeNetWrapper(m2)
        w2.eval()

        onnx_file = "bisenet.onnx"
        torch.onnx.export(
            w2, torch.randn(1, 3, 512, 512), onnx_file,
            input_names=["input"], output_names=["output"],
            opset_version=11,
        )
        print(f"  ONNX: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

        sim_file = "bisenet_sim.onnx"
        subprocess.run([sys.executable, "-m", "onnxsim", onnx_file, sim_file],
                       capture_output=True, text=True)
        src = sim_file if os.path.exists(sim_file) else onnx_file
        ret = subprocess.run(["pnnx", src], capture_output=True, text=True, timeout=300)
        print(f"  PNNX stdout (last 300): {ret.stdout[-300:]}")

        if not os.path.exists(pnnx_param) or not os.path.exists(pnnx_bin):
            print("ERROR: All conversion paths failed!")
            sys.exit(1)

    fp32_size = os.path.getsize(pnnx_bin)
    print(f"\nPNNX OK: {fp32_size / 1024 / 1024:.1f} MB")

    # 4. Copy fp32 to output
    os.makedirs("output", exist_ok=True)
    shutil.copy(pnnx_param, "output/biSeNet.param")
    shutil.copy(pnnx_bin, "output/biSeNet.bin")
    print("fp32 copied to output/")

    # 5. Convert to fp16
    print("\nGenerating fp16 version...")
    ret = subprocess.run(
        [sys.executable, "scripts/ncnn_fp16_convert.py",
         "output/biSeNet.bin", "output/biSeNet_fp16.bin"],
        capture_output=True, text=True, timeout=120,
    )
    print(f"  {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"  fp16 failed:\n{ret.stderr[-500:]}")
    else:
        shutil.copy("output/biSeNet.param", "output/biSeNet_fp16.param")

    # 6. Verify
    print("\n=== Output ===")
    for f in ["output/biSeNet.param", "output/biSeNet.bin",
              "output/biSeNet_fp16.param", "output/biSeNet_fp16.bin"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            print(f"  {f}: {sz / 1024 / 1024:.1f} MB" if sz > 1024*1024
                  else f"  {f}: {sz / 1024:.1f} KB")
        else:
            print(f"  {f}: MISSING")

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
