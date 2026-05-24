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

    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

    # 2. Export ONNX
    print("\nExporting to ONNX...")
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)
    onnx_file = "bisenet.onnx"
    torch.onnx.export(
        wrapper, dummy, onnx_file,
        input_names=["input"], output_names=["output"],
        opset_version=11,
    )
    print(f"   ONNX: {os.path.getsize(onnx_file) / 1024 / 1024:.1f} MB")

    # 3. Simplify ONNX
    print("\nSimplifying ONNX...")
    sim_file = "bisenet_sim.onnx"
    ret = subprocess.run(
        [sys.executable, "-m", "onnxsim", onnx_file, sim_file],
        capture_output=True, text=True,
    )
    if ret.returncode != 0:
        print(f"   Failed, using original")
        sim_file = onnx_file
    else:
        print(f"   OK: {os.path.getsize(sim_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy
    gc.collect()

    os.makedirs("output", exist_ok=True)

    # 4. fp32: ONNX -> NCNN via PNNX
    print("\n--- fp32 ---")
    fp32_param = "bisenet_sim.ncnn.param"
    fp32_bin = "bisenet_sim.ncnn.bin"
    for f in [fp32_param, fp32_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", sim_file],
        capture_output=True, text=True, timeout=300,
    )
    print(f"  stdout (last 300): {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-300:]}")

    if not os.path.exists(fp32_param) or not os.path.exists(fp32_bin):
        # Fallback: trace + pnnx
        print("\nPNNX ONNX failed, trying TorchScript fallback...")
        from model import BiSeNet as BS2
        m2 = BS2(n_classes=19)
        m2.load_state_dict(clean, strict=False)
        m2.eval()
        w2 = BiSeNetWrapper(m2)
        w2.eval()
        traced = torch.jit.trace(w2, torch.randn(1, 3, 512, 512))
        pt_file = "bisenet_traced.pt"
        traced.save(pt_file)

        ret = subprocess.run(
            ["pnnx", pt_file, "inputshape=[1,3,512,512]"],
            capture_output=True, text=True, timeout=300,
        )
        print(f"  PNNX stdout (last 300): {ret.stdout[-300:]}")

        if not os.path.exists(fp32_param) or not os.path.exists(fp32_bin):
            print("ERROR: All fp32 paths failed!")
            sys.exit(1)

    shutil.copy(fp32_param, "output/biSeNet.param")
    shutil.copy(fp32_bin, "output/biSeNet.bin")
    fp32_sz = os.path.getsize(fp32_bin)
    print(f"  fp32 OK: {fp32_sz / 1024 / 1024:.1f} MB")

    # 5. fp16: convert ONNX weights -> PNNX
    print("\n--- fp16 ---")
    fp16_onnx = "bisenet_fp16.onnx"
    ret = subprocess.run(
        [sys.executable, "scripts/convert_onnx_fp16.py", sim_file, fp16_onnx],
        capture_output=True, text=True, timeout=120,
    )
    print(f"  {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"  FAILED: {ret.stderr[-300:]}")
        sys.exit(1)

    fp16_param = "bisenet_fp16.ncnn.param"
    fp16_bin = "bisenet_fp16.ncnn.bin"
    for f in [fp16_param, fp16_bin]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", fp16_onnx],
        capture_output=True, text=True, timeout=300,
    )
    if ret.returncode != 0:
        print(f"  stderr: {ret.stderr[-300:]}")

    if not os.path.exists(fp16_param) or not os.path.exists(fp16_bin):
        print("  PNNX fp16 output not found!")
        sys.exit(1)

    shutil.copy(fp16_param, "output/biSeNet_fp16.param")
    shutil.copy(fp16_bin, "output/biSeNet_fp16.bin")
    fp16_sz = os.path.getsize(fp16_bin)
    pct = (1 - fp16_sz / fp32_sz) * 100 if fp32_sz > 0 else 0
    print(f"  fp16 OK: {fp16_sz / 1024 / 1024:.1f} MB (-{pct:.0f}%)")

    # 6. Verify
    print("\n=== Output ===")
    for f in ["output/biSeNet.param", "output/biSeNet.bin",
              "output/biSeNet_fp16.param", "output/biSeNet_fp16.bin"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            unit = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024*1024 else f"{sz / 1024:.1f} KB"
            print(f"  {f}: {unit}")
        else:
            print(f"  {f}: MISSING")

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
