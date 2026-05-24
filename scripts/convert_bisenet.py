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

    # 1. Load + trace fp32
    model = BiSeNet(n_classes=19)
    state_dict = torch.load("repo_bisenet/79999_iter.pth", map_location='cpu')
    clean = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=False)
    model.eval()
    print("Weights loaded!")

    wrapper = BiSeNetWrapper(model)
    wrapper.eval()
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward: {list(dummy.shape)} -> {list(out.shape)}")

    traced = torch.jit.trace(wrapper, dummy)
    pt_file = "bisenet_traced.pt"
    traced.save(pt_file)
    print(f"Traced: {os.path.getsize(pt_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy, traced, out
    gc.collect()

    os.makedirs("output", exist_ok=True)

    # 2. fp32 via PNNX
    print("\n--- fp32 ---")
    for f in ["bisenet_traced.ncnn.param", "bisenet_traced.ncnn.bin"]:
        if os.path.exists(f):
            os.remove(f)

    ret = subprocess.run(
        ["pnnx", pt_file, "inputshape=[1,3,512,512]"],
        capture_output=True, text=True, timeout=300,
    )
    if ret.stdout.strip():
        print(f"  stdout (last 200): {ret.stdout[-200:]}")

    p_param = "bisenet_traced.ncnn.param"
    p_bin = "bisenet_traced.ncnn.bin"

    if not os.path.exists(p_param) or not os.path.exists(p_bin):
        print("  PNNX failed, trying ONNX fallback...")
        sys.path.insert(0, 'repo_bisenet')
        from model import BiSeNet as BS2
        m2 = BS2(n_classes=19)
        m2.load_state_dict(clean, strict=False)
        m2.eval()
        w2 = BiSeNetWrapper(m2)
        w2.eval()

        onnx_file = "bisenet.onnx"
        torch.onnx.export(w2, torch.randn(1, 3, 512, 512), onnx_file,
                          input_names=["input"], output_names=["output"],
                          opset_version=11)

        sim_file = "bisenet_sim.onnx"
        subprocess.run([sys.executable, "-m", "onnxsim", onnx_file, sim_file],
                       capture_output=True, text=True)

        src = sim_file if os.path.exists(sim_file) else onnx_file
        subprocess.run(["pnnx", src], capture_output=True, text=True, timeout=300)

        if not os.path.exists(p_param) or not os.path.exists(p_bin):
            print("ERROR: All fp32 paths failed!")
            sys.exit(1)

    shutil.copy(p_param, "output/biSeNet.param")
    shutil.copy(p_bin, "output/biSeNet.bin")
    fp32_sz = os.path.getsize(p_bin)
    print(f"  fp32 OK: {fp32_sz / 1024 / 1024:.1f} MB")

    # 3. fp16 via custom converter
    print("\n--- fp16 ---")
    ret = subprocess.run(
        [sys.executable, "scripts/ncnn_fp16_convert.py",
         "output/biSeNet.bin", "output/biSeNet_fp16.bin"],
        capture_output=True, text=True, timeout=120,
    )
    print(f"  {ret.stdout.strip()}")
    if ret.returncode != 0:
        print(f"  FAILED:\n{ret.stderr[-500:]}")
        sys.exit(1)

    shutil.copy("output/biSeNet.param", "output/biSeNet_fp16.param")

    # 4. Verify
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
