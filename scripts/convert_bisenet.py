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


def run_pnnx(input_file, extra_args=None, label=""):
    """Run PNNX and return (param_path, bin_path) or None."""
    base = os.path.splitext(os.path.basename(input_file))[0]
    out_param = f"{base}.ncnn.param"
    out_bin = f"{base}.ncnn.bin"

    for f in [out_param, out_bin]:
        if os.path.exists(f):
            os.remove(f)

    cmd = ["pnnx", input_file] + (extra_args or [])
    print(f"  Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(f"  stdout (last 300): {ret.stdout[-300:]}")
    if ret.returncode != 0:
        print(f"  stderr (last 300): {ret.stderr[-300:]}")

    if not os.path.exists(out_param) or not os.path.exists(out_bin):
        print(f"  PNNX {label} output not found!")
        return None, None

    return out_param, out_bin


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

    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward: {list(dummy.shape)} -> {list(out.shape)}")

    # 2. Export to ONNX (needed for fp16 path)
    print("\nExporting to ONNX...")
    onnx_file = "bisenet.onnx"
    torch.onnx.export(
        wrapper, dummy, onnx_file,
        input_names=["input"], output_names=["output"],
        opset_version=11,
    )
    onnx_sz = os.path.getsize(onnx_file) / 1024 / 1024
    print(f"  Saved: {onnx_sz:.1f} MB")

    # 3. Trace TorchScript (primary fp32 path)
    print("\nTracing TorchScript...")
    traced = torch.jit.trace(wrapper, dummy)
    pt_file = "bisenet_traced.pt"
    traced.save(pt_file)
    print(f"  Saved: {os.path.getsize(pt_file) / 1024 / 1024:.1f} MB")

    del wrapper, model, dummy, traced, out
    gc.collect()

    os.makedirs("output", exist_ok=True)

    # 4. fp32: PNNX (TorchScript primary, ONNX fallback)
    print("\n--- fp32 conversion ---")
    fp32_p, fp32_b = run_pnnx(pt_file, extra_args=["inputshape=[1,3,512,512]"],
                               label="fp32-torchscript")

    if not fp32_p:
        print("TorchScript PNNX failed, trying ONNX fallback...")
        sim_file = "bisenet_sim.onnx"
        subprocess.run([sys.executable, "-m", "onnxsim", onnx_file, sim_file],
                       capture_output=True, text=True)
        src = sim_file if os.path.exists(sim_file) else onnx_file
        fp32_p, fp32_b = run_pnnx(src, label="fp32-onnx")

        if not fp32_p:
            print("ERROR: All fp32 paths failed!")
            sys.exit(1)

    shutil.copy(fp32_p, "output/biSeNet.param")
    shutil.copy(fp32_b, "output/biSeNet.bin")
    fp32_size = os.path.getsize(fp32_b)
    print(f"  fp32 OK: {fp32_size / 1024 / 1024:.1f} MB")

    # 5. fp16: PNNX fp16=1 (use ONNX with different name)
    print("\n--- fp16 conversion ---")
    fp16_onnx = "bisenet_fp16_input.onnx"
    shutil.copy(onnx_file, fp16_onnx)

    fp16_p, fp16_b = run_pnnx(fp16_onnx, extra_args=["fp16=1"],
                               label="fp16")

    if not fp16_p:
        print("PNNX fp16=1 failed, trying ONNX weight conversion fallback...")

        import onnx
        from onnx import numpy_helper, TensorProto
        import numpy as np

        model = onnx.load(onnx_file)
        converted = 0
        for init in model.graph.initializer:
            if init.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(init).astype(np.float16)
                init.CopyFrom(numpy_helper.from_array(arr, init.name))
                converted += 1
        onnx.save(model, fp16_onnx)
        print(f"  Converted {converted} tensors to fp16 in ONNX")

        fp16_p, fp16_b = run_pnnx(fp16_onnx, label="fp16-fallback")

        if not fp16_p:
            print("ERROR: All fp16 paths failed!")
            sys.exit(1)

    shutil.copy(fp16_p, "output/biSeNet_fp16.param")
    shutil.copy(fp16_b, "output/biSeNet_fp16.bin")
    fp16_size = os.path.getsize(fp16_b)
    pct = (1 - fp16_size / fp32_size) * 100 if fp32_size > 0 else 0
    print(f"  fp16 OK: {fp16_size / 1024 / 1024:.1f} MB (-{pct:.0f}%)")

    # 6. Verify
    print("\n=== Output ===")
    for f in ["output/biSeNet.param", "output/biSeNet.bin",
              "output/biSeNet_fp16.param", "output/biSeNet_fp16.bin"]:
        if os.path.exists(f):
            sz = os.path.getsize(f)
            unit = f"{sz / 1024 / 1024:.1f} MB" if sz > 1024 * 1024 else f"{sz / 1024:.1f} KB"
            print(f"  {f}: {unit}")
        else:
            print(f"  {f}: MISSING")

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
