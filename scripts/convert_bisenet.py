# scripts/convert_bisenet.py

import torch
import torch.nn as nn
import pnnx
import os
import sys
import shutil
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
    print("=== BiSeNet → NCNN (PNNX) ===\n")

    sys.path.insert(0, 'repo_bisenet')
    from model import BiSeNet

    model = BiSeNet(n_classes=19)
    model.eval()

    weight_path = "repo_bisenet/79999_iter.pth"
    state_dict = torch.load(weight_path, map_location='cpu')

    clean = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        clean[name] = v

    model.load_state_dict(clean, strict=False)
    print("Weights loaded!")

    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Forward OK: {list(dummy.shape)} → {list(out.shape)}")

    print("\nExporting via PNNX...")
    pnnx.export(wrapper, "bisenet", inputs=dummy)
    print("PNNX export done!")

    del wrapper, model, dummy
    gc.collect()

    os.makedirs("output", exist_ok=True)
    for suffix in [".ncnn.param", ".ncnn.bin"]:
        src = f"bisenet{suffix}"
        dst = f"output/biSeNet{suffix}"
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  {dst}: {os.path.getsize(dst) / 1024:.1f} KB")
        else:
            print(f"  MISSING: {src}")
            sys.exit(1)

    print("\nBiSeNet OK!")


if __name__ == "__main__":
    main()
