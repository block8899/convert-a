# scripts/convert_bisenet.py

import torch
import torch.nn as nn
import pnnx
import os
import sys
import shutil
import gc


# ═══════════════════════════════════════════════════
# Wrapper — chỉ lấy output chính cho NCNN
# ═══════════════════════════════════════════════════

class BiSeNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # Repo trả về tuple (main, aux1, aux2) hoặc list
        if isinstance(out, (tuple, list)):
            return out[0]  # Chỉ lấy output chính
        return out


def main():
    print("=== BiSeNet → NCNN ===\n")

    # 1. Import từ repo
    sys.path.insert(0, 'repo_bisenet')
    try:
        from model import BiSeNet
        print("Imported BiSeNet from repo_bisenet/model.py")
    except ImportError as e:
        print(f"Cannot import: {e}")
        print("Available files:")
        for f in os.listdir('repo_bisenet'):
            print(f"  {f}")
        sys.exit(1)

    # 2. Create + wrap
    model = BiSeNet(n_classes=19)
    model.eval()

    # 3. Load weights
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

    # 4. Wrap — chỉ lấy output chính
    wrapper = BiSeNetWrapper(model)
    wrapper.eval()

    params = sum(p.numel() for p in wrapper.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # 5. Test forward
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            out = wrapper(dummy)
        print(f"  Forward OK: input={list(dummy.shape)} → output={list(out.shape)}")
    except Exception as e:
        print(f"  Forward failed: {e}")
        # Debug: show raw output type
        try:
            raw = model(dummy)
            print(f"  Raw output type: {type(raw)}")
            if isinstance(raw, (tuple, list)):
                for i, r in enumerate(raw):
                    if isinstance(r, torch.Tensor):
                        print(f"    [{i}]: {list(r.shape)}")
                    else:
                        print(f"    [{i}]: {type(r)}")
        except Exception as e2:
            print(f"  Raw forward also failed: {e2}")
        sys.exit(1)

    # 6. PNNX export
    print("\nConverting via PNNX...")
    try:
        pnnx.export(wrapper, "bisenet", inputs=dummy)
        print("  PNNX export done!")
    except Exception as e:
        print(f"  PNNX failed: {e}")
        sys.exit(1)

    del wrapper, model, dummy
    gc.collect()

    # 7. Move outputs
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

    print("\nBiSeNet conversion OK!")


if __name__ == "__main__":
    main()
