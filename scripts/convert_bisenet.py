# scripts/convert_bisenet.py

import torch
import pnnx
import os
import sys
import shutil
import gc

# ═══════════════════════════════════════════════════
# Import BiSeNet từ repo gốc — không tự define
# ═══════════════════════════════════════════════════

def main():
    print("=== BiSeNet → NCNN ===\n")

    # 1. Import model từ repo đã clone
    sys.path.insert(0, 'repo_bisenet')
    try:
        from model import BiSeNet
        print("Imported BiSeNet from repo_bisenet/model.py")
    except ImportError as e:
        print(f"Cannot import from repo_bisenet: {e}")
        print("Available files:")
        if os.path.exists('repo_bisenet'):
            for f in os.listdir('repo_bisenet'):
                print(f"  {f}")
            if os.path.exists('repo_bisenet/model'):
                for f in os.listdir('repo_bisenet/model'):
                    print(f"  model/{f}")
        sys.exit(1)

    # 2. Create model — 19 classes (CelebAMask-HQ)
    model = BiSeNet(n_classes=19)
    model.eval()

    # 3. Load weights
    weight_path = "repo_bisenet/79999_iter.pth"
    if not os.path.exists(weight_path):
        print(f"MISSING: {weight_path}")
        sys.exit(1)

    print(f"Loading: {weight_path}")
    state_dict = torch.load(weight_path, map_location='cpu')

    # Clean prefixes
    clean = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        clean[name] = v

    # Debug: show checkpoint keys
    print(f"\nCheckpoint keys: {len(clean)}")
    for k, v in sorted(clean.items()):
        print(f"  {k}: {list(v.shape)}")

    # Load
    try:
        model.load_state_dict(clean, strict=False)
        print("\nWeights loaded!")
    except Exception as e:
        print(f"\nLoad error: {e}")
        model_sd = model.state_dict()
        print("\n--- MISMATCHES ---")
        for k in clean:
            if k in model_sd and clean[k].shape != model_sd[k].shape:
                print(f"  {k}: checkpoint={list(clean[k].shape)} "
                      f"model={list(model_sd[k].shape)}")
        sys.exit(1)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # 4. Export via PNNX
    print("\nConverting via PNNX...")
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    # Test forward first
    print("  Testing forward pass...")
    try:
        with torch.no_grad():
            out = model(dummy)
        print(f"  Forward OK: input={list(dummy.shape)} → output={list(out.shape)}")
    except Exception as e:
        print(f"  Forward failed: {e}")
        sys.exit(1)

    # PNNX export
    try:
        pnnx.export(model, "bisenet", inputs=dummy)
        print("  PNNX export done!")
    except Exception as e:
        print(f"  PNNX failed: {e}")
        sys.exit(1)

    del model, dummy
    gc.collect()

    # 5. Move outputs
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
