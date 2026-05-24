# scripts/convert_animeganv3.py
# Đổi tên file cho đúng: convert_animegan.py

import torch
import torch.nn as nn
import pnnx
import os
import sys
import shutil

# ═══════════════════════════════════════════════════
# AnimeGANv2 Generator (bryandlee/animegan2-pytorch)
# Face Paint 512 v2 — Hayao style
# Input: [1, 3, H, W] normalized [-1, 1]
# Output: [1, 3, H, W] normalized [-1, 1]
# ═══════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """AnimeGANv2/v3 generator — encoder-bottleneck-decoder"""
    def __init__(self, base_ch=64):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, base_ch, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResBlock(base_ch * 4) for _ in range(8)])

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, 3, 7, 1, 0),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.res_blocks(e3)

        d3 = self.dec3(b)
        d2 = self.dec2(d3)
        out = self.dec1(d2)
        return out


# ═══════════════════════════════════════════════════
# CONVERT
# ═══════════════════════════════════════════════════

def main():
    print("=== AnimeGANv2 → NCNN ===\n")

    # 1. Create model
    print("1. Creating Generator...")
    model = Generator(base_ch=64)
    model.eval()

    # 2. Load weights
    # bryandlee repo provides .pt files directly
    weight_paths = [
        "repo_animegan/face_paint_512_v2.pt",       # Hayao portrait
        "repo_animegan/face_paint_512_v2_0.pt",
        "repo_animegan/checkpoints/face_paint_512_v2.pt",
    ]

    loaded = False
    for wp in weight_paths:
        if os.path.exists(wp):
            print(f"   Loading: {wp}")
            checkpoint = torch.load(wp, map_location='cpu')

            if isinstance(checkpoint, dict):
                for key in ['generator', 'state_dict', 'model', 'g_ema']:
                    if key in checkpoint:
                        checkpoint = checkpoint[key]
                        break

            # Remove prefixes
            clean = {}
            for k, v in checkpoint.items():
                name = k
                for prefix in ['module.', 'generator.', 'model.']:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                clean[name] = v

            model.load_state_dict(clean, strict=False)
            print("   Weights loaded!")
            loaded = True
            break

    if not loaded:
        print("   WARNING: No weights found — using random weights")
        print("   Download face_paint_512_v2.pt from:")
        print("   https://github.com/bryandlee/animegan2-pytorch")

    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # 3. Export via PNNX
    print("\n2. Converting via PNNX...")
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    try:
        pnnx.export(model, "animegan", inputs=dummy)
        print("   PNNX export done!")
    except Exception as e:
        print(f"   PNNX failed: {e}")
        sys.exit(1)

    del model, dummy
    import gc
    gc.collect()

    # 4. Move outputs
    print("\n3. Collecting output files...")
    os.makedirs("output", exist_ok=True)

    for suffix in [".ncnn.param", ".ncnn.bin"]:
        src = f"animegan{suffix}"
        dst = f"output/animegan{suffix}"
        if os.path.exists(src):
            shutil.move(src, dst)
            size = os.path.getsize(dst) / 1024
            print(f"   {dst}: {size:.1f} KB")
        else:
            print(f"   MISSING: {src}")
            sys.exit(1)

    print("\nAnimeGAN conversion OK!")


if __name__ == "__main__":
    main()
