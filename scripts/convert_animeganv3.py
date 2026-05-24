# scripts/convert_animeganv3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pnnx
import os
import sys
import shutil

# ═══════════════════════════════════════════════════
# AnimeGANv3 Portrait Generator Architecture
# Based on AnimeGANv3 (TachibanaYoshinori)
# Lightweight U-Net style generator
# Input: [1, 3, 512, 512] normalized [-1, 1]
# Output: [1, 3, 512, 512] normalized [-1, 1]
# ═══════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1,
                 bias=True, norm='none', activation='relu'):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias)]
        if norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_c))
        elif norm == 'batch':
            layers.append(nn.BatchNorm2d(out_c))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'lrelu':
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == 'none':
            pass
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


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


class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.act(self.norm(self.conv(x)))


class GeneratorV3(nn.Module):
    """
    AnimeGANv3 Portrait Generator
    Encoder-Decoder with skip connections
    3 encoder blocks → bottleneck → 3 decoder blocks
    """
    def __init__(self, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_ch, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.enc2 = DownSample(base_ch, base_ch * 2)
        self.enc3 = DownSample(base_ch * 2, base_ch * 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResBlock(base_ch * 4) for _ in range(4)])

        # Decoder
        self.dec3 = UpSample(base_ch * 4, base_ch * 2)
        self.dec2 = UpSample(base_ch * 2, base_ch)

        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, 3, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(b)
        d3 = d3 + e2  # skip connection
        d2 = self.dec2(d3)
        d2 = d2 + e1  # skip connection

        return self.out_conv(d2)


# ═══════════════════════════════════════════════════
# CONVERT
# ═══════════════════════════════════════════════════

def main():
    print("=== AnimeGANv3 Portrait → NCNN ===\n")

    # 1. Create model
    print("1. Creating AnimeGANv3 generator...")
    model = GeneratorV3(base_ch=64)
    model.eval()

    # 2. Load pretrained weights
    weight_paths = [
        "repo_animegan/checkpoints/hayao.pth",
        "repo_animegan/checkpoints/face_paint_512_v2.pt",
        "repo_animegan/checkpoints/AnimeGANv3_Hayao.pth",
    ]

    loaded = False
    for wp in weight_paths:
        if os.path.exists(wp):
            print(f"   Loading weights: {wp}")
            checkpoint = torch.load(wp, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'generator' in checkpoint:
                    state_dict = checkpoint['generator']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Remove common prefixes
            clean_dict = {}
            for k, v in state_dict.items():
                name = k
                for prefix in ['module.', 'generator.', 'model.']:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                clean_dict[name] = v

            model.load_state_dict(clean_dict, strict=False)
            print("   Weights loaded!")
            loaded = True
            break

    if not loaded:
        print("   WARNING: No weights found — using random weights")
        print("   Expected at: repo_animegan/checkpoints/hayao.pth")
        print("   Download: https://github.com/TachibanaYoshinori/AnimeGANv3/releases")

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

    print("\nAnimeGANv3 conversion OK!")


if __name__ == "__main__":
    main()
