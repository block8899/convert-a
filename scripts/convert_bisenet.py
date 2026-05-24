# scripts/convert_bisenet.py

import torch
import torch.nn as nn
import pnnx
import os
import sys
import shutil
import torchvision.models as models
import gc


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=2, keepdim=True)
        atten = torch.mean(atten, dim=3, keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32, feat_cp = self.resnet(x)
        avg = torch.mean(feat_cp, dim=2, keepdim=True)
        avg = torch.mean(avg, dim=3, keepdim=True)
        avg = self.conv_avg(avg)
        avg = nn.functional.interpolate(avg, size=feat_cp.size()[2:], mode='nearest')
        feat32_arm = self.arm32(feat_cp) + avg
        feat32_arm = self.conv_head32(feat32_arm)
        feat32_arm = nn.functional.interpolate(feat32_arm, size=feat16.size()[2:], mode='nearest')
        feat16_arm = self.arm16(feat16) + feat32_arm
        feat16_arm = self.conv_head16(feat16_arm)
        return feat8, feat16, feat16_arm, feat32_arm


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv_out(x)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=2, keepdim=True)
        atten = torch.mean(atten, dim=3, keepdim=True)
        atten = self.relu(self.conv1(atten))
        atten = self.sigmoid(self.conv2(atten))
        return atten * feat + feat


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        # ★ Fix: FFM output=128, BiSeNetOutput input=128
        self.ffm = FeatureFusionModule(256, 128)
        self.conv_out = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        feat_sp = self.sp(x)                           # 128 ch
        feat8, feat16, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)         # 128+128 → 128
        feat_out = self.conv_out(feat_fuse)              # 128 → 64 → 19
        out = nn.functional.interpolate(
            feat_out, size=x.size()[2:],
            mode='bilinear', align_corners=True)
        return out


def main():
    print("=== BiSeNet → NCNN ===\n")

    model = BiSeNet(n_classes=19)

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

    # ★ Debug: print all shapes from checkpoint
    print("\n--- Checkpoint shapes ---")
    for k, v in sorted(clean.items()):
        print(f"  {k}: {list(v.shape)}")

    # ★ Debug: print all shapes from model
    print("\n--- Model shapes ---")
    for k, v in sorted(model.state_dict().items()):
        print(f"  {k}: {list(v.shape)}")

    # Try load
    try:
        model.load_state_dict(clean, strict=False)
        print("\nWeights loaded (strict=False)")
    except Exception as e:
        print(f"\nLoad error: {e}")
        # ★ Auto-detect mismatch and suggest fix
        print("\n--- MISMATCHES ---")
        model_sd = model.state_dict()
        for k in clean:
            if k in model_sd and clean[k].shape != model_sd[k].shape:
                print(f"  {k}: checkpoint={list(clean[k].shape)} model={list(model_sd[k].shape)}")
        sys.exit(1)

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # Export via PNNX
    print("\nConverting via PNNX...")
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    try:
        pnnx.export(model, "bisenet", inputs=dummy)
        print("PNNX export done!")
    except Exception as e:
        print(f"PNNX failed: {e}")
        sys.exit(1)

    del model, dummy
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

    print("\nBiSeNet conversion OK!")


if __name__ == "__main__":
    main()
