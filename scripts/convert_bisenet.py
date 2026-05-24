# scripts/convert_bisenet.py
import torch
import torch.nn as nn
import pnnx
import os
import sys
import shutil
import torchvision.models as models

# ═══════════════════════════════════════════════════
# BiSeNet Architecture (face-parsing.PyTorch)
# ResNet-18 backbone + Spatial/Context Path
# Input: [1, 3, 512, 512]
# Output: [1, 19, 512, 512]
# ═══════════════════════════════════════════════════

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
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
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
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
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
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

        feat32_arm = self.arm32(feat_cp)
        feat32_arm = feat32_arm + avg
        feat32_arm = self.conv_head32(feat32_arm)
        feat32_arm = nn.functional.interpolate(feat32_arm, size=feat16.size()[2:], mode='nearest')

        feat16_arm = self.arm16(feat16)
        feat16_arm = feat16_arm + feat32_arm
        feat16_arm = self.conv_head16(feat16_arm)

        return feat8, feat16, feat16_arm, feat32_arm


class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv_out(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
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
        atten = atten * feat
        out = atten + feat
        return out


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 64, n_classes)

    def forward(self, x):
        feat_sp = self.sp(x)
        feat8, feat16, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        # Upsample to input size
        feat_out = self.conv_out(feat_fuse)
        out = nn.functional.interpolate(feat_out, size=x.size()[2:], mode='bilinear', align_corners=True)
        return out


# ═══════════════════════════════════════════════════
# CONVERT
# ═══════════════════════════════════════════════════

def main():
    print("=== BiSeNet Face Parsing → NCNN ===\n")

    # 1. Create model
    print("1. Creating BiSeNet model...")
    model = BiSeNet(n_classes=19)
    model.eval()

    # 2. Load pretrained weights
    weight_path = "repo_bisenet/79999_iter.pth"
    if os.path.exists(weight_path):
        print(f"   Loading weights: {weight_path}")
        state_dict = torch.load(weight_path, map_location='cpu')
        # Remove 'module.' prefix if DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("   Weights loaded!")
    else:
        print(f"   WARNING: {weight_path} not found — using random weights")
        print("   Download from: https://drive.google.com/file/d/154JgKbzCPWg2WDYjvGP3DdgVpfFljHXc")

    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # 3. Export via PNNX
    print("\n2. Converting via PNNX...")
    torch.set_grad_enabled(False)
    dummy = torch.randn(1, 3, 512, 512)

    try:
        pnnx.export(model, "bisenet", inputs=dummy)
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
        src = f"bisenet{suffix}"
        dst = f"output/biSeNet{suffix}"
        if os.path.exists(src):
            shutil.move(src, dst)
            size = os.path.getsize(dst) / 1024
            print(f"   {dst}: {size:.1f} KB")
        else:
            print(f"   MISSING: {src}")
            sys.exit(1)

    print("\nBiSeNet conversion OK!")


if __name__ == "__main__":
    main()
