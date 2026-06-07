import site, os, sys

dysample_path = None
for sp in site.getsitepackages():
    p = os.path.join(sp, "spandrel", "architectures",
                     "__arch_helpers", "dysample.py")
    if os.path.exists(p):
        dysample_path = p
        break

assert dysample_path, "dysample.py not found!"
print(f"Found: {dysample_path}")

# Read original to confirm what we're replacing
with open(dysample_path) as f:
    original = f.read()
print("Original forward contains arange:", "coords_h = torch.arange" in original)

new_content = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "\n"
    "\n"
    "class DySample(nn.Module):\n"
    '    """Adapted from Learning to Upsample by Learning to Sample"""\n'
    "\n"
    "    def __init__(self, in_channels, out_ch, scale=2, groups=4, end_convolution=True):\n"
    "        super().__init__()\n"
    "        try:\n"
    "            assert in_channels >= groups and in_channels % groups == 0\n"
    "        except:\n"
    '            raise ValueError("Incorrect in_channels and groups values.")\n'
    "        out_channels = 2 * groups * scale**2\n"
    "        self.scale = scale\n"
    "        self.groups = groups\n"
    "        self.end_convolution = end_convolution\n"
    "        if end_convolution:\n"
    "            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)\n"
    "        self.offset = nn.Conv2d(in_channels, out_channels, 1)\n"
    "        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)\n"
    "        if self.training:\n"
    "            nn.init.trunc_normal_(self.offset.weight, std=0.02)\n"
    "            nn.init.constant_(self.scope.weight, val=0)\n"
    '        self.register_buffer("init_pos", self._init_pos())\n'
    "\n"
    "    def _init_pos(self):\n"
    "        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale\n"
    "        return (\n"
    '            torch.stack(torch.meshgrid([h, h], indexing="ij"))\n'
    "            .transpose(1, 2)\n"
    "            .repeat(1, self.groups, 1)\n"
    "            .reshape(1, -1, 1, 1)\n"
    "        )\n"
    "\n"
    "    def forward(self, x):\n"
    "        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos\n"
    "        B, _, H, W = offset.shape\n"
    "        offset = offset.view(B, 2, -1, H, W)\n"
    "        # PATCHED: affine_grid replaces arange/meshgrid coords\n"
    "        # Original used torch.arange(H) + torch.arange(W) which PNNX\n"
    "        # serializes as huge MemoryData nodes, crashing pass_level0.\n"
    "        identity = torch.zeros(B, 2, 3, dtype=x.dtype, device=x.device)\n"
    "        identity[:, 0, 0] = 1.0\n"
    "        identity[:, 1, 1] = 1.0\n"
    "        grid = F.affine_grid(identity, (B, 1, H, W), align_corners=False)\n"
    "        coords = (\n"
    "            (grid + 1.0) * 0.5\n"
    "            * torch.tensor([W, H], dtype=x.dtype, device=x.device)\n"
    "            - 0.5\n"
    "        ).permute(0, 3, 1, 2).unsqueeze(2)\n"
    "        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)\n"
    "        coords = 2 * (coords + offset) / normalizer - 1\n"
    "        coords = (\n"
    "            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)\n"
    "            .view(B, 2, -1, self.scale * H, self.scale * W)\n"
    "            .permute(0, 2, 3, 4, 1)\n"
    "            .contiguous()\n"
    "            .flatten(0, 1)\n"
    "        )\n"
    "        output = F.grid_sample(\n"
    "            x.reshape(B * self.groups, -1, H, W),\n"
    "            coords,\n"
    '            mode="bilinear",\n'
    "            align_corners=False,\n"
    '            padding_mode="border",\n'
    "        ).view(B, -1, self.scale * H, self.scale * W)\n"
    "        if self.end_convolution:\n"
    "            output = self.end_conv(output)\n"
    "        return output\n"
)

with open(dysample_path, "w") as f:
    f.write(new_content)

# Verify
with open(dysample_path) as f:
    content = f.read()

print("After patch:")
print("  affine_grid present :", "affine_grid" in content)
print("  coords_h arange gone:", "coords_h = torch.arange" not in content)
print("  coords_w arange gone:", "coords_w = torch.arange" not in content)

assert "affine_grid" in content, "FAIL: affine_grid not written"
assert "coords_h = torch.arange" not in content, "FAIL: coords_h arange still present"
assert "coords_w = torch.arange" not in content, "FAIL: coords_w arange still present"

print("Patch OK")
