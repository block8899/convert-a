import site, os, sys

dysample_path = None
for sp in site.getsitepackages():
    p = os.path.join(sp, "spandrel", "architectures",
                     "__arch_helpers", "dysample.py")
    if os.path.exists(p):
        dysample_path = p
        break

assert dysample_path, "dysample.py not found!"
print(f"Patching: {dysample_path}")

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
    '        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale\n'
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
    "        # PATCHED: use affine_grid instead of arange/meshgrid\n"
    "        # arange(H)/arange(W) causes PNNX to create huge MemoryData nodes\n"
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

with open(dysample_path) as f:
    content = f.read()
assert "affine_grid" in content, "Patch failed: affine_grid not found"
assert "arange(H)" not in content, "Patch failed: old arange(H) still present"
print("Patch verified OK")
