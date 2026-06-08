#!/usr/bin/env python3
"""
Helper: Convert RealPLKSR PTH → ONNX (for NCNN pipeline).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
#  RealPLKSR Architecture (self-contained)
# ═══════════════════════════════════════════════════════════════════

class PLKBlock(nn.Module):
    def __init__(self, dim, kernel_size=17, dilation=1, reduction=4):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size,
                                 padding=padding, groups=dim, dilation=dilation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, dim // reduction), nn.GELU(),
            nn.Linear(dim // reduction, dim), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.pw_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        residual = x
        x = self.dw_conv(x)
        ca = self.channel_attention(x)
        x = x * ca.unsqueeze(-1).unsqueeze(-1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pw_conv(x)
        return x + residual


class RealPLKSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64,
                 n_blocks=28, upscaling_factor=1, kernel_size=17, dilation=1):
        super().__init__()
        self.head = nn.Conv2d(in_channels, dim, 3, 1, 1)
        self.body = nn.Sequential(
            *[PLKBlock(dim, kernel_size, dilation) for _ in range(n_blocks)]
        )
        self.upsample = None
        if upscaling_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(dim, dim * (upscaling_factor ** 2), 3, 1, 1),
                nn.PixelShuffle(upscaling_factor),
            )
        self.tail = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.tail(x)
        return x


# ═══════════════════════════════════════════════════════════════════
#  Auto-detect params from state_dict
# ═══════════════════════════════════════════════════════════════════

def detect_params(state_dict):
    dim = None
    n_blocks = 0
    kernel_size = None
    in_ch = 3
    out_ch = 3
    block_ids = set()

    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if 'head' in k and 'weight' in k and len(v.shape) == 4:
            dim = v.shape[0]
            in_ch = v.shape[1]
        if 'tail' in k and 'weight' in k and len(v.shape) == 4:
            out_ch = v.shape[0]
        if ('dw_conv' in k) and 'weight' in k and len(v.shape) == 4 and v.shape[2] > 1:
            kernel_size = v.shape[2]
        for part in k.split('.'):
            if part.isdigit():
                block_ids.add(int(part))

    if block_ids:
        n_blocks = max(block_ids) + 1

    return dict(in_channels=in_ch, out_channels=out_ch,
                dim=dim or 64, n_blocks=n_blocks or 28,
                kernel_size=kernel_size or 17)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='model.onnx')
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--n_blocks', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=0)
    parser.add_argument('--input_shape', default='1,3,256,256')
    args = parser.parse_args()

    # Load
    ckpt = torch.load(args.input, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ['params', 'params_ema', 'state_dict', 'model']:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Clean keys
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Detect / override params
    p = detect_params(state_dict)
    if args.dim > 0:
        p['dim'] = args.dim
    if args.n_blocks > 0:
        p['n_blocks'] = args.n_blocks
    if args.kernel_size > 0:
        p['kernel_size'] = args.kernel_size

    p['upscaling_factor'] = 1  # 1x denoising
    print(f"Model params: {p}")

    # Create model
    model = RealPLKSR(**p)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Quick test
    with torch.no_grad():
        t = torch.randn(1, 3, 64, 64)
        o = model(t)
        print(f"Test: input={t.shape} output={o.shape}")

    # Export ONNX
    shape = tuple(map(int, args.input_shape.split(',')))
    dummy = torch.randn(*shape)

    torch.onnx.export(
        model, dummy, args.output,
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        },
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"✅ ONNX saved: {args.output} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()
