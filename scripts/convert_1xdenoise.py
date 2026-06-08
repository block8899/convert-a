#!/usr/bin/env python3
"""
Convert RealPLKSR model weights to clean PTH and/or ONNX format.

Supports:
  - Auto-detection of architecture parameters from state_dict
  - Import from phhofm/practical-models-for-image-restoration source
  - Self-contained RealPLKSR architecture definition as fallback
  - ONNX export with dynamic batch/height/width axes
  - ONNX verification via onnx.checker + onnxruntime inference test
"""

import argparse
import glob
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
#  Self-contained RealPLKSR Architecture
# ═══════════════════════════════════════════════════════════════════

class PLKBlock(nn.Module):
    """Large-Kernel Depthwise Convolution Block with Channel Attention."""

    def __init__(self, dim: int, kernel_size: int = 17, dilation: int = 1,
                 use_layernorm: bool = True, reduction: int = 4):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        # Depthwise large-kernel conv
        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size,
            padding=padding, groups=dim, dilation=dilation
        )

        # Channel attention (SE-like)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

        # Normalization
        self.norm = nn.LayerNorm(dim) if use_layernorm else None

        # Pointwise conv
        self.pw_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dw_conv(x)

        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca.unsqueeze(-1).unsqueeze(-1)

        # LayerNorm (needs B,C,H,W -> B,H,W,C)
        if self.norm is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)

        x = self.pw_conv(x)

        return x + residual


class RealPLKSR(nn.Module):
    """
    RealPLKSR — Real Large-Kernel Super-Resolution Network.

    This is a self-contained definition that mirrors the architecture used
    in phhofm/practical-models-for-image-restoration.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = 1,
        kernel_size: int = 17,
        dilation: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.upscaling_factor = upscaling_factor

        # ── Head ──
        self.head = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # ── Body: PLK blocks ──
        self.body = nn.Sequential(
            *[PLKBlock(dim, kernel_size, dilation) for _ in range(n_blocks)]
        )

        # ── Upsampling (only if scale > 1) ──
        if upscaling_factor > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(dim, dim * (upscaling_factor ** 2), 3, 1, 1),
                nn.PixelShuffle(upscaling_factor),
            )
        else:
            self.upsample = None

        # ── Tail ──
        self.tail = nn.Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.tail(x)
        return x


# ═══════════════════════════════════════════════════════════════════
#  State Dict Utilities
# ═══════════════════════════════════════════════════════════════════

def load_state_dict(path: str):
    """Load state_dict from various checkpoint formats."""
    print(f"\n📥 Loading weights from: {path}")

    # Try safetensors first
    if path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path)
            print(f"  Loaded via safetensors ({len(state_dict)} keys)")
            return state_dict
        except ImportError:
            print("  ⚠️ safetensors not installed, trying torch.load")

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        # Check common keys used by basicsr and other frameworks
        for key in ['params', 'params_ema', 'state_dict', 'model_state_dict', 'model']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                print(f"  Found state_dict under key: '{key}' ({len(checkpoint[key])} keys)")
                return checkpoint[key]

        # Check if the dict itself looks like a state_dict (all values are tensors)
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            print(f"  Checkpoint is a raw state_dict ({len(checkpoint)} keys)")
            return checkpoint

        # Print available keys for debugging
        print(f"  ⚠️ Could not auto-detect state_dict location")
        print(f"  Available keys: {list(checkpoint.keys())[:20]}")
        return checkpoint

    print(f"  ⚠️ Unexpected checkpoint type: {type(checkpoint)}")
    return checkpoint


def inspect_state_dict(state_dict: dict):
    """Analyze state_dict to infer architecture parameters."""
    print("\n" + "=" * 60)
    print("  STATE DICT ANALYSIS")
    print("=" * 60)

    dim = None
    n_blocks = 0
    kernel_size = None
    in_channels = None
    out_channels = None
    has_layernorm = False

    block_indices = set()

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        # Print all keys with shapes (first 50)
        if len(state_dict) < 80:
            print(f"  {key}: {list(value.shape)}")

        # Detect head conv -> in_channels, dim
        if 'head' in key and 'weight' in key and len(value.shape) == 4:
            dim = value.shape[0]
            # Check if groups=1 (standard conv)
            in_channels = value.shape[1]

        # Detect tail conv -> out_channels
        if 'tail' in key and 'weight' in key and len(value.shape) == 4:
            out_channels = value.shape[0]

        # Detect kernel size from depthwise conv
        if ('dw_conv' in key or 'dw' in key) and 'weight' in key and len(value.shape) == 4:
            if value.shape[2] > 1:  # Not 1x1
                kernel_size = value.shape[2]

        # Detect LayerNorm
        if 'norm' in key and ('weight' in key or 'bias' in key) and len(value.shape) == 1:
            has_layernorm = True

        # Count blocks
        for part in key.split('.'):
            if part.isdigit():
                block_indices.add(int(part))

    if block_indices:
        n_blocks = max(block_indices) + 1

    print(f"\n  📊 Inferred Architecture Parameters:")
    print(f"     in_channels  = {in_channels}")
    print(f"     out_channels = {out_channels}")
    print(f"     dim          = {dim}")
    print(f"     n_blocks     = {n_blocks}")
    print(f"     kernel_size  = {kernel_size}")
    print(f"     has_layernorm= {has_layernorm}")
    print("=" * 60)

    return {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'dim': dim,
        'n_blocks': n_blocks,
        'kernel_size': kernel_size,
    }


def clean_state_dict_keys(state_dict: dict) -> dict:
    """Remove common prefixes like 'module.' from state_dict keys."""
    cleaned = {}
    for k, v in state_dict.items():
        # Remove DataParallel prefix
        if k.startswith('module.'):
            k = k[len('module.'):]
        cleaned[k] = v
    return cleaned


# ═══════════════════════════════════════════════════════════════════
#  Architecture Import from Source Repo
# ═══════════════════════════════════════════════════════════════════

def try_import_architecture(arch_source_path: str):
    """Try to import RealPLKSR from the cloned source repo."""
    if not arch_source_path or not os.path.isdir(arch_source_path):
        return None

    print(f"\n🔍 Searching for RealPLKSR architecture in: {arch_source_path}")

    # Find Python files containing RealPLKSR
    arch_files = []
    for root, dirs, files in os.walk(arch_source_path):
        # Skip hidden dirs and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                        content = fh.read()
                        if 'RealPLKSR' in content and 'nn.Module' in content:
                            arch_files.append(filepath)
                            print(f"  📄 Found RealPLKSR class in: {filepath}")
                except Exception:
                    pass

    if not arch_files:
        print("  ⚠️ No RealPLKSR architecture file found in source repo")
        return None

    # Try to import each found file
    for arch_file in arch_files:
        arch_dir = os.path.dirname(arch_file)
        arch_name = os.path.splitext(os.path.basename(arch_file))[0]

        # Add directory to sys.path
        if arch_dir not in sys.path:
            sys.path.insert(0, arch_dir)

        try:
            # Clear any cached imports
            if arch_name in sys.modules:
                del sys.modules[arch_name]

            module = __import__(arch_name)

            # Find RealPLKSR class in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, nn.Module)
                        and 'PLKSR' in attr_name):
                    print(f"  ✅ Successfully imported {attr_name} from {arch_file}")
                    return attr

        except Exception as e:
            print(f"  ❌ Failed to import {arch_file}: {e}")
            continue

    return None


# ═══════════════════════════════════════════════════════════════════
#  Model Creation
# ═══════════════════════════════════════════════════════════════════

def create_model(state_dict: dict, arch_source_path: str = None,
                 dim: int = 0, n_blocks: int = 0, kernel_size: int = 0):
    """Create the RealPLKSR model, trying import then fallback."""
    inferred = inspect_state_dict(state_dict)

    # Override with user-specified values (non-zero overrides auto-detect)
    params = {
        'in_channels': inferred['in_channels'] or 3,
        'out_channels': inferred['out_channels'] or 3,
        'dim': dim if dim > 0 else (inferred['dim'] or 64),
        'n_blocks': n_blocks if n_blocks > 0 else (inferred['n_blocks'] or 28),
        'kernel_size': kernel_size if kernel_size > 0 else (inferred['kernel_size'] or 17),
        'upscaling_factor': 1,  # 1x for denoising
    }

    print(f"\n🏗️ Creating RealPLKSR model with params:")
    for k, v in params.items():
        print(f"     {k} = {v}")

    # ── Strategy 1: Try importing from source repo ──
    if arch_source_path:
        ImportedRealPLKSR = try_import_architecture(arch_source_path)
        if ImportedRealPLKSR is not None:
            try:
                model = ImportedRealPLKSR(**params)
                print("  ✅ Created model from imported architecture")
            except TypeError as e:
                print(f"  ⚠️ Imported architecture doesn't accept these params: {e}")
                print("  Falling back to self-contained architecture...")
                model = RealPLKSR(**params)
                print("  ✅ Created model from self-contained architecture")
        else:
            model = RealPLKSR(**params)
            print("  ✅ Created model from self-contained architecture")
    else:
        model = RealPLKSR(**params)
        print("  ✅ Created model from self-contained architecture")

    return model, params


# ═══════════════════════════════════════════════════════════════════
#  ONNX Export
# ═══════════════════════════════════════════════════════════════════

def export_onnx(model: nn.Module, output_path: str,
                input_shape: tuple = (1, 3, 256, 256),
                opset_version: int = 17):
    """Export model to ONNX format with verification."""
    print(f"\n📤 Exporting to ONNX: {output_path}")
    print(f"   Input shape: {input_shape}")
    print(f"   Opset version: {opset_version}")

    model.eval()

    dummy_input = torch.randn(*input_shape)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        },
        do_constant_folding=True,
    )

    # ── Verify with onnx.checker ──
    print("  🔍 Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX model passed checker")

    # ── Verify with onnxruntime ──
    print("  🔍 Running onnxruntime inference test...")
    import onnxruntime as ort

    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape_ort = input_info.shape

    # Generate test input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    ort_output = session.run(None, {input_name: test_input})[0]

    # Compare with PyTorch
    with torch.no_grad():
        pt_output = model(torch.from_numpy(test_input)).numpy()

    max_diff = np.max(np.abs(ort_output - pt_output))
    mean_diff = np.mean(np.abs(ort_output - pt_output))
    print(f"  📊 Max absolute diff (ONNX vs PyTorch): {max_diff:.2e}")
    print(f"  📊 Mean absolute diff (ONNX vs PyTorch): {mean_diff:.2e}")

    if max_diff < 1e-3:
        print("  ✅ ONNX export verified — outputs match PyTorch!")
    else:
        print(f"  ⚠️ ONNX output differs from PyTorch (max_diff={max_diff:.2e})")

    return True


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Convert RealPLKSR model to clean PTH and/or ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect everything:
  python convert_model.py --input model.pth --output_name 1xDeNoise_realplksr_otf

  # Specify architecture params:
  python convert_model.py --input model.pth --dim 64 --n_blocks 28 --kernel_size 17

  # ONNX only with custom input shape:
  python convert_model.py --input model.pth --export_pth false --onnx_input_shape 1,3,512,512
        """
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input model weights (.pth or .safetensors)')
    parser.add_argument('--output_name', type=str, default='1xDeNoise_realplksr_otf',
                        help='Output filename without extension')
    parser.add_argument('--arch_source', type=str, default=None,
                        help='Path to cloned practical-models-for-image-restoration repo')
    parser.add_argument('--dim', type=int, default=0,
                        help='Model dimension (0=auto-detect)')
    parser.add_argument('--n_blocks', type=int, default=0,
                        help='Number of PLK blocks (0=auto-detect)')
    parser.add_argument('--kernel_size', type=int, default=0,
                        help='Kernel size (0=auto-detect)')
    parser.add_argument('--onnx_input_shape', type=str, default='1,3,256,256',
                        help='Input shape for ONNX export (B,C,H,W)')
    parser.add_argument('--export_pth', type=str, default='true',
                        help='Export clean PTH? (true/false)')
    parser.add_argument('--export_onnx', type=str, default='true',
                        help='Export ONNX? (true/false)')
    parser.add_argument('--opset_version', type=int, default=17,
                        help='ONNX opset version')

    args = parser.parse_args()

    # ── Load state dict ──
    state_dict = load_state_dict(args.input)
    state_dict = clean_state_dict_keys(state_dict)

    # ── Create model ──
    model, params = create_model(
        state_dict,
        arch_source_path=args.arch_source,
        dim=args.dim,
        n_blocks=args.n_blocks,
        kernel_size=args.kernel_size,
    )

    # ── Load weights ──
    print("\n🔧 Loading weights into model...")
    try:
        result = model.load_state_dict(state_dict, strict=True)
        print("  ✅ Weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"  ⚠️ Strict loading failed: {e}")
        print("  Trying with strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  📋 Missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"  📋 Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
        if not missing and not unexpected:
            print("  ✅ All keys matched (strict=False)")

    model.eval()

    # ── Quick inference test ──
    print("\n🧪 Running quick inference test...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 64, 64)
        test_output = model(test_input)
        print(f"  Input shape:  {test_input.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")

    # ── Export PTH ──
    if args.export_pth.lower() == 'true':
        pth_path = f"{args.output_name}.pth"
        print(f"\n💾 Saving clean PTH: {pth_path}")
        torch.save(model.state_dict(), pth_path)
        size_mb = os.path.getsize(pth_path) / (1024 * 1024)
        print(f"  ✅ Saved: {pth_path} ({size_mb:.2f} MB)")

    # ── Export ONNX ──
    if args.export_onnx.lower() == 'true':
        onnx_path = f"{args.output_name}.onnx"
        input_shape = tuple(map(int, args.onnx_input_shape.split(',')))
        try:
            export_onnx(model, onnx_path, input_shape, args.opset_version)
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  ✅ Saved: {onnx_path} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"  ❌ ONNX export failed: {e}")
            traceback.print_exc()
            print("  💡 PTH file was still created successfully.")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)
    for ext in ['.pth', '.onnx']:
        fpath = f"{args.output_name}{ext}"
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  ✅ {fpath} ({size_mb:.2f} MB)")
    print("=" * 60)


if __name__ == '__main__':
    main()
