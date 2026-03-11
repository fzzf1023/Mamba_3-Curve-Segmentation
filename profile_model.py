"""
模型 FLOPs / 参数量 / 推理速度分析

用法:
  python profile_model.py --model sota --img_size 512
  python profile_model.py --model base --img_size 512
  python profile_model.py --model sota --img_size 512 --device cpu
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from train import build_model_and_criterion


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters, plus per-submodule breakdown."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        breakdown[name] = n
    return {"total": total, "trainable": trainable, "breakdown": breakdown}


def measure_fps(model: nn.Module, input_tensor: torch.Tensor,
                warmup: int = 10, repeats: int = 50) -> float:
    """Measure inference FPS with warmup."""
    model.eval()
    device = input_tensor.device
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)
        if use_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            model(input_tensor)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return repeats / elapsed


def try_flops(model: nn.Module, input_tensor: torch.Tensor) -> str:
    """Try to compute FLOPs using fvcore or thop, fallback to N/A."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return f"{flops.total() / 1e9:.2f} G"
    except (ImportError, RuntimeError, Exception) as e:
        if isinstance(e, ImportError):
            pass
        else:
            print(f"  [fvcore] FLOPs estimation failed: {e}")
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(input_tensor,), verbose=False)
        return f"{flops / 1e9:.2f} G"
    except (ImportError, RuntimeError, Exception) as e:
        if isinstance(e, ImportError):
            pass
        else:
            print(f"  [thop] FLOPs estimation failed: {e}")
    return "N/A (install fvcore or thop)"


def main():
    parser = argparse.ArgumentParser(description="Model Profiling")
    parser.add_argument("--model", choices=["base", "sota"], default="sota")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--encoder_dims", type=int, nargs=4, default=[64, 128, 256, 512])
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--num_queries", type=int, default=240)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"{'='*60}")
    print(f"Model Profiling: {args.model} @ {args.img_size}x{args.img_size}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    model, _ = build_model_and_criterion(args)
    model = model.to(device).eval()

    # Parameter count
    info = count_parameters(model)
    print(f"\nTotal parameters:     {info['total'] / 1e6:.2f} M")
    print(f"Trainable parameters: {info['trainable'] / 1e6:.2f} M")
    print(f"\nPer-module breakdown:")
    for name, n in sorted(info["breakdown"].items(), key=lambda x: -x[1]):
        pct = 100.0 * n / max(info["total"], 1)
        print(f"  {name:30s} {n / 1e6:8.2f} M  ({pct:5.1f}%)")

    # FLOPs
    dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    print(f"\nFLOPs: {try_flops(model, dummy)}")

    # FPS
    if device.type == "cuda":
        fps = measure_fps(model, dummy, warmup=args.warmup, repeats=args.repeats)
        print(f"FPS (batch={args.batch_size}): {fps:.1f}")
        print(f"Latency: {1000.0 / fps:.1f} ms")
    else:
        fps = measure_fps(model, dummy, warmup=3, repeats=10)
        print(f"FPS (CPU, batch={args.batch_size}): {fps:.2f}")
        print(f"Latency: {1000.0 / fps:.0f} ms")

    # Memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(dummy)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak GPU memory: {peak_mb:.0f} MB")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
