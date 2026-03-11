"""
Snake Scan 偏移场可视化

用法:
  python visualize_offsets.py --checkpoint checkpoints/best.pth --image test.png
  python visualize_offsets.py --checkpoint checkpoints/best.pth --image test.png --stage 2
  python visualize_offsets.py --checkpoint checkpoints/best.pth --image test.png --model base
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train import build_model_and_criterion


def register_offset_hooks(
    model: nn.Module,
    target_stages: List[int],
) -> Tuple[Dict[str, List[Tensor]], list]:
    """Register forward hooks on SnakeScanBranch offset_net to capture offsets."""
    offsets: Dict[str, List[Tensor]] = {}
    handles = []

    # Unwrap DDP/DataParallel wrapper if present
    base_model = model.module if hasattr(model, "module") else model
    for stage_idx, stage in enumerate(base_model.encoder.stages):
        if stage_idx not in target_stages:
            continue
        for block_idx, block in enumerate(stage):
            for branch_name in ("snake_h", "snake_v", "snake_d45", "snake_d135"):
                if not hasattr(block, branch_name):
                    continue
                snake_branch = getattr(block, branch_name)
                key = f"stage{stage_idx}_block{block_idx}_{branch_name}"
                offsets[key] = []

                def make_hook(storage_list):
                    def hook_fn(module, input, output):
                        # output is (out_2d, offset)
                        if isinstance(output, tuple) and len(output) == 2:
                            storage_list.append(output[1].detach().cpu())
                    return hook_fn

                h = snake_branch.register_forward_hook(make_hook(offsets[key]))
                handles.append(h)

    return offsets, handles


def visualize_offset_field(
    offset: np.ndarray,
    title: str,
    image: np.ndarray = None,
    save_path: str = None,
):
    """Visualize offset field as quiver plot overlaid on image."""
    import matplotlib.pyplot as plt

    h, w = offset.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: heatmap of offset magnitude
    ax = axes[0]
    im = ax.imshow(np.abs(offset), cmap="hot", interpolation="nearest")
    ax.set_title(f"{title} — |offset|")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Right: quiver plot (subsample for readability)
    ax = axes[1]
    if image is not None:
        ax.imshow(image, alpha=0.5)

    step = max(1, min(h, w) // 32)
    y_coords = np.arange(0, h, step)
    x_coords = np.arange(0, w, step)
    Y, X = np.meshgrid(y_coords, x_coords, indexing="ij")
    off_sub = offset[::step, ::step]

    # For horizontal snake: arrows point vertically by offset amount
    if "snake_h" in title or "snake_d" in title:
        U = np.zeros_like(off_sub)
        V = off_sub
    else:
        U = off_sub
        V = np.zeros_like(off_sub)

    ax.quiver(X, Y, U, V, np.abs(off_sub), cmap="coolwarm", scale=50, alpha=0.8)
    ax.set_title(f"{title} — quiver")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Snake Scan Offsets")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", choices=["base", "sota"], default="sota")
    parser.add_argument("--stage", type=int, nargs="+", default=None,
                        help="Stages to visualize (default: all)")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--output_dir", default="offset_vis")
    parser.add_argument("--encoder_dims", type=int, nargs=4, default=[64, 128, 256, 512])
    parser.add_argument("--decoder_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--num_queries", type=int, default=240)
    args = parser.parse_args()

    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, _ = build_model_and_criterion(args)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Using EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # Load and preprocess image
    image = cv2.imread(args.image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (args.img_size, args.img_size))
    image_t = torch.from_numpy(
        image_resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0).to(device)

    # Determine target stages
    n_stages = len(args.encoder_dims)
    target_stages = args.stage if args.stage else list(range(n_stages))

    # Register hooks
    offsets, handles = register_offset_hooks(model, target_stages)

    # Forward pass
    with torch.no_grad():
        outputs = model(image_t)

    # Remove hooks
    for h in handles:
        h.remove()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each captured offset
    for key, offset_list in offsets.items():
        if not offset_list:
            continue
        # Take the first (and usually only) offset
        offset = offset_list[0][0, 0].numpy()  # (H, W)
        save_path = str(output_dir / f"{key}.png")
        visualize_offset_field(
            offset, key, image=image_resized, save_path=save_path
        )

    print(f"\nVisualized {len(offsets)} offset fields to {output_dir}/")


if __name__ == "__main__":
    main()
