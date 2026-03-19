"""
训练脚本 — CurveSOTAQueryNet（或 CurveInstanceMamba3Net 基础版）

用法示例：
  # 基础版（轻量，速度快）
  python train.py --model base --train_dir data/train --val_dir data/val

  # SOTA 版（更强，需更多显存）
  python train.py --model sota --train_dir data/train --val_dir data/val \
      --batch_size 4 --img_size 512 --epochs 100 --lr 2e-4

  # 消融：禁用图例引导 (Innovation A+E+LCAB+C)
  python train.py --model sota --train_dir data/train --no_legend_queries

  # 从断点继续
  python train.py --model sota --train_dir data/train --resume checkpoints/last.pth
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import build_dataloaders, targets_to_instance_list
from evaluate import evaluate_batch, aggregate_metrics
from mamba3_curve_instance_seg import (
    CurveInstanceLoss,
    CurveLossWeights,
    CurveInstanceMamba3Net,
    CurveSegConfig,
)
from curve_sota_query_seg import (
    CurveSOTACriterion,
    CurveSOTAConfig,
    CurveSOTAQueryNet,
    SOTALossWeights,
)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Standard in SOTA segmentation (Mask2Former, MaskDINO).
    Use ema_model for validation/inference after training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chart Curve Instance Segmentation Training")
    p.add_argument("--model",      choices=["base", "sota"], default="sota",
                   help="base=CurveInstanceMamba3Net, sota=CurveSOTAQueryNet")
    p.add_argument("--train_dir",  required=True,  help="训练数据目录")
    p.add_argument("--val_dir",    default=None,   help="验证数据目录（可选）")
    p.add_argument("--output_dir", default="checkpoints", help="检查点保存目录")

    # 训练超参
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--grad_clip",   type=float, default=1.0,
                   help="梯度裁剪最大范数（SSM 必须设置，建议 0.5-2.0）")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--stroke_width",type=int, default=3,
                   help="JSON 中未指定宽度时的默认描边宽度（像素）")

    # 模型配置
    p.add_argument("--preset", choices=["chart", "legacy"], default="chart",
                   help="chart=针对图表曲线任务的默认优化配置，legacy=原始更重配置")
    p.add_argument("--encoder_dims", type=int, nargs=4, default=None)
    p.add_argument("--decoder_dim",  type=int, default=None)
    p.add_argument("--embed_dim",    type=int, default=None)
    p.add_argument("--num_queries",  type=int, default=None,
                   help="SOTA 模型的 query 数量，建议 ≥ 图片中最大曲线数 × 4")
    p.add_argument("--style_head", action="store_true", default=False,
                   help="启用每实例 style 分类头（需要 instance_styles 标注）")
    p.add_argument("--num_styles", type=int, default=5,
                   help="style 头类别数，仅在 --style_head 时生效")
    p.add_argument("--style_loss_weight", type=float, default=0.5,
                   help="style 监督损失权重")
    p.add_argument("--layering_head", action="store_true", default=False,
                   help="启用 layering 预测头（需要有效 layering_target 标注）")
    p.add_argument("--layering_loss_weight", type=float, default=0.3,
                   help="layering 监督损失权重")
    p.add_argument("--efd_head", action="store_true", default=False,
                   help="启用 EFD 轮廓正则头（不需要额外标注，但会增加开销）")
    p.add_argument("--efd_loss_weight", type=float, default=0.05,
                   help="EFD 轮廓正则损失权重，仅在 --efd_head 时生效")
    legend_group = p.add_mutually_exclusive_group()
    legend_group.add_argument("--legend_queries", dest="legend_queries", action="store_true",
                              help="启用图例引导 Query（当数据提供 legend_patches 时才建议开启）")
    legend_group.add_argument("--no_legend_queries", dest="legend_queries", action="store_false",
                              help="[消融] 禁用图例引导 Query (Innovation A+E+LCAB+C)")
    p.set_defaults(legend_queries=None)

    # 其他
    p.add_argument("--resume",   default=None, help="从检查点文件恢复训练")
    p.add_argument("--amp",      action="store_true", default=True,
                   help="使用 AMP 混合精度（默认开启）")
    p.add_argument("--no_amp",   action="store_false", dest="amp")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--log_every", type=int, default=20, help="每 N 步打印一次日志")
    p.add_argument("--save_every", type=int, default=10, help="每 N epoch 保存一次检查点")
    p.add_argument("--multi_scale", action="store_true", default=False,
                   help="启用多尺度训练（每 epoch 随机选择分辨率）")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="梯度累积步数（等效 batch_size = batch_size × accum_steps）")

    # 训练质量开关
    p.add_argument("--ema_decay",        type=float, default=0.999,
                   help="EMA decay 系数（小数据集推荐 0.999，大数据集可用 0.9999）")
    p.add_argument("--loss_ramp_epochs", type=int,   default=10,
                   help="前 N epoch 内将 boundary/cape/pcc/snake_offset 从 0 线性增至全权重（0=不调度）")
    p.add_argument("--grad_checkpoint",  action="store_true", default=False,
                   help="启用梯度检查点（显著节省显存，训练速度略降）")

    # 消融实验开关 (Ablation flags)
    p.add_argument("--no_bato",             action="store_true", help="[消融] 禁用 BATO 模块")
    p.add_argument("--no_query_align",      action="store_true", help="[消融] 禁用 Top-k 查询对齐")
    p.add_argument("--no_position_relation",action="store_true", help="[消融] 禁用 Relation-DETR 位置关系")
    p.add_argument("--no_cape",             action="store_true", help="[消融] 禁用 CAPE 连通性损失")
    p.add_argument("--no_pcc",              action="store_true", help="[消融] 禁用 PCC 对比损失")
    p.add_argument("--no_snake_offset",     action="store_true", help="[消融] 禁用 Snake 偏移对齐损失")
    p.add_argument("--no_stem_skip",        action="store_true", help="[消融] 禁用 FPN H/2 stem skip")
    p.add_argument("--no_grid_suppression", action="store_true", help="[消融] 禁用 additive grid bias")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 模型 & 损失函数构建
# ---------------------------------------------------------------------------

def _resolve_backbone_cfg(args) -> CurveSegConfig:
    preset = getattr(args, "preset", "chart")
    grad_checkpoint = getattr(args, "grad_checkpoint", False)
    if preset == "legacy":
        backbone_cfg = CurveSegConfig.legacy_preset(
            in_channels=3,
            use_grad_checkpoint=grad_checkpoint,
        )
    else:
        backbone_cfg = CurveSegConfig.chart_preset(
            in_channels=3,
            use_grad_checkpoint=grad_checkpoint,
        )

    if getattr(args, "encoder_dims", None) is not None:
        backbone_cfg = replace(backbone_cfg, encoder_dims=tuple(args.encoder_dims))
    if getattr(args, "decoder_dim", None) is not None:
        backbone_cfg = replace(backbone_cfg, decoder_dim=int(args.decoder_dim))
    if getattr(args, "embed_dim", None) is not None:
        backbone_cfg = replace(backbone_cfg, embed_dim=int(args.embed_dim))
    backbone_cfg = replace(
        backbone_cfg,
        use_stem_skip=not getattr(args, "no_stem_skip", False),
        use_grid_suppression=not getattr(args, "no_grid_suppression", False),
    )
    return backbone_cfg


def _resolve_sota_cfg(args, backbone_cfg: CurveSegConfig) -> CurveSOTAConfig:
    preset = getattr(args, "preset", "chart")
    if preset == "legacy":
        sota_cfg = CurveSOTAConfig.legacy_preset(backbone=backbone_cfg)
    else:
        sota_cfg = CurveSOTAConfig.chart_preset(backbone=backbone_cfg)

    if getattr(args, "num_queries", None) is not None:
        sota_cfg = replace(sota_cfg, num_queries=int(args.num_queries))
    if getattr(args, "legend_queries", None) is not None:
        sota_cfg = replace(sota_cfg, use_legend_queries=bool(args.legend_queries))
    return replace(
        sota_cfg,
        use_bato=not getattr(args, "no_bato", False),
        use_query_align=not getattr(args, "no_query_align", False),
        use_position_relation=not getattr(args, "no_position_relation", False),
        use_style_head=bool(getattr(args, "style_head", False)),
        use_layering_head=bool(getattr(args, "layering_head", False)),
        use_efd_head=bool(getattr(args, "efd_head", False)),
        num_styles=int(getattr(args, "num_styles", sota_cfg.num_styles)),
    )


def build_model_and_criterion(args) -> Tuple[nn.Module, nn.Module]:
    backbone_cfg = _resolve_backbone_cfg(args)

    if args.model == "base":
        model     = CurveInstanceMamba3Net(backbone_cfg)
        # Apply base-model ablation flags
        base_weights = CurveLossWeights()
        if getattr(args, "no_snake_offset", False):
            base_weights.snake_offset = 0.0
        criterion = CurveInstanceLoss(embed_dim=backbone_cfg.embed_dim, weights=base_weights)
    else:
        sota_cfg = _resolve_sota_cfg(args, backbone_cfg)
        model = CurveSOTAQueryNet(sota_cfg)
        # Apply SOTA-model ablation flags via loss weights
        if getattr(args, "preset", "chart") == "legacy":
            sota_weights = SOTALossWeights.legacy_preset()
        else:
            sota_weights = SOTALossWeights.chart_preset()
        if getattr(args, "no_cape", False):
            sota_weights.cape = 0.0
        if getattr(args, "no_pcc", False):
            sota_weights.pcc = 0.0
        if getattr(args, "no_snake_offset", False):
            sota_weights.snake_offset = 0.0
        if not getattr(sota_cfg, "use_legend_queries", False):
            sota_weights.legend_contrastive = 0.0
        if getattr(sota_cfg, "use_style_head", False):
            sota_weights.style = float(getattr(args, "style_loss_weight", sota_weights.style))
        else:
            sota_weights.style = 0.0
        if getattr(sota_cfg, "use_layering_head", False):
            sota_weights.layering = float(getattr(args, "layering_loss_weight", sota_weights.layering))
        else:
            sota_weights.layering = 0.0
        if getattr(sota_cfg, "use_efd_head", False):
            sota_weights.efd = float(getattr(args, "efd_loss_weight", sota_weights.efd))
        else:
            sota_weights.efd = 0.0
        use_uncertainty_weighting = getattr(args, "preset", "chart") == "legacy"
        criterion = CurveSOTACriterion(
            weights=sota_weights,
            legend_contrastive_tau=sota_cfg.legend_contrastive_tau,
            use_uncertainty_weighting=use_uncertainty_weighting,
        )

    return model, criterion


# ---------------------------------------------------------------------------
# 损失调度（复杂损失项线性预热）
# ---------------------------------------------------------------------------

_RAMP_LOSS_KEYS = ("boundary", "cape", "pcc", "snake_offset")


def _save_original_weights(criterion: nn.Module) -> Dict[str, float]:
    """保存 criterion 中待调度的损失权重原始值。"""
    saved: Dict[str, float] = {}
    if hasattr(criterion, "weights"):
        for key in _RAMP_LOSS_KEYS:
            if hasattr(criterion.weights, key):
                value = float(getattr(criterion.weights, key))
                if value > 0.0:
                    saved[key] = value
    return saved


def _apply_loss_ramp(
    criterion: nn.Module,
    epoch: int,
    ramp_epochs: int,
    original_weights: Dict[str, float],
) -> None:
    """将复杂损失项从 0 线性增至原始权重，避免训练初期噪声梯度。"""
    if ramp_epochs <= 0 or not hasattr(criterion, "weights"):
        return
    factor = min(1.0, epoch / ramp_epochs)
    for key, orig_val in original_weights.items():
        if hasattr(criterion.weights, key):
            setattr(criterion.weights, key, orig_val * factor)


# ---------------------------------------------------------------------------
# 优化器 & 调度器
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(
    model: nn.Module,
    args,
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    # Mamba 参数（SSM 核心）使用较小学习率
    mamba_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ("row_mamba", "col_mamba", "snake_h", "snake_v", "snake_d45", "snake_d135")):
            mamba_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,  "lr": args.lr,        "weight_decay": args.weight_decay},
            {"params": mamba_params,  "lr": args.lr * 0.5,  "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # 余弦退火，最低 lr = 1e-6
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
        return max(1e-6 / args.lr, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# 单步前向
# ---------------------------------------------------------------------------

def forward_step(
    model: nn.Module,
    criterion: nn.Module,
    images: Tensor,
    targets: Dict[str, Tensor],
    device: torch.device,
    use_amp: bool,
    legend_patches: Optional[list] = None,
) -> Tensor:
    images = images.to(device, non_blocking=True)
    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

    # Move legend_patches tensors to device (per-item variable-length list)
    if legend_patches is not None:
        legend_patches = [
            p.to(device, non_blocking=True) if p is not None else None
            for p in legend_patches
        ]

    # autocast is device-aware: CUDA uses fp16, CPU uses bfloat16 (or no-op if disabled)
    amp_ctx = torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda"))
    with amp_ctx:
        if isinstance(model, CurveSOTAQueryNet):
            # SOTA 模型：将 targets 转换为 instance_targets 以启用 DN queries
            if model.training:
                instance_targets = targets_to_instance_list(targets, device)
            else:
                instance_targets = None
            outputs = model(images, instance_targets=instance_targets,
                            legend_patches=legend_patches)
            losses = criterion(outputs, targets)
        else:
            # Base 模型
            outputs = model(images)
            losses = criterion(outputs, targets)

    return losses["total"]


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    count = 0
    all_metrics: list = []
    for images, targets in val_loader:
        images_d = images.to(device, non_blocking=True)
        # Extract legend_patches before moving targets to device (they're per-image lists)
        legend_patches = targets.pop("legend_patches", None)
        targets_d = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        if legend_patches is not None:
            legend_patches = [
                p.to(device, non_blocking=True) if p is not None else None
                for p in legend_patches
            ]

        # Single forward pass: compute outputs once, derive both loss and metrics
        amp_ctx = torch.amp.autocast(device.type, enabled=(use_amp and device.type == "cuda"))
        with amp_ctx:
            if isinstance(model, CurveSOTAQueryNet):
                outputs = model(images_d, instance_targets=None,
                                legend_patches=legend_patches)
            else:
                outputs = model(images_d)
            losses = criterion(outputs, targets_d)

        total_loss += losses["total"].item()
        count += 1

        batch_metrics = evaluate_batch(outputs, targets_d, device)
        all_metrics.append(batch_metrics)

    # restore original training state (EMA shadow stays eval, training model goes back)
    if was_training:
        model.train()
    result = {"val_loss": total_loss / max(1, count)}
    result.update(aggregate_metrics(all_metrics))
    return result


# ---------------------------------------------------------------------------
# 检查点
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_loss: float,
    ema: Optional[ModelEMA] = None,
) -> None:
    ckpt = {
        "epoch":          epoch,
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "scaler":         scaler.state_dict(),
        "best_val_loss":  best_val_loss,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: torch.device = torch.device("cpu"),
    ema: Optional[ModelEMA] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    epoch = ckpt.get("epoch", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[Resume] 已从 epoch {epoch} 恢复，best_val_loss={best_val_loss:.4f}")
    return epoch, best_val_loss


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args) -> None:
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] 设备: {device}  AMP: {args.amp}")

    # ---- 数据 ----
    train_loader, val_loader = build_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stroke_width=args.stroke_width,
        multi_scale=args.multi_scale,
    )
    steps_per_epoch = len(train_loader)
    print(f"[Train] 训练集: {len(train_loader.dataset)} 张  "
          f"每 epoch {steps_per_epoch} 步")

    # ---- 模型 ----
    model, criterion = build_model_and_criterion(args)
    model = model.to(device)
    criterion = criterion.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.model}  可训练参数: {n_params / 1e6:.1f}M")

    # ---- 优化器 ----
    optimizer, scheduler = build_optimizer_scheduler(model, args, steps_per_epoch)
    # AMP is only meaningful on CUDA; CPU/MPS require disabling GradScaler
    _amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)

    # ---- EMA ----
    ema_decay = getattr(args, "ema_decay", 0.999)
    ema = ModelEMA(model, decay=ema_decay)
    print(f"[EMA] 已创建 EMA shadow model (decay={ema_decay})")

    # ---- 断点恢复 ----
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device, ema=ema
        )
        start_epoch += 1   # 从下一个 epoch 开始

    # ---- 输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 损失调度：保存复杂损失原始权重 ----
    _orig_complex_weights = _save_original_weights(criterion)
    if _orig_complex_weights and args.loss_ramp_epochs > 0:
        keys_str = ", ".join(_orig_complex_weights.keys())
        print(f"[LossRamp] 将在前 {args.loss_ramp_epochs} epoch 内线性引入: {keys_str}")

    # ---- 训练 ----
    model.train()
    for epoch in range(start_epoch, args.epochs):
        # 多尺度训练：每 epoch 设置随机分辨率
        if hasattr(train_loader.dataset, "set_epoch_scale"):
            train_loader.dataset.set_epoch_scale(epoch)

        # 损失调度：复杂项线性预热
        _apply_loss_ramp(criterion, epoch, args.loss_ramp_epochs, _orig_complex_weights)

        t0 = time.time()
        epoch_loss = 0.0
        optimizer.zero_grad()
        accum_steps = getattr(args, "accum_steps", 1)

        for step, (images, targets) in enumerate(train_loader):
            global_step = epoch * steps_per_epoch + step

            # Extract legend_patches before moving targets to device (per-image lists)
            legend_patches = targets.pop("legend_patches", None)

            # 前向 + AMP
            loss = forward_step(model, criterion, images, targets, device, _amp_enabled,
                                legend_patches=legend_patches)
            loss = loss / accum_steps  # normalize for gradient accumulation

            # 反向
            scaler.scale(loss).backward()

            # Gradient accumulation: only step every accum_steps
            if (step + 1) % accum_steps == 0 or (step + 1) == steps_per_epoch:
                # 梯度裁剪（SSM 的梯度容易爆炸，必须设置）
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # EMA 更新
                ema.update(model)

            epoch_loss += loss.item() * accum_steps  # undo normalization for logging

            if step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Epoch [{epoch+1}/{args.epochs}]  "
                      f"Step [{step+1}/{steps_per_epoch}]  "
                      f"Loss: {loss.item() * accum_steps:.4f}  "
                      f"LR: {lr_now:.2e}")

        avg_loss = epoch_loss / steps_per_epoch
        elapsed = time.time() - t0
        print(f"Epoch [{epoch+1}/{args.epochs}]  "
              f"avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # ---- 验证（使用 EMA shadow model）----
        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = validate(ema.shadow, criterion, val_loader, device, _amp_enabled)
            metric_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"  [Val-EMA] {metric_str}")

            val_loss = val_metrics["val_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    str(output_dir / "best.pth"),
                    model, optimizer, scheduler, scaler, epoch, best_val_loss,
                    ema=ema,
                )
                print(f"  [Best] 新最佳模型已保存 (val_loss={best_val_loss:.4f})")

        # ---- 定期保存 ----
        save_checkpoint(
            str(output_dir / "last.pth"),
            model, optimizer, scheduler, scaler, epoch, best_val_loss,
            ema=ema,
        )
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                str(output_dir / f"epoch_{epoch+1:04d}.pth"),
                model, optimizer, scheduler, scaler, epoch, best_val_loss,
                ema=ema,
            )

    print(f"\n训练完成！best_val_loss={best_val_loss:.4f}")
    print(f"最佳模型: {output_dir / 'best.pth'}")


# ---------------------------------------------------------------------------
# 推理示例（训练完成后使用）
# ---------------------------------------------------------------------------

def infer_single(
    image_path: str,
    checkpoint_path: str,
    model_type: str = "sota",
    img_size: int = 512,
    score_thresh: float = 0.4,
) -> None:
    """对单张图片做推理并保存可视化结果。"""
    import cv2 as cv2_vis
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    args_dummy = argparse.Namespace(
        model=model_type,
        preset="chart",
        encoder_dims=None,
        decoder_dim=None,
        embed_dim=None,
        num_queries=None,
        legend_queries=None,
    )
    model, _ = build_model_and_criterion(args_dummy)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # prefer EMA weights if available
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("[Infer] 使用 EMA 权重")
    else:
        model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    # 预处理
    image = cv2_vis.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    image_rgb = cv2_vis.cvtColor(image, cv2_vis.COLOR_BGR2RGB)
    image_resized = cv2_vis.resize(image_rgb, (img_size, img_size))
    image_t = torch.from_numpy(
        image_resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    ).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(image_t)

    # 取 composed_mask（base）或 pred_masks（sota）
    if "composed_mask" in outputs:
        mask = outputs["composed_mask"][0, 0].cpu().numpy()
    else:
        scores = outputs["pred_logits"][0].softmax(-1)[:, 1]
        pred_masks = torch.sigmoid(outputs["pred_masks"][0])
        top_idx = (scores > score_thresh).nonzero(as_tuple=True)[0]
        if top_idx.numel() == 0:
            print("未检测到曲线")
            return
        mask = pred_masks[top_idx].max(0).values.cpu().numpy()

    # 可视化
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_resized = cv2_vis.resize(mask_u8, (orig_w, orig_h))
    overlay = image.copy()
    overlay[mask_resized > 128] = [0, 255, 0]
    out_path = Path(image_path).stem + "_pred.jpg"
    cv2_vis.imwrite(out_path, overlay)
    print(f"结果已保存到 {out_path}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
