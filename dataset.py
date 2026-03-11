"""
CurvePolylineDataset — 将 polyline JSON 标注转换为模型所需的全部训练目标张量。

支持的 JSON 格式（自动识别）：
  · LabelMe  (shape_type: "linestrip" / "line")
  · 自定义简单格式  {"curves": [{"points": [[x,y],...], "label": "curve"}]}

目录结构（任选其一）：
  images/  *.jpg / *.png
  labels/  *.json           （与图片同名）

  或者 图片与 JSON 放在同一目录，同名即可。

目标张量说明（与 CurveInstanceLoss / CurveSOTACriterion 完全对应）：
  curve_mask       (H, W) float32 [0,1]   — 所有曲线的扩张 mask（有宽度）
  centerline_mask  (H, W) float32 [0,1]   — 所有曲线的中心线（1px）
  instance_ids     (H, W) int64           — 每像素所属的实例编号，0=背景
  direction_vectors(4, H, W) float32      — 主切线(ch0-1) + 交叉次切线(ch2-3)
  crossing_mask    (H, W) float32 [0,1]   — 两条及以上曲线重叠的像素
  layering_target  (H, W) float32         — 上层曲线=1，下层=0，无标注=-1
  grid_mask        (H, W) float32 [0,1]   — 网格背景区域（若有标注）
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _find_pairs(root: str) -> List[Tuple[Path, Path]]:
    """
    在 root 目录（含子目录）中寻找 (image_path, json_path) 配对。
    图片扩展名: jpg, jpeg, png, bmp, tiff
    JSON 文件与图片同名（扩展名不同）。
    """
    root = Path(root)
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            # 尝试 labels/ 子目录
            json_path = img_path.parent.parent / "labels" / img_path.with_suffix(".json").name
        if json_path.exists():
            pairs.append((img_path, json_path))
    return pairs


def _parse_json(json_path: Path) -> List[Dict]:
    """
    解析标注 JSON，返回 curve 列表，每项格式：
      {"points": [[x,y],...], "label": str, "width": int (optional)}

    自动兼容：
      · LabelMe (shapes 列表, shape_type in {"linestrip","line","polyline"})
      · 自定义  (curves 列表, 或 annotations 列表)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    curves: List[Dict] = []

    # ---- LabelMe 格式 ----
    if "shapes" in data:
        for shape in data["shapes"]:
            stype = shape.get("shape_type", "")
            if stype not in ("linestrip", "line", "polyline", "lines"):
                continue
            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            label = shape.get("label", "curve")
            # 支持 label 中携带宽度信息，例如 "curve_w5" 表示宽度 5
            width = _parse_width_from_label(label)
            curves.append({"points": pts, "label": label, "width": width})

    # ---- 自定义格式 ----
    elif "curves" in data:
        for item in data["curves"]:
            pts = item.get("points", [])
            if len(pts) < 2:
                continue
            label = item.get("label", "curve")
            width = item.get("width", None) or _parse_width_from_label(label)
            curves.append({"points": pts, "label": label, "width": width})

    elif "annotations" in data:
        for ann in data["annotations"]:
            pts = ann.get("points", ann.get("segmentation", []))
            # 展平成 [[x,y],...] 如果是 COCO 格式的扁平列表
            if pts and not isinstance(pts[0], (list, tuple)):
                pts = [[pts[i], pts[i + 1]] for i in range(0, len(pts) - 1, 2)]
            if len(pts) < 2:
                continue
            label = ann.get("label", ann.get("category", "curve"))
            width = ann.get("width", None) or _parse_width_from_label(str(label))
            curves.append({"points": pts, "label": str(label), "width": width})

    return curves


def _parse_width_from_label(label: str) -> Optional[int]:
    """从 label 字符串提取宽度，例如 'curve_w5' → 5，否则返回 None。"""
    label = label.lower()
    for part in label.replace("-", "_").split("_"):
        if part.startswith("w") and part[1:].isdigit():
            return int(part[1:])
    return None


def _rasterize_polylines(
    curves: List[Dict],
    canvas_h: int,
    canvas_w: int,
    default_stroke_width: int = 3,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    将 polyline 列表光栅化为所有训练目标 numpy 数组。

    返回 dict，key 与 Dataset.__getitem__ 返回的 targets 对应：
      curve_mask, centerline_mask, instance_ids,
      direction_vectors (4, H, W), crossing_mask,
      layering_target, grid_mask (全 -1，不含网格标注)
    """
    H, W = canvas_h, canvas_w
    curve_mask     = np.zeros((H, W), dtype=np.float32)
    centerline     = np.zeros((H, W), dtype=np.float32)
    instance_ids   = np.zeros((H, W), dtype=np.int64)
    overlap_count  = np.zeros((H, W), dtype=np.uint8)     # 计数重叠

    # 为精确计算交叉处次方向，先独立存储每条曲线的方向场
    per_curve_dir_x: List[np.ndarray] = []
    per_curve_dir_y: List[np.ndarray] = []
    per_curve_cl_mask: List[np.ndarray] = []

    for inst_id, curve in enumerate(curves, start=1):
        pts_raw = curve["points"]
        stroke_w = curve.get("width") or default_stroke_width

        # 坐标缩放（当图像被 resize 时同步变换点坐标）
        pts = np.array([[p[0] * scale_x, p[1] * scale_y] for p in pts_raw],
                       dtype=np.float32)
        pts_int = np.round(pts).astype(np.int32)

        # ---- 中心线 & 实例 ID ----
        inst_mask_cl = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(inst_mask_cl, [pts_int], False, 1, thickness=1)
        centerline = np.clip(centerline + inst_mask_cl.astype(np.float32), 0, 1)
        # instance_ids: 后绘制的实例覆盖前者（对于不重叠区域保留各自 ID）
        instance_ids[inst_mask_cl > 0] = inst_id

        # ---- 扩张 curve mask ----
        inst_mask_fat = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(inst_mask_fat, [pts_int], False, 1, thickness=stroke_w)
        curve_mask = np.clip(curve_mask + inst_mask_fat.astype(np.float32), 0, 1)
        # 宽 mask 的实例 ID（中心线已覆盖，宽 mask 补充空洞）
        instance_ids[(inst_mask_fat > 0) & (instance_ids == 0)] = inst_id
        overlap_count += inst_mask_fat

        # ---- 每条曲线独立的方向向量 ----
        dx_i = np.zeros((H, W), dtype=np.float32)
        dy_i = np.zeros((H, W), dtype=np.float32)
        _fill_direction(pts_int, inst_mask_cl, dx_i, dy_i, H, W)
        _normalize_dir_field(dx_i, dy_i)
        per_curve_dir_x.append(dx_i)
        per_curve_dir_y.append(dy_i)
        per_curve_cl_mask.append(inst_mask_cl)

    # ---- 重叠像素（交叉 mask）----
    crossing_mask = (overlap_count >= 2).astype(np.float32)

    # ---- 构建主方向 & 次方向场 ----
    # 主方向：最后绘制的曲线方向（与 instance_ids 一致）
    # 次方向：交叉区域中，非主方向的另一条曲线的切线方向
    dir_x_pri = np.zeros((H, W), dtype=np.float32)
    dir_y_pri = np.zeros((H, W), dtype=np.float32)
    dir_x_sec = np.zeros((H, W), dtype=np.float32)
    dir_y_sec = np.zeros((H, W), dtype=np.float32)

    # 逐曲线叠加：后绘制的方向覆盖主方向，被覆盖的方向变为次方向
    for i in range(len(curves)):
        mask_i = per_curve_cl_mask[i] > 0
        # 在此曲线的中心线像素上：当前主方向降为次方向，新方向成为主方向
        has_existing = mask_i & ((dir_x_pri != 0) | (dir_y_pri != 0))
        dir_x_sec[has_existing] = dir_x_pri[has_existing]
        dir_y_sec[has_existing] = dir_y_pri[has_existing]
        # 写入新的主方向
        dir_x_pri[mask_i] = per_curve_dir_x[i][mask_i]
        dir_y_pri[mask_i] = per_curve_dir_y[i][mask_i]

    direction_vectors = np.stack([dir_x_pri, dir_y_pri, dir_x_sec, dir_y_sec], axis=0)

    # ---- 实例边界 mask ----
    # Boundary: pixels where adjacent instance IDs differ (both non-background)
    boundary_mask = np.zeros((H, W), dtype=np.float32)
    ids = instance_ids
    # horizontal neighbors
    h_diff = (ids[:, 1:] != ids[:, :-1]) & (ids[:, 1:] > 0) & (ids[:, :-1] > 0)
    # vertical neighbors
    v_diff = (ids[1:, :] != ids[:-1, :]) & (ids[1:, :] > 0) & (ids[:-1, :] > 0)
    boundary_mask[:, 1:]  += h_diff.astype(np.float32)
    boundary_mask[:, :-1] += h_diff.astype(np.float32)
    boundary_mask[1:, :]  += v_diff.astype(np.float32)
    boundary_mask[:-1, :] += v_diff.astype(np.float32)
    boundary_mask = (boundary_mask > 0).astype(np.float32)

    return {
        "curve_mask":       curve_mask,
        "centerline_mask":  centerline,
        "instance_ids":     instance_ids,
        "direction_vectors": direction_vectors,
        "crossing_mask":    crossing_mask,
        "boundary_mask":    boundary_mask,
        "layering_target":  np.full((H, W), -1.0, dtype=np.float32),
        "grid_mask":        np.zeros((H, W), dtype=np.float32),
    }


def _fill_direction(
    pts: np.ndarray,
    mask: np.ndarray,
    dir_x: np.ndarray,
    dir_y: np.ndarray,
    H: int,
    W: int,
) -> None:
    """
    对 polyline 中每条线段，用中央差分计算切线方向并写入 dir_x/dir_y。
    """
    n = len(pts)
    for i in range(n):
        # 中央差分：使用前后点
        prev_p = pts[max(0, i - 1)]
        next_p = pts[min(n - 1, i + 1)]
        dx = float(next_p[0] - prev_p[0])
        dy = float(next_p[1] - prev_p[1])
        length = math.sqrt(dx * dx + dy * dy) + 1e-6
        dx_n, dy_n = dx / length, dy / length

        if i < n - 1:
            # 在当前线段上的所有像素填入切线方向
            seg_mask = np.zeros((H, W), dtype=np.uint8)
            p0 = (int(pts[i][0]),     int(pts[i][1]))
            p1 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
            cv2.line(seg_mask, p0, p1, 1, thickness=1)
            seg_bool = (seg_mask > 0) & (mask > 0)
            dir_x[seg_bool] = dx_n
            dir_y[seg_bool] = dy_n


def _normalize_dir_field(dir_x: np.ndarray, dir_y: np.ndarray) -> None:
    """In-place 归一化方向场（避免 /0）。"""
    norms = np.sqrt(dir_x ** 2 + dir_y ** 2) + 1e-6
    nonzero = norms > 1e-5
    dir_x[nonzero] /= norms[nonzero]
    dir_y[nonzero] /= norms[nonzero]


# ---------------------------------------------------------------------------
# 数据增强（同时作用于图像和所有 mask）
# ---------------------------------------------------------------------------

class SyncTransform:
    """
    同步增强：图像与所有 mask 使用完全相同的几何变换。
    支持：水平翻转、随机缩放裁剪、颜色抖动（仅图像）。

    使用 albumentations（推荐）或降级到手动实现。
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        flip_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        color_jitter: bool = True,
        use_albumentations: bool = True,
    ):
        self.img_size = img_size
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.color_jitter = color_jitter

        self.albu = None
        if use_albumentations:
            try:
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                # 使用 ReplayCompose 以便对 instance_ids / direction / layering
                # 施加完全相同的几何变换（通过 replay 重放随机参数）
                self.albu_geo = A.ReplayCompose([
                    A.HorizontalFlip(p=flip_prob),
                    A.RandomResizedCrop(
                        height=img_size[0], width=img_size[1],
                        scale=scale_range, ratio=(0.9, 1.1),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
                ], additional_targets={
                    "curve_mask":       "mask",
                    "centerline_mask":  "mask",
                    "crossing_mask":    "mask",
                    "grid_mask":        "mask",
                    "instance_ids":     "mask",   # NEAREST interpolation for integer labels
                    "layering_target":  "mask",   # NEAREST interpolation for integer labels
                    "boundary_mask":    "mask",   # NEAREST interpolation
                })
                self.albu_color = A.Compose([
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
                    A.GaussNoise(var_limit=(5, 25), p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                ]) if color_jitter else None
                self.albu = True
            except ImportError:
                pass  # 退回到手动实现

    def __call__(
        self,
        image: np.ndarray,
        targets: Dict[str, np.ndarray],
        override_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        H, W = override_size if override_size is not None else self.img_size

        if self.albu is not None:
            return self._albu_transform(image, targets, H, W)
        else:
            return self._manual_transform(image, targets, H, W)

    def _albu_transform(self, image, targets, H, W):
        import albumentations as A

        # 1) 对 image + 所有 mask 施加几何变换（albumentations 对 mask 类型
        #    自动使用 NEAREST 插值，对 image 类型使用 LINEAR）
        #    instance_ids 需要转为 uint8/uint16 让 albumentations 识别为整型
        ids_original = targets["instance_ids"]
        layer_original = targets["layering_target"]

        aug_input = {
            "image":           image,
            "curve_mask":      targets["curve_mask"],
            "centerline_mask": targets["centerline_mask"],
            "crossing_mask":   targets["crossing_mask"],
            "grid_mask":       targets["grid_mask"],
            "instance_ids":    ids_original.astype(np.int32),
            "layering_target": layer_original,
            "boundary_mask":   targets["boundary_mask"],
        }
        aug_out = self.albu_geo(**aug_input)
        replay_data = aug_out["replay"]

        image = aug_out["image"]
        targets = dict(targets)  # shallow copy
        targets["curve_mask"]      = aug_out["curve_mask"]
        targets["centerline_mask"] = aug_out["centerline_mask"]
        targets["crossing_mask"]   = aug_out["crossing_mask"]
        targets["grid_mask"]       = aug_out["grid_mask"]
        targets["instance_ids"]    = aug_out["instance_ids"].astype(np.int64)
        targets["layering_target"] = aug_out["layering_target"]
        targets["boundary_mask"]   = aug_out["boundary_mask"]

        # 2) 用 replay 对 direction_vectors 逐通道施加同样变换
        dir_v = targets["direction_vectors"]  # (4, H_orig, W_orig)
        dir_channels = []
        for c in range(4):
            ch_aug = A.ReplayCompose.replay(replay_data, image=dir_v[c])["image"]
            dir_channels.append(ch_aug)
        dir_transformed = np.stack(dir_channels, axis=0)

        # 检查 replay 中是否包含几何变换，若有则修正方向向量
        for t in replay_data.get("transforms", []):
            cls_name = t.get("__class_fullname__", "")
            if "HorizontalFlip" in cls_name and t.get("applied", False):
                dir_transformed[0] = -dir_transformed[0]  # 主方向 x
                dir_transformed[2] = -dir_transformed[2]  # 次方向 x
            if "VerticalFlip" in cls_name and t.get("applied", False):
                dir_transformed[1] = -dir_transformed[1]  # 主方向 y
                dir_transformed[3] = -dir_transformed[3]  # 次方向 y
            if "Rotate" in cls_name and t.get("applied", False):
                # albumentations Rotate: positive angle = CCW rotation in screen coords (y-down).
                # For a direction vector [dx, dy] in image space (y-down), applying CCW rotation θ:
                #   dx' =  dx·cos(θ) + dy·sin(θ)
                #   dy' = -dx·sin(θ) + dy·cos(θ)
                angle_deg = t.get("params", {}).get("angle", 0.0)
                theta = math.radians(angle_deg)
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                dx0, dy0 = dir_transformed[0], dir_transformed[1]
                dir_transformed[0] =  cos_t * dx0 + sin_t * dy0
                dir_transformed[1] = -sin_t * dx0 + cos_t * dy0
                dx2, dy2 = dir_transformed[2], dir_transformed[3]
                dir_transformed[2] =  cos_t * dx2 + sin_t * dy2
                dir_transformed[3] = -sin_t * dx2 + cos_t * dy2
        targets["direction_vectors"] = dir_transformed

        # 3) 颜色增强（仅作用于 image）
        if self.albu_color is not None:
            image = self.albu_color(image=image)["image"]

        # 4) 多尺度 resize：若目标尺寸与 RandomResizedCrop 输出不同，额外 resize
        cur_h, cur_w = image.shape[:2]
        if (cur_h, cur_w) != (H, W):
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            for k in ("curve_mask", "centerline_mask"):
                targets[k] = cv2.resize(targets[k], (W, H), interpolation=cv2.INTER_LINEAR)
            for k in ("crossing_mask", "grid_mask", "boundary_mask"):
                targets[k] = cv2.resize(targets[k], (W, H), interpolation=cv2.INTER_NEAREST)
            targets["instance_ids"] = cv2.resize(
                targets["instance_ids"].astype(np.float32), (W, H),
                interpolation=cv2.INTER_NEAREST).astype(np.int64)
            targets["direction_vectors"] = np.stack([
                cv2.resize(targets["direction_vectors"][c], (W, H),
                           interpolation=cv2.INTER_LINEAR)
                for c in range(4)
            ], axis=0)
            targets["layering_target"] = cv2.resize(
                targets["layering_target"], (W, H), interpolation=cv2.INTER_NEAREST)

        return image, targets

    def _manual_transform(self, image, targets, H, W):
        """不依赖 albumentations 的简化增强。"""
        # Resize image
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)

        _nearest_keys = {"instance_ids", "layering_target", "crossing_mask",
                         "boundary_mask", "grid_mask"}
        resized = {}
        for k, v in targets.items():
            if k == "instance_ids":
                resized[k] = cv2.resize(v.astype(np.float32), (W, H),
                                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
            elif k == "direction_vectors":
                resized[k] = np.stack([
                    cv2.resize(v[c], (W, H), interpolation=cv2.INTER_LINEAR)
                    for c in range(4)
                ], axis=0)
            elif k in _nearest_keys:
                resized[k] = cv2.resize(v, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                resized[k] = cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR)

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = image[:, ::-1].copy()
            for k, v in resized.items():
                if k == "direction_vectors":
                    flipped = v[:, :, ::-1].copy()
                    flipped[0] = -flipped[0]   # x 分量反向
                    flipped[2] = -flipped[2]
                    resized[k] = flipped
                else:
                    resized[k] = v[:, ::-1].copy() if v.ndim == 2 else v[:, :, ::-1].copy()

        # 简单颜色抖动
        if self.color_jitter:
            image = _simple_color_jitter(image)

        return image, resized


def _simple_color_jitter(image: np.ndarray) -> np.ndarray:
    """简单亮度+对比度抖动，不依赖外部库。"""
    alpha = random.uniform(0.8, 1.2)   # 对比度
    beta  = random.uniform(-20, 20)    # 亮度
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CurvePolylineDataset(Dataset):
    """
    从 (image, json) 对构建训练/验证数据集。

    Args:
        data_dir:         根目录，包含图片和 JSON 文件
        img_size:         (H, W) 统一缩放尺寸，默认 (512, 512)
        stroke_width:     默认描边宽度（像素），label 中有 _wN 则覆盖
        augment:          是否开启数据增强（训练时 True，验证时 False）
        use_albumentations: 是否尝试使用 albumentations（推荐安装）
        max_instances:    单张图片最多保留的实例数量（按面积降序截取）
        multi_scale:      是否启用多尺度训练（训练时随机选择分辨率）
        scale_choices:    多尺度分辨率选项列表，如 [384, 448, 512, 576, 640]
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (512, 512),
        stroke_width: int = 3,
        augment: bool = True,
        use_albumentations: bool = True,
        max_instances: int = 50,
        multi_scale: bool = False,
        scale_choices: Optional[List[int]] = None,
    ):
        self.img_size = img_size   # (H, W)
        self.stroke_width = stroke_width
        self.augment = augment
        self.max_instances = max_instances
        self.multi_scale = multi_scale and augment  # only during training
        self.scale_choices = scale_choices or [384, 448, 512, 576, 640]
        self._current_scale: Optional[int] = None  # set per-epoch for batch uniformity

        self.pairs = _find_pairs(data_dir)
        if not self.pairs:
            raise FileNotFoundError(
                f"在 {data_dir} 中未找到任何 (image, json) 配对。\n"
                "请确认图片与 JSON 同名，或检查 labels/ 子目录。"
            )

        self.transform = SyncTransform(
            img_size=img_size,
            use_albumentations=use_albumentations,
        ) if augment else None

        print(f"[Dataset] 找到 {len(self.pairs)} 对标注文件，img_size={img_size}")

    def set_epoch_scale(self, epoch: int) -> None:
        """Set per-epoch random scale for multi-scale training.
        Call at the start of each epoch. All samples in the epoch use the same size
        so that batches can be collated without padding."""
        if self.multi_scale:
            rng = random.Random(epoch)
            self._current_scale = rng.choice(self.scale_choices)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        img_path, json_path = self.pairs[idx]

        # ---- 读取图像 ----
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # ---- 读取标注 ----
        try:
            curves = _parse_json(json_path)
        except Exception as e:
            raise ValueError(f"解析 JSON 失败: {json_path}\n{e}")

        if not curves:
            # 无标注时返回全背景样本（损失可跳过）
            curves = []

        # 截断实例数量
        if len(curves) > self.max_instances:
            curves = curves[: self.max_instances]

        # ---- 多尺度训练：使用 per-epoch 随机分辨率 ----
        if self.multi_scale and self._current_scale is not None:
            target_h, target_w = self._current_scale, self._current_scale
        else:
            target_h, target_w = self.img_size

        # ---- 坐标缩放比例（先光栅化到原图尺寸，再统一 resize）----
        scale_x = 1.0  # 先用原尺寸光栅化
        scale_y = 1.0

        targets_np = _rasterize_polylines(
            curves, orig_h, orig_w, self.stroke_width, scale_x, scale_y
        )

        # ---- 数据增强 / Resize ----
        if self.transform is not None:
            ms_override = (target_h, target_w) if (target_h, target_w) != self.img_size else None
            image, targets_np = self.transform(image, targets_np, override_size=ms_override)
        else:
            # 验证集仅做 resize
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            targets_np["curve_mask"] = cv2.resize(
                targets_np["curve_mask"], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            targets_np["centerline_mask"] = cv2.resize(
                targets_np["centerline_mask"], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            targets_np["crossing_mask"] = cv2.resize(
                targets_np["crossing_mask"], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            targets_np["boundary_mask"] = cv2.resize(
                targets_np["boundary_mask"], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            targets_np["grid_mask"] = cv2.resize(
                targets_np["grid_mask"], (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            targets_np["instance_ids"] = cv2.resize(
                targets_np["instance_ids"].astype(np.float32),
                (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            targets_np["direction_vectors"] = np.stack([
                cv2.resize(targets_np["direction_vectors"][c],
                           (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                for c in range(4)
            ], axis=0)
            targets_np["layering_target"] = cv2.resize(
                targets_np["layering_target"], (target_w, target_h),
                interpolation=cv2.INTER_NEAREST)

        # ---- 图像 → Tensor  (C, H, W)  float32  [0,1] ----
        image_t = torch.from_numpy(
            image.astype(np.float32).transpose(2, 0, 1) / 255.0
        )

        # ---- Targets → Tensor ----
        targets_t: Dict[str, Tensor] = {
            "curve_mask":       torch.from_numpy(targets_np["curve_mask"]).float(),
            "centerline_mask":  torch.from_numpy(targets_np["centerline_mask"]).float(),
            "instance_ids":     torch.from_numpy(targets_np["instance_ids"]).long(),
            "direction_vectors": torch.from_numpy(targets_np["direction_vectors"]).float(),
            "crossing_mask":    torch.from_numpy(targets_np["crossing_mask"]).float(),
            "boundary_mask":    torch.from_numpy(targets_np["boundary_mask"]).float(),
            "layering_target":  torch.from_numpy(targets_np["layering_target"]).float(),
            "grid_mask":        torch.from_numpy(targets_np["grid_mask"]).float(),
        }

        return image_t, targets_t


# ---------------------------------------------------------------------------
# 用于 CurveSOTACriterion 的 instance targets（query-based 模型使用）
# ---------------------------------------------------------------------------

def targets_to_instance_list(
    targets: Dict[str, Tensor],
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    """
    将 batch targets dict 拆分成 CurveSOTACriterion 所需的
    List[{"masks": (N,H,W), "labels": (N,)}] 格式。

    在 Criterion.forward 中直接传入 targets dict 也可以，
    因为 _dict_to_instance_targets 会自动完成此转换。
    此函数供需要手动控制的场景使用。
    """
    batch_size = targets["instance_ids"].shape[0]
    ids = targets["instance_ids"].to(device)
    out: List[Dict[str, Tensor]] = []
    for b in range(batch_size):
        uids = torch.unique(ids[b])
        uids = uids[uids > 0]
        if uids.numel() == 0:
            h, w = ids.shape[-2:]
            out.append({
                "masks":  torch.zeros(0, h, w, device=device),
                "labels": torch.zeros(0, dtype=torch.long, device=device),
            })
        else:
            masks = torch.stack([(ids[b] == uid).float() for uid in uids], dim=0)
            out.append({
                "masks":  masks,
                "labels": torch.ones(uids.numel(), dtype=torch.long, device=device),
            })
    return out


# ---------------------------------------------------------------------------
# DataLoader 工厂函数
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    img_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_workers: int = 4,
    stroke_width: int = 3,
    pin_memory: bool = True,
    multi_scale: bool = False,
):
    """
    返回 (train_loader, val_loader)，val_loader 在无 val_dir 时为 None。
    """
    from torch.utils.data import DataLoader

    train_ds = CurvePolylineDataset(
        train_dir, img_size=img_size, stroke_width=stroke_width, augment=True,
        multi_scale=multi_scale,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = None
    if val_dir and Path(val_dir).exists():
        val_ds = CurvePolylineDataset(
            val_dir, img_size=img_size, stroke_width=stroke_width, augment=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 快速调试：可视化一个 batch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    ds = CurvePolylineDataset(data_dir, img_size=(512, 512), stroke_width=3, augment=False)
    print(f"数据集大小: {len(ds)}")

    image, targets = ds[0]
    print("image shape:", tuple(image.shape))
    for k, v in targets.items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}  "
              f"min={v.float().min():.2f} max={v.float().max():.2f}")

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    img_np = image.permute(1, 2, 0).numpy()
    axes[0, 0].imshow(img_np); axes[0, 0].set_title("image")
    axes[0, 1].imshow(targets["curve_mask"].numpy(), cmap="gray"); axes[0, 1].set_title("curve_mask")
    axes[0, 2].imshow(targets["centerline_mask"].numpy(), cmap="gray"); axes[0, 2].set_title("centerline")
    axes[0, 3].imshow(targets["instance_ids"].numpy(), cmap="tab20"); axes[0, 3].set_title("instance_ids")
    axes[1, 0].imshow(targets["crossing_mask"].numpy(), cmap="hot"); axes[1, 0].set_title("crossing")
    axes[1, 1].imshow(targets["boundary_mask"].numpy(), cmap="hot"); axes[1, 1].set_title("boundary")
    axes[1, 2].imshow(targets["direction_vectors"][0].numpy(), cmap="RdBu"); axes[1, 2].set_title("dir_x")
    axes[1, 3].imshow(targets["direction_vectors"][1].numpy(), cmap="RdBu"); axes[1, 3].set_title("dir_y")
    plt.tight_layout()
    plt.savefig("dataset_debug.png", dpi=150)
    print("可视化已保存到 dataset_debug.png")
