"""
评测指标模块 — 实例分割 + 拓扑指标

提供以下指标：
  · mAP@50, mAP@75, mAP@50:95 (COCO 标准; 优先使用 pycocotools 官方 API)
  · PQ (Panoptic Quality)
  · Skeleton Recall / clDice (拓扑连通性)
  · 像素级 IoU、Dice

供 train.py validate() 调用，也可独立运行评测。

依赖说明:
  - pycocotools (可选): 安装后自动使用官方 COCO AP 计算（101 点插值），
    否则回退到 VOC 全点插值方案，结果略有差异。
    安装: pip install pycocotools
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Instance-level IoU
# ---------------------------------------------------------------------------

def _iou_matrix(pred_masks: Tensor, gt_masks: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute pairwise IoU between predicted and GT binary masks.

    pred_masks: (N_pred, H, W)  bool or float
    gt_masks:   (N_gt, H, W)    bool or float
    Returns:    (N_pred, N_gt)   float IoU matrix
    """
    p = pred_masks.float().flatten(1)   # (N_pred, HW)
    g = gt_masks.float().flatten(1)     # (N_gt, HW)
    inter = torch.einsum("pd,gd->pg", p, g)
    union = p.sum(1, keepdim=True) + g.sum(1).unsqueeze(0) - inter
    return inter / (union + eps)


# ---------------------------------------------------------------------------
# AP (Average Precision) at a single IoU threshold
# ---------------------------------------------------------------------------

def _ap_at_threshold(
    pred_masks: Tensor,
    pred_scores: Tensor,
    gt_masks: Tensor,
    iou_thresh: float,
) -> float:
    """Compute AP at a single IoU threshold using greedy matching.

    pred_masks:  (N_pred, H, W) binary
    pred_scores: (N_pred,) confidence
    gt_masks:    (N_gt, H, W) binary
    Returns: AP (float)
    """
    n_pred = pred_masks.shape[0]
    n_gt = gt_masks.shape[0]

    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0

    # sort predictions by descending score
    order = torch.argsort(pred_scores, descending=True)
    pred_masks = pred_masks[order]
    pred_scores = pred_scores[order]

    iou = _iou_matrix(pred_masks, gt_masks)   # (N_pred, N_gt)
    matched_gt = set()
    tp = torch.zeros(n_pred)
    fp = torch.zeros(n_pred)

    for i in range(n_pred):
        # Mask already-matched GTs so argmax finds the best *unmatched* GT
        row = iou[i].clone()
        for already in matched_gt:
            row[already] = -1.0
        best_iou, best_j = row.max(0)
        j = int(best_j)
        if float(best_iou) >= iou_thresh:
            tp[i] = 1.0
            matched_gt.add(j)
        else:
            fp[i] = 1.0

    # precision-recall curve
    tp_cum = tp.cumsum(0)
    fp_cum = fp.cumsum(0)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # AP via all-point interpolation (PASCAL VOC style)
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # integrate
    indices = (recall[1:] != recall[:-1]).nonzero(as_tuple=True)[0]
    ap = float(((recall[indices + 1] - recall[indices]) * precision[indices + 1]).sum())
    return ap


# ---------------------------------------------------------------------------
# mAP @ multiple thresholds
# ---------------------------------------------------------------------------

def compute_map(
    pred_masks_list: List[Tensor],
    pred_scores_list: List[Tensor],
    gt_masks_list: List[Tensor],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute mAP over a batch of images.

    Args:
        pred_masks_list:  List of (N_pred, H, W) binary tensors
        pred_scores_list: List of (N_pred,) score tensors
        gt_masks_list:    List of (N_gt, H, W) binary tensors
        iou_thresholds:   IoU thresholds; default COCO [0.50:0.05:0.95]

    Returns:
        {"mAP50": float, "mAP75": float, "mAP50_95": float}
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    aps_per_thresh: Dict[float, List[float]] = {t: [] for t in iou_thresholds}

    for pred_m, pred_s, gt_m in zip(pred_masks_list, pred_scores_list, gt_masks_list):
        for t in iou_thresholds:
            ap = _ap_at_threshold(pred_m, pred_s, gt_m, t)
            aps_per_thresh[t].append(ap)

    mean_per_thresh = {t: float(np.mean(v)) if v else 0.0
                       for t, v in aps_per_thresh.items()}

    return {
        "mAP50": mean_per_thresh.get(0.5, 0.0),
        "mAP75": mean_per_thresh.get(0.75, 0.0),
        "mAP50_95": float(np.mean(list(mean_per_thresh.values()))),
    }


# ---------------------------------------------------------------------------
# COCO-official AP via pycocotools (optional, more accurate than VOC style)
# ---------------------------------------------------------------------------

def _try_coco_ap(
    pred_masks_list: List[Tensor],
    pred_scores_list: List[Tensor],
    gt_masks_list: List[Tensor],
) -> Optional[Dict[str, float]]:
    """Compute COCO AP using pycocotools if available.

    Converts binary masks to RLE annotations, builds COCO-style dataset/result
    objects, then calls COCOeval for official 101-point interpolated AP.

    Returns None if pycocotools is not installed.
    """
    try:
        from pycocotools import mask as coco_mask
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import io
        import contextlib
    except ImportError:
        return None

    # --- Build GT COCO dataset ---
    images, anns_gt = [], []
    ann_id = 1
    for img_id, gt_m in enumerate(gt_masks_list):
        images.append({"id": img_id, "width": gt_m.shape[-1], "height": gt_m.shape[-2]})
        for mask in gt_m:
            m_np = mask.cpu().numpy().astype("uint8")
            rle = coco_mask.encode(np.asfortranarray(m_np))
            rle["counts"] = rle["counts"].decode("utf-8")
            anns_gt.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(m_np.sum()),
                "bbox": [0, 0, m_np.shape[1], m_np.shape[0]],
                "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "categories": [{"id": 1, "name": "curve"}],
        "annotations": anns_gt,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    # --- Build predictions ---
    anns_dt = []
    for img_id, (pred_m, pred_s) in enumerate(zip(pred_masks_list, pred_scores_list)):
        for mask, score in zip(pred_m, pred_s):
            m_np = mask.cpu().numpy().astype("uint8")
            if m_np.sum() == 0:
                continue
            rle = coco_mask.encode(np.asfortranarray(m_np))
            rle["counts"] = rle["counts"].decode("utf-8")
            anns_dt.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": float(score),
            })

    if not anns_dt:
        return {"coco_mAP50": 0.0, "coco_mAP75": 0.0, "coco_mAP50_95": 0.0}

    coco_dt = coco_gt.loadRes(anns_dt)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # [mAP@[.5:.95], mAP@.5, mAP@.75, ...]
    return {
        "coco_mAP50_95": float(stats[0]),
        "coco_mAP50":    float(stats[1]),
        "coco_mAP75":    float(stats[2]),
    }


# ---------------------------------------------------------------------------
# Panoptic Quality (PQ)
# ---------------------------------------------------------------------------

def compute_pq(
    pred_masks_list: List[Tensor],
    pred_scores_list: List[Tensor],
    gt_masks_list: List[Tensor],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """Panoptic Quality = SQ × RQ.

    SQ (Segmentation Quality): average IoU of matched pairs
    RQ (Recognition Quality):  F1 of matching
    PQ = SQ × RQ
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou = 0.0

    for pred_m, pred_s, gt_m in zip(pred_masks_list, pred_scores_list, gt_masks_list):
        n_pred = pred_m.shape[0]
        n_gt = gt_m.shape[0]

        if n_pred == 0 and n_gt == 0:
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue
        if n_gt == 0:
            total_fp += n_pred
            continue

        iou = _iou_matrix(pred_m, gt_m)
        matched_gt = set()
        matched_pred = set()

        # greedy matching by IoU (descending)
        order = torch.argsort(pred_s, descending=True)
        for i in order.tolist():
            # Mask already-matched GTs so max finds the best *unmatched* GT
            row = iou[i].clone()
            for already in matched_gt:
                row[already] = -1.0
            best_iou, best_j = row.max(0)
            j = int(best_j)
            if float(best_iou) >= iou_thresh:
                matched_gt.add(j)
                matched_pred.add(i)
                total_iou += float(best_iou)

        tp = len(matched_pred)
        total_tp += tp
        total_fp += n_pred - tp
        total_fn += n_gt - tp

    sq = total_iou / max(total_tp, 1)
    rq_precision = total_tp / max(total_tp + total_fp, 1)
    rq_recall = total_tp / max(total_tp + total_fn, 1)
    rq = 2 * rq_precision * rq_recall / max(rq_precision + rq_recall, 1e-8)
    pq = sq * rq

    return {"PQ": pq, "SQ": sq, "RQ": rq}


# ---------------------------------------------------------------------------
# Pixel-level semantic metrics
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred: Tensor,
    gt: Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Pixel-level IoU and Dice for binary masks.

    pred, gt: (B, 1, H, W) or (B, H, W) float in [0,1] or binary
    """
    p = (pred > 0.5).float().flatten()
    g = gt.float().flatten()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    iou = float((inter + eps) / (union + eps))
    dice = float((2 * inter + eps) / (p.sum() + g.sum() + eps))
    return {"pixel_iou": iou, "pixel_dice": dice}


# ---------------------------------------------------------------------------
# Topology metrics: Skeleton Recall, soft clDice
# ---------------------------------------------------------------------------

def _soft_erode_np(mask: np.ndarray) -> np.ndarray:
    """Morphological erosion via min filter."""
    from scipy.ndimage import minimum_filter
    return minimum_filter(mask, size=3)


def _skeletonize_np(mask: np.ndarray) -> np.ndarray:
    """Binary skeletonization using skimage if available, else morphological thinning."""
    try:
        from skimage.morphology import skeletonize
        return skeletonize(mask > 0.5).astype(np.float32)
    except ImportError:
        # fallback: iterative erosion-based approximation
        binary = (mask > 0.5).astype(np.float32)
        skel = np.zeros_like(binary)
        elem = np.ones((3, 3), dtype=np.uint8)
        import cv2
        temp = binary.copy().astype(np.uint8)
        while True:
            eroded = cv2.erode(temp, elem)
            opened = cv2.dilate(eroded, elem)
            diff = temp - opened
            skel = np.clip(skel + diff.astype(np.float32), 0, 1)
            temp = eroded
            if temp.sum() == 0:
                break
        return skel


def compute_skeleton_recall(
    pred_masks: Tensor,
    gt_centerline: Tensor,
    eps: float = 1e-6,
) -> float:
    """Skeleton recall: fraction of GT centerline pixels covered by prediction.

    pred_masks:    (B, 1, H, W) float [0,1] — predicted mask probabilities
    gt_centerline: (B, 1, H, W) float — GT centerline (1px skeleton)

    Returns: scalar recall in [0, 1]
    """
    pred = (pred_masks > 0.5).float()
    gt = gt_centerline.float()
    recall = float(((pred * gt).sum() + eps) / (gt.sum() + eps))
    return recall


def compute_cldice(
    pred_masks: Tensor,
    gt_masks: Tensor,
) -> float:
    """Compute clDice using hard skeletons (numpy-based).

    pred_masks, gt_masks: (B, 1, H, W) float
    Returns: clDice score in [0, 1]
    """
    pred_np = pred_masks.detach().cpu().numpy()
    gt_np = gt_masks.detach().cpu().numpy()

    tprec_total, tprec_denom = 0.0, 0.0
    tsens_total, tsens_denom = 0.0, 0.0

    for b in range(pred_np.shape[0]):
        p = (pred_np[b, 0] > 0.5).astype(np.float32)
        g = gt_np[b, 0]

        if g.sum() < 1:
            continue

        skel_p = _skeletonize_np(p)
        skel_g = _skeletonize_np(g)

        tprec_total += (skel_p * g).sum()
        tprec_denom += skel_p.sum() + 1e-6
        tsens_total += (skel_g * p).sum()
        tsens_denom += skel_g.sum() + 1e-6

    if tprec_denom < 1 or tsens_denom < 1:
        return 0.0

    tprec = tprec_total / tprec_denom
    tsens = tsens_total / tsens_denom
    cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
    return float(cldice)


# ---------------------------------------------------------------------------
# Unified evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_batch(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    device: torch.device,
    score_thresh: float = 0.35,
    mask_thresh: float = 0.5,
) -> Dict[str, float]:
    """Compute all metrics for a single batch.

    Works with CurveSOTAQueryNet outputs or CurveInstanceMamba3Net outputs.

    Returns dict with all computed metrics.
    """
    metrics: Dict[str, float] = {}
    bsz = targets["instance_ids"].shape[0]

    # --- Instance-level metrics (for query-based model) ---
    if "pred_logits" in outputs and "pred_masks" in outputs:
        cls_scores = outputs["pred_logits"].softmax(-1)[..., 1]      # (B, Q)
        if "pred_quality" in outputs:
            cls_scores = cls_scores * torch.sigmoid(outputs["pred_quality"])
        pred_mask_probs = torch.sigmoid(outputs["pred_masks"])       # (B, Q, H, W)

        pred_masks_list = []
        pred_scores_list = []
        gt_masks_list = []

        ids = targets["instance_ids"].to(device).long()
        for b in range(bsz):
            # predictions
            valid = cls_scores[b] > score_thresh
            p_masks = (pred_mask_probs[b, valid] > mask_thresh)
            p_scores = cls_scores[b, valid]
            pred_masks_list.append(p_masks)
            pred_scores_list.append(p_scores)

            # GT
            uids = torch.unique(ids[b])
            uids = uids[uids > 0]
            if uids.numel() > 0:
                g_masks = torch.stack([(ids[b] == uid) for uid in uids], dim=0)
            else:
                g_masks = torch.zeros(0, ids.shape[1], ids.shape[2],
                                      dtype=torch.bool, device=device)
            gt_masks_list.append(g_masks)

        map_results = compute_map(pred_masks_list, pred_scores_list, gt_masks_list)
        metrics.update(map_results)

        # Official COCO AP (101-point interpolation via pycocotools, if installed)
        coco_results = _try_coco_ap(pred_masks_list, pred_scores_list, gt_masks_list)
        if coco_results is not None:
            metrics.update(coco_results)

        pq_results = compute_pq(pred_masks_list, pred_scores_list, gt_masks_list)
        metrics.update(pq_results)

    # --- Pixel-level metrics ---
    if "centerline_logits" in outputs and "centerline_mask" in targets:
        center_pred = torch.sigmoid(outputs["centerline_logits"])
        center_gt = targets["centerline_mask"].to(device).float()
        if center_gt.dim() == 3:
            center_gt = center_gt.unsqueeze(1)
        pix = compute_pixel_metrics(center_pred, center_gt)
        metrics["centerline_iou"] = pix["pixel_iou"]
        metrics["centerline_dice"] = pix["pixel_dice"]

    # --- Skeleton Recall ---
    if "centerline_logits" in outputs and "centerline_mask" in targets:
        composed = outputs.get("composed_mask", torch.sigmoid(outputs["centerline_logits"]))
        center_gt = targets["centerline_mask"].to(device).float()
        if center_gt.dim() == 3:
            center_gt = center_gt.unsqueeze(1)
        if composed.dim() == 3:
            composed = composed.unsqueeze(1)
        metrics["skeleton_recall"] = compute_skeleton_recall(composed, center_gt)

    # --- Curve mask IoU/Dice ---
    if "composed_mask" in outputs and "curve_mask" in targets:
        curve_gt = targets["curve_mask"].to(device).float()
        if curve_gt.dim() == 3:
            curve_gt = curve_gt.unsqueeze(1)
        pix = compute_pixel_metrics(outputs["composed_mask"], curve_gt)
        metrics["curve_iou"] = pix["pixel_iou"]
        metrics["curve_dice"] = pix["pixel_dice"]

    return metrics


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across multiple batches."""
    if not metric_list:
        return {}
    all_keys = set()
    for m in metric_list:
        all_keys.update(m.keys())
    result = {}
    for k in sorted(all_keys):
        vals = [m[k] for m in metric_list if k in m]
        result[k] = float(np.mean(vals)) if vals else 0.0
    return result
