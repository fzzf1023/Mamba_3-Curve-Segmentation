"""
Evaluation metrics for chart curve segmentation and extraction.

The metrics are organized around the repository's end goal:
  - final curve extraction quality
  - topology / continuity quality
  - instance separation quality for the query-based SOTA model

For `CurveSOTAQueryNet`, both instance metrics and merged curve masks are
computed from the post-processed inference results so that validation matches
the model's actual extraction behavior.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def _ensure_channel_first(mask: Tensor) -> Tensor:
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask


# ---------------------------------------------------------------------------
# Instance-level IoU
# ---------------------------------------------------------------------------

def _iou_matrix(pred_masks: Tensor, gt_masks: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute pairwise IoU between predicted and GT binary masks."""
    p = pred_masks.float().flatten(1)
    g = gt_masks.float().flatten(1)
    inter = torch.einsum("pd,gd->pg", p, g)
    union = p.sum(1, keepdim=True) + g.sum(1).unsqueeze(0) - inter
    return inter / (union + eps)


# ---------------------------------------------------------------------------
# AP (Average Precision)
# ---------------------------------------------------------------------------

def _ap_at_threshold(
    pred_masks: Tensor,
    pred_scores: Tensor,
    gt_masks: Tensor,
    iou_thresh: float,
) -> float:
    """Compute AP at a single IoU threshold using greedy matching."""
    n_pred = pred_masks.shape[0]
    n_gt = gt_masks.shape[0]

    if n_gt == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0

    order = torch.argsort(pred_scores, descending=True)
    pred_masks = pred_masks[order]
    pred_scores = pred_scores[order]

    iou = _iou_matrix(pred_masks, gt_masks)
    matched_gt = set()
    tp = torch.zeros(n_pred)
    fp = torch.zeros(n_pred)

    for i in range(n_pred):
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

    tp_cum = tp.cumsum(0)
    fp_cum = fp.cumsum(0)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = (recall[1:] != recall[:-1]).nonzero(as_tuple=True)[0]
    ap = float(((recall[indices + 1] - recall[indices]) * precision[indices + 1]).sum())
    return ap


def compute_map(
    pred_masks_list: List[Tensor],
    pred_scores_list: List[Tensor],
    gt_masks_list: List[Tensor],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute mAP across IoU thresholds."""
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    aps_per_thresh: Dict[float, List[float]] = {t: [] for t in iou_thresholds}

    for pred_m, pred_s, gt_m in zip(pred_masks_list, pred_scores_list, gt_masks_list):
        for t in iou_thresholds:
            ap = _ap_at_threshold(pred_m, pred_s, gt_m, t)
            aps_per_thresh[t].append(ap)

    mean_per_thresh = {
        t: float(np.mean(v)) if v else 0.0
        for t, v in aps_per_thresh.items()
    }

    return {
        "mAP50": mean_per_thresh.get(0.5, 0.0),
        "mAP75": mean_per_thresh.get(0.75, 0.0),
        "mAP50_95": float(np.mean(list(mean_per_thresh.values()))),
    }


# ---------------------------------------------------------------------------
# COCO AP via pycocotools
# ---------------------------------------------------------------------------

def _try_coco_ap(
    pred_masks_list: List[Tensor],
    pred_scores_list: List[Tensor],
    gt_masks_list: List[Tensor],
) -> Optional[Dict[str, float]]:
    """Compute official COCO mask AP if pycocotools is available."""
    try:
        import contextlib
        import io

        from pycocotools import mask as coco_mask
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None

    images, anns_gt = [], []
    ann_id = 1
    for img_id, gt_m in enumerate(gt_masks_list):
        images.append({"id": img_id, "width": gt_m.shape[-1], "height": gt_m.shape[-2]})
        for mask in gt_m:
            m_np = mask.cpu().numpy().astype("uint8")
            rle = coco_mask.encode(np.asfortranarray(m_np))
            rle["counts"] = rle["counts"].decode("utf-8")
            anns_gt.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": float(m_np.sum()),
                    "bbox": [0, 0, m_np.shape[1], m_np.shape[0]],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "categories": [{"id": 1, "name": "curve"}],
        "annotations": anns_gt,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    anns_dt = []
    for img_id, (pred_m, pred_s) in enumerate(zip(pred_masks_list, pred_scores_list)):
        for mask, score in zip(pred_m, pred_s):
            m_np = mask.cpu().numpy().astype("uint8")
            if m_np.sum() == 0:
                continue
            rle = coco_mask.encode(np.asfortranarray(m_np))
            rle["counts"] = rle["counts"].decode("utf-8")
            anns_dt.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "score": float(score),
                }
            )

    if not anns_dt:
        return {"coco_mAP50": 0.0, "coco_mAP75": 0.0, "coco_mAP50_95": 0.0}

    coco_dt = coco_gt.loadRes(anns_dt)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "coco_mAP50_95": float(stats[0]),
        "coco_mAP50": float(stats[1]),
        "coco_mAP75": float(stats[2]),
    }


# ---------------------------------------------------------------------------
# Panoptic Quality
# ---------------------------------------------------------------------------

def compute_pq(
    pred_masks_list: List[Tensor],
    pred_scores_list: List[Tensor],
    gt_masks_list: List[Tensor],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """Panoptic Quality = SQ x RQ."""
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

        order = torch.argsort(pred_s, descending=True)
        for i in order.tolist():
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
# Pixel / topology metrics
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred: Tensor,
    gt: Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Pixel-level IoU, Dice, precision, and recall for binary masks."""
    p = (pred > 0.5).float().flatten()
    g = (gt > 0.5).float().flatten()
    tp = (p * g).sum()
    fp = (p * (1.0 - g)).sum()
    fn = ((1.0 - p) * g).sum()
    union = tp + fp + fn

    return {
        "pixel_iou": float((tp + eps) / (union + eps)),
        "pixel_dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "pixel_precision": float((tp + eps) / (tp + fp + eps)),
        "pixel_recall": float((tp + eps) / (tp + fn + eps)),
    }


def _skeletonize_np(mask: np.ndarray) -> np.ndarray:
    """Binary skeletonization using skimage if available, else morphology."""
    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask > 0.5).astype(np.float32)
    except ImportError:
        from scipy import ndimage as ndi

        binary = mask > 0.5
        skel = np.zeros_like(binary, dtype=bool)
        elem = np.ones((3, 3), dtype=bool)
        temp = binary.copy()
        while temp.any():
            eroded = ndi.binary_erosion(temp, structure=elem, border_value=0)
            opened = ndi.binary_dilation(eroded, structure=elem)
            skel |= temp & ~opened
            temp = eroded
        return skel.astype(np.float32)


def compute_skeleton_recall(
    pred_masks: Tensor,
    gt_centerline: Tensor,
    eps: float = 1e-6,
) -> float:
    """Fraction of GT centerline pixels covered by the final extracted curves."""
    pred = (pred_masks > 0.5).float()
    gt = (gt_centerline > 0.5).float()
    recall = float(((pred * gt).sum() + eps) / (gt.sum() + eps))
    return recall


def compute_cldice(pred_masks: Tensor, gt_masks: Tensor) -> float:
    """Topology-aware clDice computed from hard masks."""
    pred_np = pred_masks.detach().cpu().numpy()
    gt_np = gt_masks.detach().cpu().numpy()

    tprec_total, tprec_denom = 0.0, 0.0
    tsens_total, tsens_denom = 0.0, 0.0

    for b in range(pred_np.shape[0]):
        p = (pred_np[b, 0] > 0.5).astype(np.float32)
        g = (gt_np[b, 0] > 0.5).astype(np.float32)

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
    return float(2 * tprec * tsens / (tprec + tsens + 1e-8))


def _count_components_np(mask: np.ndarray) -> int:
    binary = mask > 0.5
    if not binary.any():
        return 0
    try:
        from scipy import ndimage as ndi

        _, count = ndi.label(binary)
        return int(count)
    except ImportError:
        visited = np.zeros_like(binary, dtype=bool)
        count = 0
        h, w = binary.shape
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or visited[y, x]:
                    continue
                count += 1
                stack = [(y, x)]
                visited[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if not visited[ny, nx] and binary[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
        return int(count)


def _count_endpoints_np(mask: np.ndarray) -> int:
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return 0
    padded = np.pad(binary, ((1, 1), (1, 1)), mode="constant")
    h, w = binary.shape
    neighbours = np.zeros_like(binary, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbours += padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return int(((binary > 0) & (neighbours == 1)).sum())


def compute_connectivity_metrics(pred_masks: Tensor, gt_masks: Tensor) -> Dict[str, float]:
    pred_np = pred_masks.detach().cpu().numpy()
    gt_np = gt_masks.detach().cpu().numpy()
    connectivity_scores = []
    fragmentation_scores = []
    endpoint_errors = []

    for b in range(pred_np.shape[0]):
        pred_bin = (pred_np[b, 0] > 0.5).astype(np.float32)
        gt_bin = (gt_np[b, 0] > 0.5).astype(np.float32)
        if gt_bin.sum() < 1:
            continue

        pred_skel = _skeletonize_np(pred_bin)
        gt_skel = _skeletonize_np(gt_bin)
        pred_components = _count_components_np(pred_skel)
        gt_components = max(_count_components_np(gt_skel), 1)
        pred_endpoints = _count_endpoints_np(pred_skel)
        gt_endpoints = max(_count_endpoints_np(gt_skel), 1)

        if pred_components == 0:
            connectivity_scores.append(0.0)
            fragmentation_scores.append(1.0)
            endpoint_errors.append(1.0)
            continue

        connectivity_scores.append(min(gt_components / pred_components, 1.0))
        fragmentation_scores.append(max(pred_components - gt_components, 0) / gt_components)
        endpoint_errors.append(min(abs(pred_endpoints - gt_endpoints) / gt_endpoints, 1.0))

    if not connectivity_scores:
        return {
            "curve_connectivity": 0.0,
            "curve_fragmentation": 0.0,
            "curve_endpoint_error": 0.0,
        }
    return {
        "curve_connectivity": float(np.mean(connectivity_scores)),
        "curve_fragmentation": float(np.mean(fragmentation_scores)),
        "curve_endpoint_error": float(np.mean(endpoint_errors)),
    }


def compute_crossing_metrics(pred: Tensor, gt: Tensor, eps: float = 1e-6) -> Dict[str, float]:
    pix = compute_pixel_metrics(pred, gt, eps=eps)
    precision = pix["pixel_precision"]
    recall = pix["pixel_recall"]
    return {
        "crossing_iou": pix["pixel_iou"],
        "crossing_precision": precision,
        "crossing_recall": recall,
        "crossing_f1": float(2 * precision * recall / max(precision + recall, eps)),
    }


def compute_split_merge_rates(
    pred_masks_list: List[Tensor],
    gt_masks_list: List[Tensor],
    overlap_thresh: float = 0.1,
) -> Dict[str, float]:
    split_rates = []
    merge_rates = []
    count_errors = []

    for pred_masks, gt_masks in zip(pred_masks_list, gt_masks_list):
        n_pred = int(pred_masks.shape[0])
        n_gt = int(gt_masks.shape[0])
        count_errors.append(min(abs(n_pred - n_gt) / max(n_gt, 1), 1.0))

        if n_pred == 0 or n_gt == 0:
            split_rates.append(0.0 if n_gt == 0 else 1.0)
            merge_rates.append(0.0 if n_pred == 0 else 1.0)
            continue

        pred_f = pred_masks.float().flatten(1)
        gt_f = gt_masks.float().flatten(1)
        inter = torch.einsum("pd,gd->pg", pred_f, gt_f)
        pred_cover = inter / pred_f.sum(1, keepdim=True).clamp_min(1.0)
        gt_cover = inter / gt_f.sum(1).unsqueeze(0).clamp_min(1.0)

        split_rates.append(float(((gt_cover >= overlap_thresh).sum(dim=0) > 1).float().mean()))
        merge_rates.append(float(((pred_cover >= overlap_thresh).sum(dim=1) > 1).float().mean()))

    if not split_rates:
        return {"split_rate": 0.0, "merge_rate": 0.0, "instance_count_error": 0.0}
    return {
        "split_rate": float(np.mean(split_rates)),
        "merge_rate": float(np.mean(merge_rates)),
        "instance_count_error": float(np.mean(count_errors)),
    }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _instance_ids_to_mask_list(instance_ids: Tensor) -> List[Tensor]:
    """Convert a batch of instance-id maps into per-image mask stacks."""
    results: List[Tensor] = []
    for ids in instance_ids:
        uids = torch.unique(ids)
        uids = uids[uids > 0]
        if uids.numel() > 0:
            masks = torch.stack([(ids == uid) for uid in uids], dim=0)
        else:
            h, w = ids.shape[-2:]
            masks = torch.zeros(0, h, w, dtype=torch.bool, device=ids.device)
        results.append(masks)
    return results


def _postprocessed_instance_results(
    outputs: Dict[str, Tensor],
    score_thresh: float,
    mask_thresh: float,
) -> List[Dict[str, Tensor]]:
    """Run SOTA post-processing so validation matches final inference."""
    from curve_sota_query_seg import InferenceConfig, postprocess_curve_instances

    cfg = InferenceConfig(score_thresh=score_thresh, mask_thresh=mask_thresh)
    return postprocess_curve_instances(outputs, cfg)


def _merge_instance_masks(
    instance_results: List[Dict[str, Tensor]],
    height: int,
    width: int,
) -> Tensor:
    """Union post-processed instances into a final extracted curve mask."""
    merged: List[Tensor] = []
    for result in instance_results:
        masks = result["masks"]
        if masks.numel() == 0:
            merged.append(
                torch.zeros(1, height, width, dtype=torch.float32, device=masks.device)
            )
        else:
            merged.append(masks.any(dim=0, keepdim=True).float())
    return torch.stack(merged, dim=0)


def _extract_curve_predictions(
    outputs: Dict[str, Tensor],
    score_thresh: float,
    mask_thresh: float,
) -> Tuple[Tensor, Optional[List[Tensor]], Optional[List[Tensor]]]:
    """Build the final extracted curve masks used for extraction metrics."""
    if "pred_logits" in outputs and "pred_masks" in outputs:
        instance_results = _postprocessed_instance_results(outputs, score_thresh, mask_thresh)
        height, width = outputs["pred_masks"].shape[-2:]
        curve_probs = _merge_instance_masks(instance_results, height, width)
        pred_masks_list = [result["masks"] for result in instance_results]
        pred_scores_list = [result["scores"] for result in instance_results]
        return curve_probs, pred_masks_list, pred_scores_list

    if "composed_mask" in outputs:
        return _ensure_channel_first(outputs["composed_mask"]), None, None

    if "centerline_logits" in outputs:
        return _ensure_channel_first(torch.sigmoid(outputs["centerline_logits"])), None, None

    raise KeyError("No usable curve extraction output found in model outputs.")


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_batch(
    outputs: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    device: torch.device,
    score_thresh: float = 0.35,
    mask_thresh: float = 0.5,
) -> Dict[str, float]:
    """Compute extraction-first metrics for a validation batch."""
    metrics: Dict[str, float] = {}
    curve_probs, pred_masks_list, pred_scores_list = _extract_curve_predictions(
        outputs, score_thresh, mask_thresh
    )

    # Instance-level metrics for the SOTA model only.
    if pred_masks_list is not None and pred_scores_list is not None and "instance_ids" in targets:
        gt_masks_list = _instance_ids_to_mask_list(targets["instance_ids"].to(device).long())
        metrics.update(compute_map(pred_masks_list, pred_scores_list, gt_masks_list))

        coco_results = _try_coco_ap(pred_masks_list, pred_scores_list, gt_masks_list)
        if coco_results is not None:
            metrics.update(coco_results)

        metrics.update(compute_pq(pred_masks_list, pred_scores_list, gt_masks_list))
        metrics.update(compute_split_merge_rates(pred_masks_list, gt_masks_list))

    # Dense centerline-head quality.
    if "centerline_logits" in outputs and "centerline_mask" in targets:
        center_pred = _ensure_channel_first(torch.sigmoid(outputs["centerline_logits"]))
        center_gt = _ensure_channel_first(targets["centerline_mask"].to(device).float())
        pix = compute_pixel_metrics(center_pred, center_gt)
        metrics["centerline_iou"] = pix["pixel_iou"]
        metrics["centerline_dice"] = pix["pixel_dice"]
        metrics["centerline_precision"] = pix["pixel_precision"]
        metrics["centerline_recall"] = pix["pixel_recall"]

        # Final extracted curve coverage over GT centerlines.
        metrics["skeleton_recall"] = compute_skeleton_recall(curve_probs, center_gt)
        metrics.update(compute_connectivity_metrics(curve_probs, center_gt))

    # Final extracted curves versus GT curve regions.
    if "curve_mask" in targets:
        curve_gt = _ensure_channel_first(targets["curve_mask"].to(device).float())
        pix = compute_pixel_metrics(curve_probs, curve_gt)
        metrics["curve_iou"] = pix["pixel_iou"]
        metrics["curve_dice"] = pix["pixel_dice"]
        metrics["curve_precision"] = pix["pixel_precision"]
        metrics["curve_recall"] = pix["pixel_recall"]
        metrics["curve_cldice"] = compute_cldice(curve_probs, curve_gt)
        if "centerline_mask" not in targets:
            metrics.update(compute_connectivity_metrics(curve_probs, curve_gt))

    crossing_key = "crossing_mask" if "crossing_mask" in targets else "junction_mask"
    if "crossing_logits" in outputs and crossing_key in targets:
        crossing_pred = _ensure_channel_first(torch.sigmoid(outputs["crossing_logits"]))
        crossing_gt = _ensure_channel_first(targets[crossing_key].to(device).float())
        metrics.update(compute_crossing_metrics(crossing_pred, crossing_gt))

    return metrics


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across multiple batches."""
    if not metric_list:
        return {}
    all_keys = set()
    for metrics in metric_list:
        all_keys.update(metrics.keys())
    result = {}
    for key in sorted(all_keys):
        vals = [metrics[key] for metrics in metric_list if key in metrics]
        result[key] = float(np.mean(vals)) if vals else 0.0
    return result
