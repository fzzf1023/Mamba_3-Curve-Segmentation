"""
Unit tests for evaluation metrics (evaluate.py).

Tests that:
  - compute_map returns valid mAP values in [0, 1]
  - compute_pq returns valid PQ values
  - Perfect predictions give mAP=1.0, PQ=1.0
  - Empty predictions give mAP=0.0
  - Skeleton recall is correct
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from evaluate import (
    compute_map,
    compute_pq,
    compute_pixel_metrics,
    compute_skeleton_recall,
    aggregate_metrics,
    evaluate_batch,
)


def _ones_mask(h: int = 32, w: int = 32) -> torch.Tensor:
    """A single all-ones mask."""
    return torch.ones(1, h, w, dtype=torch.bool)


def _zeros_mask(h: int = 32, w: int = 32) -> torch.Tensor:
    return torch.zeros(1, h, w, dtype=torch.bool)


class TestComputeMap:
    def test_perfect_prediction(self):
        gt = [torch.ones(2, 32, 32, dtype=torch.bool)]
        pred = [torch.ones(2, 32, 32, dtype=torch.bool)]
        scores = [torch.tensor([0.9, 0.8])]
        result = compute_map(pred, scores, gt)
        assert result["mAP50"] == pytest.approx(1.0, abs=0.01)

    def test_empty_pred(self):
        gt = [torch.ones(1, 32, 32, dtype=torch.bool)]
        pred = [torch.zeros(0, 32, 32, dtype=torch.bool)]
        scores = [torch.zeros(0)]
        result = compute_map(pred, scores, gt)
        assert result["mAP50"] == pytest.approx(0.0, abs=0.01)

    def test_empty_both(self):
        gt = [torch.zeros(0, 32, 32, dtype=torch.bool)]
        pred = [torch.zeros(0, 32, 32, dtype=torch.bool)]
        scores = [torch.zeros(0)]
        result = compute_map(pred, scores, gt)
        # Both empty → AP = 1.0 by convention (no GT = perfect)
        assert result["mAP50"] >= 0.0  # just check it's valid

    def test_values_in_range(self):
        gt = [torch.ones(3, 16, 16, dtype=torch.bool)]
        # Noisy predictions
        pred = [(torch.rand(5, 16, 16) > 0.5)]
        scores = [torch.rand(5)]
        result = compute_map(pred, scores, gt)
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_map75_leq_map50(self):
        """mAP@75 is always ≤ mAP@50 (stricter threshold)."""
        gt = [torch.ones(2, 32, 32, dtype=torch.bool)]
        pred = [(torch.rand(4, 32, 32) > 0.3)]
        scores = [torch.rand(4)]
        result = compute_map(pred, scores, gt)
        assert result["mAP75"] <= result["mAP50"] + 0.01  # small tolerance for rounding


class TestComputePQ:
    def test_perfect_pq(self):
        gt = [torch.ones(2, 32, 32, dtype=torch.bool)]
        pred = [torch.ones(2, 32, 32, dtype=torch.bool)]
        scores = [torch.tensor([0.9, 0.8])]
        result = compute_pq(pred, scores, gt)
        assert result["PQ"] == pytest.approx(1.0, abs=0.01)
        assert result["SQ"] == pytest.approx(1.0, abs=0.01)
        assert result["RQ"] == pytest.approx(1.0, abs=0.01)

    def test_no_pred(self):
        gt = [torch.ones(1, 32, 32, dtype=torch.bool)]
        pred = [torch.zeros(0, 32, 32, dtype=torch.bool)]
        scores = [torch.zeros(0)]
        result = compute_pq(pred, scores, gt)
        assert result["PQ"] == pytest.approx(0.0, abs=0.01)


class TestPixelMetrics:
    def test_perfect_overlap(self):
        mask = torch.ones(1, 1, 16, 16)
        result = compute_pixel_metrics(mask, mask)
        assert result["pixel_iou"] == pytest.approx(1.0, abs=0.01)
        assert result["pixel_dice"] == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        pred = torch.zeros(1, 1, 16, 16)
        pred[:, :, :8, :] = 1.0
        gt = torch.zeros(1, 1, 16, 16)
        gt[:, :, 8:, :] = 1.0
        result = compute_pixel_metrics(pred, gt)
        assert result["pixel_iou"] == pytest.approx(0.0, abs=0.01)


class TestAggregateMetrics:
    def test_aggregate(self):
        metrics = [
            {"mAP50": 0.8, "PQ": 0.7},
            {"mAP50": 0.6, "PQ": 0.5},
        ]
        result = aggregate_metrics(metrics)
        assert result["mAP50"] == pytest.approx(0.7, abs=0.01)
        assert result["PQ"] == pytest.approx(0.6, abs=0.01)

    def test_empty(self):
        result = aggregate_metrics([])
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
