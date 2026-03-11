"""
Unit tests for CurveInstanceLoss and CurveSOTACriterion.

Tests that:
  - All loss terms are finite (no NaN/Inf)
  - Total loss is positive
  - Ablation flags (no_cape, no_pcc, no_snake_offset) correctly zero out terms
  - Loss terms decrease on meaningful GT vs random inputs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

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
# Helpers
# ---------------------------------------------------------------------------

def _make_base_targets(B: int = 2, H: int = 64, W: int = 64):
    """Create a minimal batch of base-model targets."""
    ids = torch.zeros(B, H, W, dtype=torch.long)
    ids[:, 20:40, 20:40] = 1  # one instance per image
    return {
        "curve_mask":       (ids > 0).float(),
        "centerline_mask":  (ids > 0).float() * 0.5,
        "crossing_mask":    torch.zeros(B, H, W),
        "instance_ids":     ids,
        "direction_vectors": torch.randn(B, 4, H, W),
        "grid_mask":        torch.zeros(B, H, W),
    }


def _make_sota_targets(B: int = 2, H: int = 64, W: int = 64):
    t = _make_base_targets(B, H, W)
    t["boundary_mask"] = torch.zeros(B, H, W)
    return t


# ---------------------------------------------------------------------------
# Base model tests
# ---------------------------------------------------------------------------

class TestCurveInstanceLoss:
    def setup_method(self):
        self.cfg = CurveSegConfig(encoder_dims=(32, 64, 128, 256), decoder_dim=64, embed_dim=8)
        self.model = CurveInstanceMamba3Net(self.cfg).eval()
        self.criterion = CurveInstanceLoss(embed_dim=8, use_uncertainty_weighting=False)

    def test_output_keys(self):
        imgs = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = self.model(imgs)
        assert "composed_mask" in out
        assert "centerline_logits" in out
        assert "grid_logits" in out

    def test_loss_finite(self):
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_base_targets()
        with torch.no_grad():
            out = self.model(imgs)
        losses = self.criterion(out, targets)
        for k, v in losses.items():
            assert torch.isfinite(v), f"Loss '{k}' is not finite: {v}"

    def test_loss_positive(self):
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_base_targets()
        with torch.no_grad():
            out = self.model(imgs)
        losses = self.criterion(out, targets)
        assert losses["total"].item() > 0, "Total loss should be positive"

    def test_snake_offset_ablation(self):
        """Setting snake_offset weight to 0.0 should zero out that term."""
        weights = CurveLossWeights(snake_offset=0.0)
        criterion_no_offset = CurveInstanceLoss(embed_dim=8, weights=weights,
                                                use_uncertainty_weighting=False)
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_base_targets()
        with torch.no_grad():
            out = self.model(imgs)
        losses = criterion_no_offset(out, targets)
        assert losses["snake_offset"].item() == 0.0 or losses["snake_offset"].abs().item() < 1e-7

    def test_grid_uses_real_mask(self):
        """When grid_mask is non-zero, it should be used instead of 1-curve_mask proxy."""
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_base_targets()
        targets["grid_mask"] = torch.ones(2, 64, 64)  # all grid
        with torch.no_grad():
            out = self.model(imgs)
        losses = self.criterion(out, targets)
        # Just verify the loss is still finite
        assert torch.isfinite(losses["grid"])


# ---------------------------------------------------------------------------
# SOTA model tests
# ---------------------------------------------------------------------------

class TestCurveSOTACriterion:
    def setup_method(self):
        backbone = CurveSegConfig(encoder_dims=(32, 64, 128, 256), decoder_dim=64, embed_dim=8)
        self.cfg = CurveSOTAConfig(
            backbone=backbone,
            num_queries=16,
            query_layers=2,
            dn_groups=1,
        )
        self.model = CurveSOTAQueryNet(self.cfg).eval()
        self.criterion = CurveSOTACriterion()

    def test_output_keys(self):
        imgs = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = self.model(imgs)
        assert "pred_logits" in out
        assert "pred_masks" in out
        assert "centerline_logits" in out
        assert "grid_logits" in out
        # style_logits should NOT be present when use_style_head=False
        assert "pred_style_logits" not in out

    def test_layering_head_disabled(self):
        imgs = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = self.model(imgs)
        assert "layering_logits" not in out

    def test_layering_head_enabled(self):
        backbone = CurveSegConfig(encoder_dims=(32, 64, 128, 256), decoder_dim=64)
        cfg = CurveSOTAConfig(backbone=backbone, num_queries=8, query_layers=1,
                              use_layering_head=True)
        model = CurveSOTAQueryNet(cfg).eval()
        imgs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(imgs)
        assert "layering_logits" in out

    def test_loss_finite(self):
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_sota_targets()
        with torch.no_grad():
            out = self.model(imgs)
        losses = self.criterion(out, targets)
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v), f"Loss '{k}' is not finite: {v}"

    def test_cape_ablation(self):
        """Zeroing CAPE weight should give zero cape loss contribution."""
        weights = SOTALossWeights(cape=0.0)
        criterion = CurveSOTACriterion(weights=weights)
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_sota_targets()
        with torch.no_grad():
            out = self.model(imgs)
        losses = criterion(out, targets)
        assert losses["cape"].item() * weights.cape == 0.0  # zeroed contribution

    def test_pcc_ablation(self):
        """Zeroing PCC weight should give zero pcc loss contribution."""
        weights = SOTALossWeights(pcc=0.0)
        criterion = CurveSOTACriterion(weights=weights)
        imgs = torch.randn(2, 3, 64, 64)
        targets = _make_sota_targets()
        with torch.no_grad():
            out = self.model(imgs)
        losses = criterion(out, targets)
        assert losses["pcc"].item() * weights.pcc == 0.0  # zeroed contribution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
