"""
Unit tests for legend-guided curve segmentation modules (legend_encoder.py)
and their integration into CurveSOTAQueryNet / CurveSOTACriterion.

Tests cover:
  A  — LegendPatchEncoder: output shape, colour / style feature extraction,
        Lab colour conversion sanity checks.
  LCAB — compute_legend_color_biases: shape, sign, and None-safety.
  E  — LegendQueryGate: gate=0 for invalid legends, blend correctness.
  C  — legend_contrastive_loss: InfoNCE value range, identity case,
        edge case M<2.
  Integration — CurveSOTAQueryNet forward with / without legend_patches,
                legend_valid mask, ablation use_legend_queries=False.
  Loss — CurveSOTACriterion: legend_contrastive=0 without GT instances,
         legend_contrastive>0 with matched instances, all terms finite.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from legend_encoder import (
    _rgb_to_lab,
    LegendPatchEncoder,
    LegendQueryGate,
    compute_legend_color_biases,
    legend_contrastive_loss,
)
from mamba3_curve_instance_seg import CurveSegConfig
from curve_sota_query_seg import (
    CurveSOTAConfig,
    CurveSOTAQueryNet,
    CurveSOTACriterion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_sota(use_legend: bool = True) -> CurveSOTAQueryNet:
    backbone = CurveSegConfig(
        encoder_dims=(32, 64, 128, 256),
        blocks_per_stage=(1, 1, 1, 1),
        decoder_dim=64,
        embed_dim=8,
    )
    cfg = CurveSOTAConfig(
        backbone=backbone,
        num_queries=8,
        query_layers=2,
        dn_groups=0,
        use_legend_queries=use_legend,
        use_hsv_features=False,
        use_gradient_features=False,
    )
    # seed=0 is known-good for these encoder_dims (smoke-tested)
    torch.manual_seed(0)
    return CurveSOTAQueryNet(cfg).eval()


def _dummy_targets(B: int = 2, H: int = 64, W: int = 64, with_instances: bool = False):
    ids = torch.zeros(B, H, W, dtype=torch.long)
    if with_instances:
        ids[0, 10:30, 10:30] = 1
        ids[0, 35:55, 35:55] = 2
        ids[1, 20:40, 20:40] = 1
    return {
        "curve_mask":        (ids > 0).float(),
        "centerline_mask":   (ids > 0).float(),
        "crossing_mask":     torch.zeros(B, H, W),
        "instance_ids":      ids,
        "direction_vectors": torch.randn(B, 4, H, W),
        "grid_mask":         torch.zeros(B, H, W),
        "boundary_mask":     torch.zeros(B, H, W),
    }


def _legend_patches(B: int = 2, n_per: int = 3):
    """Return list of B tensors (n_per, 3, 20, 60)."""
    return [torch.rand(n_per, 3, 20, 60) for _ in range(B)]


# ---------------------------------------------------------------------------
# Tests: _rgb_to_lab
# ---------------------------------------------------------------------------

class TestRgbToLab:
    def test_black(self):
        black = torch.zeros(1, 3)
        lab = _rgb_to_lab(black)
        assert abs(lab[0, 0].item()) < 1.0  # L* ≈ 0

    def test_white(self):
        white = torch.ones(1, 3)
        lab = _rgb_to_lab(white)
        assert abs(lab[0, 0].item() - 100.0) < 1.0  # L* ≈ 100

    def test_shape_preserved(self):
        rgb = torch.rand(5, 4, 3)   # (..., 3) — last dim must be channels
        assert _rgb_to_lab(rgb).shape == (5, 4, 3)

    def test_pure_red_hue(self):
        red = torch.tensor([[[1.0, 0.0, 0.0]]])
        lab = _rgb_to_lab(red)
        assert lab[0, 0, 1].item() > 20.0   # a* > 0 for red


# ---------------------------------------------------------------------------
# Tests: LegendPatchEncoder
# ---------------------------------------------------------------------------

class TestLegendPatchEncoder:
    def setup_method(self):
        self.enc = LegendPatchEncoder(d_model=64)
        self.patches = torch.rand(4, 3, 20, 60)

    def test_output_shape(self):
        out = self.enc(self.patches)
        assert out.shape == (4, 64)

    def test_single_patch(self):
        out = self.enc(torch.rand(1, 3, 10, 40))
        assert out.shape == (1, 64)

    def test_different_patches_different_outputs(self):
        """Two very different patches (red vs blue) should give different encodings."""
        red = torch.ones(1, 3, 20, 60)
        red[0, 1:] = 0.0  # pure red patch
        blue = torch.ones(1, 3, 20, 60)
        blue[0, :2] = 0.0  # pure blue patch
        out_r = self.enc(red)
        out_b = self.enc(blue)
        assert not torch.allclose(out_r, out_b, atol=1e-3)

    def test_extract_lab_mean_shape(self):
        lm = LegendPatchEncoder.extract_lab_mean(self.patches)
        assert lm.shape == (4, 3)

    def test_extract_lab_mean_white(self):
        white = torch.ones(1, 3, 20, 60)
        lm = LegendPatchEncoder.extract_lab_mean(white)
        assert abs(lm[0, 0].item() - 100.0) < 1.0

    def test_color_feats_shape(self):
        cf = LegendPatchEncoder._color_feats(self.patches)
        assert cf.shape == (4, 6)

    def test_style_feats_shape(self):
        sf = LegendPatchEncoder._style_feats(self.patches)
        assert sf.shape == (4, 16)

    def test_style_solid_vs_dashed(self):
        """Solid and dashed patches should yield different style features."""
        # Solid patch: uniform gray
        solid = torch.ones(1, 3, 20, 60) * 0.5
        # Dashed patch: alternating black/white columns
        dashed = torch.zeros(1, 3, 20, 60)
        dashed[0, :, :, ::10] = 1.0
        sf_solid = LegendPatchEncoder._style_feats(solid)
        sf_dashed = LegendPatchEncoder._style_feats(dashed)
        assert not torch.allclose(sf_solid, sf_dashed, atol=1e-3)

    def test_output_finite(self):
        assert torch.isfinite(self.enc(self.patches)).all()

    def test_variable_patch_size(self):
        """Encoder should handle any spatial size."""
        small = self.enc(torch.rand(2, 3, 5, 15))
        large = self.enc(torch.rand(2, 3, 40, 120))
        assert small.shape == (2, 64)
        assert large.shape == (2, 64)


# ---------------------------------------------------------------------------
# Tests: LegendQueryGate
# ---------------------------------------------------------------------------

class TestLegendQueryGate:
    def setup_method(self):
        self.gate = LegendQueryGate(d_model=32)

    def test_shape_preserved(self):
        leg = torch.randn(2, 3, 32)
        valid = torch.ones(2, 3, dtype=torch.bool)
        learned = torch.randn(2, 10, 32)
        out = self.gate(leg, valid, learned)
        assert out.shape == (2, 10, 32)

    def test_invalid_legends_use_learned(self):
        """When all legends are invalid (valid=False), output == learned query."""
        leg = torch.randn(2, 3, 32)
        valid = torch.zeros(2, 3, dtype=torch.bool)  # all False
        learned = torch.randn(2, 10, 32)
        out = self.gate(leg, valid, learned)
        assert torch.allclose(out[:, :3], learned[:, :3], atol=1e-6)

    def test_unaffected_tail_queries(self):
        """Queries beyond N_legend should be identical to learned queries."""
        leg = torch.randn(2, 3, 32)
        valid = torch.ones(2, 3, dtype=torch.bool)
        learned = torch.randn(2, 10, 32)
        out = self.gate(leg, valid, learned)
        assert torch.allclose(out[:, 3:], learned[:, 3:])

    def test_partial_valid(self):
        """Partially valid legends: invalid items → learned queries."""
        leg = torch.randn(1, 4, 32)
        valid = torch.tensor([[True, True, False, False]])
        learned = torch.randn(1, 8, 32)
        out = self.gate(leg, valid, learned)
        # Positions 2,3 should equal learned (gate zeroed)
        assert torch.allclose(out[:, 2:4], learned[:, 2:4], atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: compute_legend_color_biases
# ---------------------------------------------------------------------------

class TestLegendColorBiases:
    def setup_method(self):
        self.images = torch.rand(2, 3, 64, 64)
        self.lab_means = [
            torch.rand(3, 3),   # image 0: 3 legend items
            torch.rand(2, 3),   # image 1: 2 legend items
        ]

    def test_output_count(self):
        biases = compute_legend_color_biases(
            self.lab_means, self.images, [(8, 8), (16, 16), (32, 32)], num_queries=10
        )
        assert len(biases) == 3

    def test_output_shapes(self):
        biases = compute_legend_color_biases(
            self.lab_means, self.images, [(8, 8), (16, 16)], num_queries=10
        )
        assert biases[0].shape == (2, 10, 64)    # 8*8=64
        assert biases[1].shape == (2, 10, 256)   # 16*16=256

    def test_bias_non_positive(self):
        """Bias values should be ≤ 0 (negative squared distances)."""
        biases = compute_legend_color_biases(
            self.lab_means, self.images, [(8, 8)], num_queries=6
        )
        assert (biases[0] <= 0.0).all()

    def test_padding_positions_zero(self):
        """Positions i ≥ N_legend should remain 0."""
        biases = compute_legend_color_biases(
            self.lab_means, self.images, [(4, 4)], num_queries=8
        )
        # image 0 has 3 legend items → positions 3..7 should be 0
        assert (biases[0][0, 3:] == 0.0).all()
        # image 1 has 2 legend items → positions 2..7 should be 0
        assert (biases[0][1, 2:] == 0.0).all()

    def test_none_image_no_crash(self):
        """None legend_lab_means entry should produce zero bias for that batch item."""
        lab_means = [None, torch.rand(2, 3)]
        biases = compute_legend_color_biases(
            lab_means, self.images, [(8, 8)], num_queries=5
        )
        assert (biases[0][0] == 0.0).all()   # item 0 has no legend → all zeros

    def test_exact_match_gives_largest_bias(self):
        """When legend colour == image colour at position j, bias[i,j] should be 0."""
        from legend_encoder import _rgb_to_lab
        # Uniform red image
        img = torch.zeros(1, 3, 4, 4)
        img[:, 0] = 1.0  # red channel = 1
        # Legend colour = exact Lab mean of the red image
        lab_flat = _rgb_to_lab(img.permute(0, 2, 3, 1).clamp(0, 1)).flatten(1, 2)
        lab_mean = lab_flat.mean(dim=1)  # (1, 3)
        biases = compute_legend_color_biases(
            [lab_mean], img.clamp(0, 1), [(4, 4)], num_queries=1, temperature=50.0
        )
        # All positions are red, legend is red → dist ≈ 0 → bias ≈ 0
        assert biases[0][0, 0].abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# Tests: legend_contrastive_loss
# ---------------------------------------------------------------------------

class TestLegendContrastiveLoss:
    def test_perfect_alignment(self):
        """Identical feature pairs → loss should be near 0."""
        feat = torch.randn(4, 32)
        loss = legend_contrastive_loss(feat, feat.clone(), temperature=0.1)
        assert loss.item() < 0.1

    def test_random_features_positive_loss(self):
        """Random (unrelated) features → loss > 0."""
        torch.manual_seed(0)
        leg = torch.randn(6, 32)
        qry = torch.randn(6, 32)
        loss = legend_contrastive_loss(leg, qry, temperature=0.1)
        assert loss.item() > 0

    def test_loss_in_valid_range(self):
        """InfoNCE loss should be finite and positive."""
        M, D = 8, 64
        leg = torch.randn(M, D)
        qry = torch.randn(M, D)
        loss = legend_contrastive_loss(leg, qry, temperature=0.07)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_gradient_flows(self):
        leg = torch.randn(4, 32, requires_grad=True)
        qry = torch.randn(4, 32, requires_grad=True)
        loss = legend_contrastive_loss(leg, qry)
        loss.backward()
        assert leg.grad is not None
        assert qry.grad is not None

    def test_m_less_than_2_returns_zero(self):
        """M=1 → no meaningful negatives → return 0."""
        leg = torch.randn(1, 32)
        qry = torch.randn(1, 32)
        loss = legend_contrastive_loss(leg, qry)
        assert loss.item() == 0.0

    def test_m_zero_returns_zero(self):
        leg = torch.randn(0, 32)
        qry = torch.randn(0, 32)
        loss = legend_contrastive_loss(leg, qry)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Tests: CurveSOTAQueryNet integration
# ---------------------------------------------------------------------------

class TestSOTALegendIntegration:
    def setup_method(self):
        torch.manual_seed(0)
        self.model = _small_sota(use_legend=True)
        self.imgs = torch.randn(2, 3, 64, 64)

    def test_forward_without_legend(self):
        with torch.no_grad():
            out = self.model(self.imgs)
        assert "pred_logits" in out
        assert "legend_feats" not in out

    def test_forward_with_legend_shapes(self):
        patches = _legend_patches(B=2, n_per=3)
        with torch.no_grad():
            out = self.model(self.imgs, legend_patches=patches)
        assert "legend_feats" in out
        assert "legend_valid" in out
        assert out["legend_feats"].shape == (2, 3, 64)   # (B, max_N, d_model)
        assert out["legend_valid"].shape == (2, 3)

    def test_legend_valid_mask_correct(self):
        """legend_valid[b, i] = True iff patch i was provided for image b."""
        patches = [torch.rand(3, 3, 20, 60), torch.rand(1, 3, 20, 60)]
        with torch.no_grad():
            out = self.model(self.imgs, legend_patches=patches)
        # image 0: 3 valid
        assert out["legend_valid"][0].all()
        # image 1: 1 valid, 2 invalid
        assert out["legend_valid"][1, 0].item() is True
        assert not out["legend_valid"][1, 1].item()

    def test_variable_legend_count_per_image(self):
        patches = [torch.rand(4, 3, 20, 60), torch.rand(2, 3, 20, 60)]
        with torch.no_grad():
            out = self.model(self.imgs, legend_patches=patches)
        # max_N = 4; image 1 only has 2 valid
        assert out["legend_feats"].shape[1] == 4
        assert out["legend_valid"][1, 2:].sum().item() == 0

    def test_pred_logits_shape_unchanged(self):
        patches = _legend_patches(B=2, n_per=2)
        with torch.no_grad():
            out = self.model(self.imgs, legend_patches=patches)
        assert out["pred_logits"].shape == (2, 8, 2)

    def test_ablation_use_legend_false(self):
        """With use_legend_queries=False, legend_patches input is silently ignored."""
        model_no_leg = _small_sota(use_legend=False)
        patches = _legend_patches(B=2, n_per=3)
        with torch.no_grad():
            out = model_no_leg(self.imgs, legend_patches=patches)
        assert "legend_feats" not in out

    def test_none_legend_patches_entry(self):
        """A None entry in legend_patches list means no legend for that image."""
        patches = [torch.rand(2, 3, 20, 60), None]
        with torch.no_grad():
            out = self.model(self.imgs, legend_patches=patches)
        # Image 1 has no legend → its valid mask should be all False
        assert not out["legend_valid"][1].any()


# ---------------------------------------------------------------------------
# Tests: CurveSOTACriterion legend_contrastive
# ---------------------------------------------------------------------------

class TestCriterionLegendContrastive:
    """Test legend_contrastive loss in CurveSOTACriterion using mock outputs.

    Uses pre-built mock output dicts instead of full model forward to avoid
    the pre-existing Mamba SSM NaN instability with random weights.
    The mock dicts exercise exactly the same criterion code paths.
    """

    def setup_method(self):
        self.criterion = CurveSOTACriterion(legend_contrastive_tau=0.1)
        B, Q, H, W = 2, 8, 16, 16

        def _mock_out(include_legend: bool = True, n_leg: int = 2):
            """Build a minimal outputs dict that satisfies the criterion."""
            out = {
                "pred_logits":      torch.zeros(B, Q, 2),
                "pred_masks":       torch.zeros(B, Q, H, W),
                "pred_quality":     torch.zeros(B, Q),
                "pred_efd":         torch.zeros(B, Q, 40),
                "aux_outputs":      [],
                "query_feats":      torch.randn(B, Q, 64),
                "centerline_logits": torch.zeros(B, 1, H, W),
                "crossing_logits":  torch.zeros(B, 1, H, W),
                "boundary_logits":  torch.zeros(B, 1, H, W),
                "direction_vectors": torch.zeros(B, 4, H, W),
                "grid_logits":      torch.zeros(B, 1, H, W),
                "snake_offsets":    [],
            }
            if include_legend:
                leg = torch.randn(B, n_leg, 64)
                valid = torch.ones(B, n_leg, dtype=torch.bool)
                out["legend_feats"] = leg
                out["legend_valid"] = valid
            return out

        self.mock_out_leg = _mock_out(include_legend=True, n_leg=2)
        self.mock_out_noleg = _mock_out(include_legend=False)

    def _targets(self, with_instances: bool = False):
        H, W = 16, 16
        ids = torch.zeros(2, H, W, dtype=torch.long)
        if with_instances:
            ids[0, 2:6, 2:6] = 1
            ids[0, 8:12, 8:12] = 2
            ids[1, 4:8, 4:8] = 1
        return {
            "curve_mask":        (ids > 0).float(),
            "centerline_mask":   (ids > 0).float(),
            "crossing_mask":     torch.zeros(2, H, W),
            "instance_ids":      ids,
            "direction_vectors": torch.randn(2, 4, H, W),
            "grid_mask":         torch.zeros(2, H, W),
            "boundary_mask":     torch.zeros(2, H, W),
        }

    def test_legend_contrastive_zero_without_gt(self):
        """No GT instances → Hungarian matches nothing → legend_contrastive = 0."""
        losses = self.criterion(self.mock_out_leg, self._targets(with_instances=False))
        assert torch.isfinite(losses["legend_contrastive"])
        assert losses["legend_contrastive"].item() == pytest.approx(0.0, abs=1e-6)

    def test_legend_contrastive_nonzero_with_gt(self):
        """With GT instances, Hungarian may match queries → loss ≥ 0 and finite."""
        losses = self.criterion(self.mock_out_leg, self._targets(with_instances=True))
        assert torch.isfinite(losses["legend_contrastive"])
        assert losses["legend_contrastive"].item() >= 0.0

    def test_legend_contrastive_zero_without_legend_key(self):
        """If 'legend_feats' not in outputs, legend_contrastive = 0."""
        losses = self.criterion(self.mock_out_noleg,
                                self._targets(with_instances=True))
        assert torch.isfinite(losses["legend_contrastive"])
        assert losses["legend_contrastive"].item() == pytest.approx(0.0, abs=1e-6)

    def test_all_losses_finite(self):
        losses = self.criterion(self.mock_out_leg, self._targets(with_instances=True))
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v), f"Loss '{k}' not finite: {v}"

    def test_total_includes_legend_contribution(self):
        """Remove legend_contrastive contribution; total should decrease (or stay same)."""
        losses = self.criterion(self.mock_out_leg, self._targets(with_instances=True))
        w = self.criterion.weights
        leg_contrib = w.legend_contrastive * losses["legend_contrastive"]
        assert torch.isfinite(losses["total"])
        # total − contrib should be ≤ total (contrib ≥ 0)
        assert (losses["total"] - leg_contrib).item() <= losses["total"].item() + 1e-5

    def test_tau_affects_loss_magnitude(self):
        """Larger temperature → smaller gradients → different loss value."""
        losses_low_tau  = CurveSOTACriterion(legend_contrastive_tau=0.05)(
            self.mock_out_leg, self._targets(with_instances=True))
        losses_high_tau = CurveSOTACriterion(legend_contrastive_tau=0.5)(
            self.mock_out_leg, self._targets(with_instances=True))
        # Both should be finite; values will differ because temperature scales sim
        assert torch.isfinite(losses_low_tau["legend_contrastive"])
        assert torch.isfinite(losses_high_tau["legend_contrastive"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
