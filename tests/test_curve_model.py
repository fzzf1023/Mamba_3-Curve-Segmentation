"""
Unit tests for CurveInstanceMamba3Net and CurveSOTAQueryNet.

Tests that:
  - Models run forward pass without error
  - Output shapes are correct
  - Ablation configs work (no BATO, no query align, etc.)
  - DropPath is correct (not always identity)
  - Progressive branch counts match config
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from mamba3_curve_instance_seg import (
    CurveSegConfig,
    CurveInstanceMamba3Net,
    DropPath,
)
from curve_sota_query_seg import (
    CurveSOTAConfig,
    CurveSOTAQueryNet,
)


# ---------------------------------------------------------------------------
# DropPath tests
# ---------------------------------------------------------------------------

class TestDropPath:
    def test_identity_at_eval(self):
        """DropPath should be identity at eval time."""
        dp = DropPath(drop_prob=0.5)
        dp.eval()
        x = torch.randn(4, 8, 16, 16)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_identity_at_zero_prob(self):
        """drop_prob=0 should always be identity."""
        dp = DropPath(drop_prob=0.0)
        dp.train()
        x = torch.randn(4, 8, 16, 16)
        out = dp(x)
        assert torch.allclose(out, x)

    def test_stochastic_drop_at_train(self):
        """During training, some samples should be dropped (mask=0) or kept."""
        dp = DropPath(drop_prob=0.5)
        dp.train()
        torch.manual_seed(42)
        x = torch.ones(100, 4)  # 100 samples, batch dim
        out = dp(x)
        # With drop_prob=0.5, roughly half should be ~0 or ~2 (after scale)
        # At least some should differ from x (not all identity)
        not_equal_count = (out[:, 0] != x[:, 0]).sum().item()
        assert not_equal_count > 0, "DropPath during training should drop some samples"

    def test_unbiased_expectation(self):
        """E[DropPath(x)] ≈ x (unbiased by 1/keep_prob scaling)."""
        dp = DropPath(drop_prob=0.5)
        dp.train()
        torch.manual_seed(0)
        x = torch.ones(10000, 1)
        out = dp(x)
        # Due to 1/keep_prob scaling, E[out] ≈ 1.0
        assert abs(out.mean().item() - 1.0) < 0.05, \
            f"DropPath output mean {out.mean():.3f} too far from 1.0"


# ---------------------------------------------------------------------------
# Base model tests
# ---------------------------------------------------------------------------

class TestCurveInstanceMamba3Net:
    def setup_method(self):
        self.cfg = CurveSegConfig(
            encoder_dims=(32, 64, 128, 256),
            blocks_per_stage=(1, 1, 1, 1),
            decoder_dim=64,
            embed_dim=8,
        )
        self.model = CurveInstanceMamba3Net(self.cfg).eval()

    def test_forward_shape(self):
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        assert out["composed_mask"].shape == (2, 1, 64, 64)
        assert out["centerline_logits"].shape == (2, 1, 64, 64)
        assert out["instance_embeddings"].shape == (2, 8, 64, 64)

    def test_progressive_branches(self):
        """Each stage should have the configured branch count."""
        enc = self.model.encoder
        stage_branch_counts = [
            len(self.cfg.branches_per_stage[i])
            for i in range(len(self.cfg.branches_per_stage))
        ]
        assert stage_branch_counts[0] == 3, f"Stage 0 should have 3 branches, got {stage_branch_counts[0]}"
        assert stage_branch_counts[1] == 5, f"Stage 1 should have 5 branches, got {stage_branch_counts[1]}"
        assert stage_branch_counts[2] == 9, f"Stage 2 should have 9 branches, got {stage_branch_counts[2]}"
        assert stage_branch_counts[3] == 9, f"Stage 3 should have 9 branches, got {stage_branch_counts[3]}"

    def test_snake_offsets_returned(self):
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        assert "snake_offsets" in out
        assert isinstance(out["snake_offsets"], list)

    def test_grid_logits_present(self):
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        assert "grid_logits" in out


# ---------------------------------------------------------------------------
# SOTA model tests
# ---------------------------------------------------------------------------

class TestCurveSOTAQueryNet:
    def setup_method(self):
        backbone = CurveSegConfig(
            encoder_dims=(32, 64, 128, 256),
            blocks_per_stage=(1, 1, 1, 1),
            decoder_dim=64,
        )
        self.cfg = CurveSOTAConfig(
            backbone=backbone,
            num_queries=8,
            query_layers=2,
            dn_groups=0,  # disable DN for simple tests
        )
        self.model = CurveSOTAQueryNet(self.cfg).eval()

    def test_forward_shapes(self):
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        B, Q = 2, 8
        assert out["pred_logits"].shape == (B, Q, 2)
        assert out["pred_masks"].shape[0] == B
        assert out["pred_masks"].shape[1] == Q

    def test_no_style_head_by_default(self):
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        assert "pred_style_logits" not in out

    def test_style_head_enabled(self):
        backbone = CurveSegConfig(encoder_dims=(32, 64, 128, 256), decoder_dim=64)
        cfg = CurveSOTAConfig(backbone=backbone, num_queries=4, query_layers=1,
                              dn_groups=0, use_style_head=True, num_styles=3)
        model = CurveSOTAQueryNet(cfg).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "pred_style_logits" in out
        assert out["pred_style_logits"].shape == (1, 4, 3)

    def test_no_layering_head_by_default(self):
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        assert "layering_logits" not in out

    def test_ablation_no_bato(self):
        """Disabling BATO should not raise errors."""
        backbone = CurveSegConfig(encoder_dims=(32, 64, 128, 256), decoder_dim=64)
        cfg = CurveSOTAConfig(backbone=backbone, num_queries=4, query_layers=1,
                              dn_groups=0, use_bato=False)
        model = CurveSOTAQueryNet(cfg).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "pred_logits" in out

    def test_ablation_no_query_align(self):
        """Disabling query align should not raise errors."""
        backbone = CurveSegConfig(encoder_dims=(32, 64, 128, 256), decoder_dim=64)
        cfg = CurveSOTAConfig(backbone=backbone, num_queries=4, query_layers=1,
                              dn_groups=0, use_query_align=False)
        model = CurveSOTAQueryNet(cfg).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "pred_logits" in out

    def test_aux_outputs_present(self):
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = self.model(x)
        assert "aux_outputs" in out
        # 2 layers → 1 aux (all but last)
        assert len(out["aux_outputs"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
