"""Legend-Guided Curve Segmentation Modules.

Implements four innovations that use chart legend information as prior
knowledge to guide curve instance segmentation:

  A  — LegendPatchEncoder
       Analytical Lab colour statistics + FFT line-style descriptor → d_model.
       Used to initialise one DETR query per legend item so that each query
       starts the decoding process already "knowing" what colour and line style
       it is looking for.

  C  — legend_contrastive_loss
       Bidirectional InfoNCE that aligns the legend patch encoding with the
       matched decoder query feature vector, encouraging the query space to be
       geometrically consistent with the legend's visual signature.

  LCAB — compute_legend_color_biases
       Computes a spatial colour-similarity attention bias for each decoder
       cross-attention layer.  For query i the bias at spatial position j is:
           bias[i, j] = −‖Lab(legend_i) − Lab(image_j)‖² / τ
       Added to the raw attention logits so that every cross-attention layer
       is guided toward colour-matching image regions—not only the first layer.

  E  — LegendQueryGate
       Adaptive blend gate:  query = σ(gate) · legend_init
                                    + (1 − σ(gate)) · learned_query
       When legend patches are absent (legend_valid = False) the gate
       collapses to 0, and the model falls back to its standard learned
       query embeddings.  This makes the system robust to legend-free charts
       without requiring a separate inference code path.

All modules are purely optional—CurveSOTAQueryNet accepts
``legend_patches=None`` and degrades gracefully to standard behaviour.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Differentiable sRGB → CIE Lab conversion
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: Tensor) -> Tensor:
    """Differentiable sRGB → CIE Lab (D65 illuminant, 2° observer).

    Args:
        rgb: (..., 3) float tensor in [0, 1] (sRGB).

    Returns:
        lab: (..., 3) — L* in [0, 100], a*/b* typically in [−128, 127].

    Implementation follows IEC 61966-2-1 for linearisation and the
    standard CIE 1976 formula for XYZ → L*a*b*.
    """
    rgb = rgb.clamp(0.0, 1.0)

    # Gamma linearisation (IEC 61966-2-1)
    lin = torch.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055).clamp_min(1e-10) ** 2.4,
        rgb / 12.92,
    )

    # Linear sRGB → XYZ (D65, Bradford-adapted)
    M = lin.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = torch.einsum("...c,dc->...d", lin, M)  # (..., 3)

    # Normalise by D65 white point
    wp = lin.new_tensor([0.95047, 1.00000, 1.08883])
    xyz = xyz / wp

    # Piecewise cube-root / linear transfer
    delta = 6.0 / 29.0
    f = torch.where(
        xyz > delta ** 3,
        xyz.clamp_min(1e-10).pow(1.0 / 3.0),
        xyz / (3.0 * delta ** 2) + 4.0 / 29.0,
    )

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


# ---------------------------------------------------------------------------
# Innovation A — Legend Patch Encoder
# ---------------------------------------------------------------------------

class LegendPatchEncoder(nn.Module):
    """Analytical-plus-learned legend patch encoder.

    Two analytical descriptors (no convolution, size-invariant):
      1. Lab colour statistics — mean and std for each of L*, a*, b* → 6 dims.
         Uses CIE Lab, whose Euclidean distance is perceptually uniform.
      2. FFT line-style descriptor — the magnitude spectrum of the horizontal
         projection of the grayscale patch captures line periodicity:
           • solid line  → flat spectrum (no peaks)
           • dashed line → clear fundamental-frequency peak
           • dotted line → peak at higher frequency
         First 16 bins (normalised by DC) → 16 dims.

    Total analytical features: 22 dims → 2-layer MLP → d_model.
    Only the MLP is learnable; the feature extraction is deterministic.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(22, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

    # ------------------------------------------------------------------
    # Static feature extractors (pure tensor ops, fully differentiable)
    # ------------------------------------------------------------------

    @staticmethod
    def _color_feats(patches: Tensor) -> Tensor:
        """Lab mean + std for each channel.

        patches: (N, 3, H, W) ∈ [0, 1]  →  (N, 6)
        """
        hwc = patches.permute(0, 2, 3, 1).clamp(0.0, 1.0)  # (N, H, W, 3)
        lab = _rgb_to_lab(hwc)                               # (N, H, W, 3)
        flat = lab.flatten(1, 2)                             # (N, H*W, 3)
        mean = flat.mean(dim=1)                              # (N, 3)
        std = flat.std(dim=1).clamp_min(0.0)                 # (N, 3)
        return torch.cat([mean, std], dim=-1)                # (N, 6)

    @staticmethod
    def _style_feats(patches: Tensor) -> Tensor:
        """FFT-based line-style descriptor.

        patches: (N, 3, H, W) ∈ [0, 1]  →  (N, 16)
        """
        # Luminance image
        gray = (0.2989 * patches[:, 0]
                + 0.5870 * patches[:, 1]
                + 0.1140 * patches[:, 2])        # (N, H, W)
        # Horizontal projection: average along height axis
        proj = gray.mean(dim=1)                              # (N, W)
        # Real FFT; take magnitude
        mag = torch.fft.rfft(proj, dim=-1).abs()             # (N, W//2+1)
        n_bins = min(16, mag.shape[-1])
        feats = mag[:, :n_bins]
        if n_bins < 16:
            feats = F.pad(feats, (0, 16 - n_bins))
        # Normalise by DC so absolute brightness cancels out
        feats = feats / feats[:, :1].clamp_min(1e-6)        # (N, 16)
        return feats

    @staticmethod
    def extract_lab_mean(patches: Tensor) -> Tensor:
        """Return the Lab mean colour per patch (used by LCAB, no gradient needed).

        patches: (N, 3, H, W) ∈ [0, 1]  →  (N, 3)
        """
        hwc = patches.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        return _rgb_to_lab(hwc).flatten(1, 2).mean(dim=1)   # (N, 3)

    def forward(self, patches: Tensor) -> Tensor:
        """Encode a batch of legend patches into query initialisation vectors.

        patches: (N, 3, H, W) ∈ [0, 1]
        Returns: (N, d_model)
        """
        color = self._color_feats(patches)                   # (N, 6)
        style = self._style_feats(patches)                   # (N, 16)
        return self.mlp(torch.cat([color, style], dim=-1))   # (N, d_model)


# ---------------------------------------------------------------------------
# Innovation E — Legend-Absent Adaptive Gate
# ---------------------------------------------------------------------------

class LegendQueryGate(nn.Module):
    """Adaptive blend gate for legend-absent robustness (Innovation E).

    For each of the first N_legend queries the gate computes a scalar
    α ∈ [0, 1] conditioned on the legend encoding:

        query_i = α_i · legend_init_i + (1 − α_i) · learned_query_i

    When legend patches are absent (legend_valid[b, i] = False) the mask
    forces α_i = 0, ensuring complete fallback to the learned queries without
    any legend-specific contamination.

    A high-quality, distinctive legend encoding naturally produces a large
    α (the gate is trained to recognise when the prior is useful), while a
    zero-padded or near-zero legend encoding produces α ≈ 0.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        legend_init: Tensor,    # (B, N_leg, D)
        legend_valid: Tensor,   # (B, N_leg) bool — True for real items
        learned_q: Tensor,      # (B, Q, D) — standard learnable queries
    ) -> Tensor:
        """Return (B, Q, D) with the first N_leg queries blended."""
        B, N_leg = legend_init.shape[:2]
        gate = self.gate_proj(legend_init)                      # (B, N_leg, 1)
        gate = gate * legend_valid.float().unsqueeze(-1)        # zero invalid
        q = learned_q.clone()
        q[:, :N_leg] = gate * legend_init + (1.0 - gate) * learned_q[:, :N_leg]
        return q                                                # (B, Q, D)


# ---------------------------------------------------------------------------
# LCAB — Legend-Conditioned Attention Bias
# ---------------------------------------------------------------------------

def compute_legend_color_biases(
    legend_lab_means: List[Optional[Tensor]],
    images_rgb: Tensor,
    memory_hws: List[Tuple[int, int]],
    num_queries: int,
    temperature: float = 50.0,
) -> List[Tensor]:
    """Compute per-query colour-similarity attention biases for K memory levels.

    For each decoder cross-attention the attention logit for query i at
    spatial position j is biased by:

        bias[b, i, j] = −‖Lab(legend_i) − Lab(image_j)‖² / temperature

    This guides every cross-attention layer—not only query initialisation—
    to preferentially attend to colour-matching image regions without
    introducing learnable parameters.

    Args:
        legend_lab_means: One Optional[(N_i, 3)] Lab-mean tensor per batch item.
                          None (or empty) means no legend data for that item.
        images_rgb:       (B, 3, H, W) in [0, 1].
        memory_hws:       Spatial (h, w) for each memory level.
        num_queries:      Total number of DETR queries Q.
        temperature:      Larger → sharper (more focused) colour attention.

    Returns:
        List of K tensors, each (B, Q, h_k * w_k).
        Positions i ≥ N_i (no legend) are left at 0 (no bias).
    """
    B = images_rgb.shape[0]
    device = images_rgb.device
    img_rgb01 = images_rgb.clamp(0.0, 1.0)

    biases: List[Tensor] = []
    for (h, w) in memory_hws:
        # Downsample to memory resolution and convert to Lab
        img_small = F.interpolate(img_rgb01, size=(h, w),
                                  mode="bilinear", align_corners=False)
        img_lab = _rgb_to_lab(img_small.permute(0, 2, 3, 1))   # (B, h, w, 3)
        img_lab_flat = img_lab.flatten(1, 2)                     # (B, HW, 3)

        bias = images_rgb.new_zeros(B, num_queries, h * w)
        for b_idx in range(B):
            leg = legend_lab_means[b_idx]
            if leg is None or leg.shape[0] == 0:
                continue
            leg = leg.to(device)
            n_leg = min(leg.shape[0], num_queries)
            # (n_leg, 1, 3) − (1, HW, 3) → (n_leg, HW, 3) → (n_leg, HW)
            diff = leg[:n_leg].unsqueeze(1) - img_lab_flat[b_idx].unsqueeze(0)
            sq_dist = (diff ** 2).sum(-1)
            bias[b_idx, :n_leg] = -sq_dist / temperature

        biases.append(bias)        # (B, Q, HW_k)
    return biases


# ---------------------------------------------------------------------------
# Innovation C — Legend–Curve Contrastive Alignment Loss
# ---------------------------------------------------------------------------

def legend_contrastive_loss(
    legend_feats: Tensor,
    query_feats: Tensor,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> Tensor:
    """Bidirectional InfoNCE aligning legend encodings with matched curve queries.

    For M valid (legend_i, query_j) pairs (one per matched GT instance):
      • Positive: legend_i ↔ query_j (matched by Hungarian assignment)
      • Negatives: all other cross-pairs within the batch

    This pulls the decoder query feature space toward the legend's visual
    signature and pushes different-legend queries apart, complementing the
    existing query-level PCC loss (which uses GT assignment, not visual style).

    Args:
        legend_feats: (M, D) — legend patch encodings for M valid pairs.
        query_feats:  (M, D) — matched decoder query feature vectors.
        temperature:  InfoNCE temperature (lower = sharper, harder negatives).

    Returns:
        Scalar loss.  Returns 0 (no gradient) when M < 2.
    """
    M = legend_feats.shape[0]
    if M < 2:
        return legend_feats.sum() * 0.0

    # Float32 for numerical stability under AMP
    leg = F.normalize(legend_feats.float(), dim=-1)    # (M, D)
    qry = F.normalize(query_feats.float(), dim=-1)     # (M, D)

    sim = torch.matmul(leg, qry.t()) / temperature     # (M, M)

    # Diagonal = positive pairs; off-diagonal = negatives
    labels = torch.arange(M, device=legend_feats.device)
    loss_l2q = F.cross_entropy(sim, labels)             # legend → query
    loss_q2l = F.cross_entropy(sim.t(), labels)         # query  → legend
    return (loss_l2q + loss_q2l) * 0.5
