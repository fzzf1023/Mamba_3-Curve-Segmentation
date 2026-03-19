"""Legend-guided modules for chart curve segmentation.

The upgraded legend path now disentangles colour and shape information:

  A  — LegendPatchEncoder
       Encodes legend patches with separate colour and shape branches, then
       fuses them using an analytical modality gate that can emphasise colour,
       shape, or both depending on the legend set.

  LCAB — compute_legend_color_biases
       Computes Lab-space colour attention bias for decoder cross-attention.

  LSAB — compute_legend_shape_biases
       Uses Scharr-based structural templates and normalized cross-correlation
       to highlight image regions whose local line pattern matches the legend.

  LGB  — compute_legend_guidance_biases
       Blends colour and shape bias using the per-legend modality weights.

  C  — legend_contrastive_loss
       Aligns legend encodings with matched decoder query features.

  E  — LegendQueryGate
       Falls back to learned queries when legend data is absent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def _rgb_to_gray(images: Tensor) -> Tensor:
    """Return luminance with shape (N,1,H,W) for batched RGB input."""
    if images.dim() == 4:
        r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    elif images.dim() == 3:
        r, g, b = images[..., 0], images[..., 1], images[..., 2]
    else:
        raise ValueError(f"Expected RGB tensor with 3 or 4 dims, got {images.shape}")
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _scharr_gradients(gray: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Return Scharr gx, gy, and magnitude for a (N,1,H,W) luminance tensor."""
    if gray.dim() != 4 or gray.shape[1] != 1:
        raise ValueError(f"Expected gray tensor of shape (N,1,H,W), got {gray.shape}")
    kx = torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    gm = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
    return gx, gy, gm


def _structural_maps(images_rgb: Tensor) -> Tensor:
    """Return normalized structural maps [gx, gy, |g|] for RGB images."""
    gray = _rgb_to_gray(images_rgb).unsqueeze(1) if images_rgb.dim() == 3 else _rgb_to_gray(images_rgb)
    gx, gy, gm = _scharr_gradients(gray)
    scale = gm.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return torch.cat([gx / scale, gy / scale, gm / scale], dim=1)


def _min_pairwise_distance(feats: Tensor) -> Tensor:
    """Return per-item distance to the closest other item; zero when N < 2."""
    n = feats.shape[0]
    if n < 2:
        return feats.new_zeros(n)
    dists = torch.cdist(feats, feats, p=2)
    eye = torch.eye(n, device=feats.device, dtype=torch.bool)
    dists = dists.masked_fill(eye, float("inf"))
    return dists.min(dim=1).values


def _odd_kernel_size(target: int, limit: int) -> int:
    """Clamp to an odd kernel size that fits the current feature map."""
    if limit <= 1:
        return 1
    size = min(target, limit if limit % 2 == 1 else limit - 1)
    if size <= 0:
        return 1
    if size % 2 == 0:
        size -= 1
    return max(1, size)


# ---------------------------------------------------------------------------
# Innovation A — Legend Patch Encoder
# ---------------------------------------------------------------------------

class LegendPatchEncoder(nn.Module):
    """Analytical-plus-learned legend patch encoder with colour/shape gating."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.color_mlp = nn.Sequential(
            nn.Linear(6, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.shape_mlp = nn.Sequential(
            nn.Linear(22, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
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
    def _shape_feats(patches: Tensor) -> Tensor:
        """Edge/periodicity descriptor for line style and marker structure."""
        gray = _rgb_to_gray(patches)                         # (N, H, W)
        gx, gy, gm = _scharr_gradients(gray)
        edge = gm.squeeze(1)
        edge_n = edge / edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        theta = torch.atan2(gy.squeeze(1), gx.squeeze(1))

        edge_mean = edge_n.mean(dim=(1, 2), keepdim=True)
        edge_std = edge_n.std(dim=(1, 2), keepdim=True).unsqueeze(-1)
        orient = torch.stack(
            [
                (edge_n * torch.cos(theta)).mean(dim=(1, 2)),
                (edge_n * torch.sin(theta)).mean(dim=(1, 2)),
                (edge_n * torch.cos(2.0 * theta)).mean(dim=(1, 2)),
                (edge_n * torch.sin(2.0 * theta)).mean(dim=(1, 2)),
            ],
            dim=-1,
        )

        row_proj = edge_n.mean(dim=1)
        col_proj = edge_n.mean(dim=2)
        row_fft = torch.fft.rfft(row_proj, dim=-1).abs()
        col_fft = torch.fft.rfft(col_proj, dim=-1).abs()

        row_bins = min(8, row_fft.shape[-1])
        col_bins = min(8, col_fft.shape[-1])
        row_feats = row_fft[:, :row_bins]
        col_feats = col_fft[:, :col_bins]
        if row_bins < 8:
            row_feats = F.pad(row_feats, (0, 8 - row_bins))
        if col_bins < 8:
            col_feats = F.pad(col_feats, (0, 8 - col_bins))
        row_feats = row_feats / row_feats[:, :1].clamp_min(1e-6)
        col_feats = col_feats / col_feats[:, :1].clamp_min(1e-6)

        return torch.cat(
            [
                edge_mean.flatten(1),
                edge_std.flatten(1),
                orient,
                row_feats,
                col_feats,
            ],
            dim=-1,
        )

    @staticmethod
    def _style_feats(patches: Tensor) -> Tensor:
        """Backward-compatible alias for the upgraded shape descriptor."""
        return LegendPatchEncoder._shape_feats(patches)

    @staticmethod
    def _modality_weights(color_feats: Tensor, shape_feats: Tensor) -> Tensor:
        """Estimate whether each legend is better separated by colour or shape."""
        color_norm = F.layer_norm(color_feats, (color_feats.shape[-1],))
        shape_norm = F.layer_norm(shape_feats, (shape_feats.shape[-1],))

        color_sep = _min_pairwise_distance(color_norm)
        shape_sep = _min_pairwise_distance(shape_norm)

        chroma = torch.sqrt(color_feats[:, 1].pow(2) + color_feats[:, 2].pow(2)) / 100.0
        color_var = color_feats[:, 3:].mean(dim=-1) / 25.0

        edge_energy = shape_feats[:, 0]
        orient_energy = shape_feats[:, 2:6].abs().mean(dim=-1)
        row_peak = shape_feats[:, 6:14].amax(dim=-1)
        col_peak = shape_feats[:, 14:22].amax(dim=-1)
        periodicity = 0.5 * (row_peak + col_peak)

        color_score = color_sep + 0.60 * chroma + 0.20 * color_var
        shape_score = shape_sep + 0.60 * edge_energy + 0.35 * periodicity + 0.15 * orient_energy
        joint_score = torch.minimum(color_score, shape_score)
        return torch.softmax(torch.stack([color_score, shape_score, joint_score], dim=-1) * 2.5, dim=-1)

    @staticmethod
    def extract_lab_mean(patches: Tensor) -> Tensor:
        """Return the Lab mean colour per patch (used by LCAB, no gradient needed).

        patches: (N, 3, H, W) ∈ [0, 1]  →  (N, 3)
        """
        hwc = patches.permute(0, 2, 3, 1).clamp(0.0, 1.0)
        return _rgb_to_lab(hwc).flatten(1, 2).mean(dim=1)   # (N, 3)

    @staticmethod
    def extract_shape_templates(
        patches: Tensor,
        template_hw: Tuple[int, int] = (7, 21),
    ) -> Tensor:
        """Return normalized multi-channel structure templates for NCC matching."""
        kh = _odd_kernel_size(template_hw[0], patches.shape[-2])
        kw = _odd_kernel_size(template_hw[1], patches.shape[-1])
        templates = F.adaptive_avg_pool2d(_structural_maps(patches), output_size=(kh, kw))
        templates = templates - templates.mean(dim=(-2, -1), keepdim=True)
        norms = templates.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return templates / norms.view(-1, 1, 1, 1)

    def encode_modalities(self, patches: Tensor) -> Dict[str, Tensor]:
        """Return fused legend encodings plus modality-specific diagnostics."""
        color_raw = self._color_feats(patches)
        shape_raw = self._shape_feats(patches)
        color_emb = self.color_mlp(color_raw)
        shape_emb = self.shape_mlp(shape_raw)
        joint_emb = self.joint_mlp(torch.cat([color_emb, shape_emb], dim=-1))
        weights = self._modality_weights(color_raw, shape_raw)
        fused = (
            weights[:, 0:1] * color_emb
            + weights[:, 1:2] * shape_emb
            + weights[:, 2:3] * joint_emb
        )
        return {
            "fused": fused,
            "color_emb": color_emb,
            "shape_emb": shape_emb,
            "joint_emb": joint_emb,
            "modality_weights": weights,
            "color_feats": color_raw,
            "shape_feats": shape_raw,
        }

    def forward(self, patches: Tensor) -> Tensor:
        """Encode a batch of legend patches into query initialisation vectors.

        patches: (N, 3, H, W) ∈ [0, 1]
        Returns: (N, d_model)
        """
        return self.encode_modalities(patches)["fused"]


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


def compute_legend_shape_biases(
    legend_patches: List[Optional[Tensor]],
    images_rgb: Tensor,
    memory_hws: List[Tuple[int, int]],
    num_queries: int,
    temperature: float = 0.35,
    template_hw: Tuple[int, int] = (7, 21),
) -> List[Tensor]:
    """Compute shape-template correlation bias for each legend query."""
    B = images_rgb.shape[0]
    img_rgb01 = images_rgb.clamp(0.0, 1.0)
    biases: List[Tensor] = []

    for (h, w) in memory_hws:
        img_small = F.interpolate(img_rgb01, size=(h, w), mode="bilinear", align_corners=False)
        img_struct = _structural_maps(img_small)
        bias = images_rgb.new_zeros(B, num_queries, h * w)
        for b_idx in range(B):
            patches = legend_patches[b_idx]
            if patches is None or patches.shape[0] == 0:
                continue
            n_leg = min(patches.shape[0], num_queries)
            kh = _odd_kernel_size(template_hw[0], h)
            kw = _odd_kernel_size(template_hw[1], w)
            templates = LegendPatchEncoder.extract_shape_templates(
                patches[:n_leg].to(images_rgb.device).float().clamp(0.0, 1.0),
                template_hw=(kh, kw),
            )
            img_feat = img_struct[b_idx:b_idx + 1]
            energy_kernel = bias.new_ones(1, 1, kh, kw)
            local_energy = F.conv2d(
                (img_feat ** 2).sum(dim=1, keepdim=True),
                energy_kernel,
                padding=(kh // 2, kw // 2),
            ).sqrt().clamp_min(1e-6)
            for i_leg in range(n_leg):
                response = F.conv2d(
                    img_feat,
                    templates[i_leg:i_leg + 1],
                    padding=(kh // 2, kw // 2),
                )
                bias[b_idx, i_leg] = (response / (local_energy * temperature)).flatten(1)
        biases.append(bias)
    return biases


def compute_legend_guidance_biases(
    legend_lab_means: List[Optional[Tensor]],
    legend_patches: List[Optional[Tensor]],
    legend_modality_weights: List[Optional[Tensor]],
    images_rgb: Tensor,
    memory_hws: List[Tuple[int, int]],
    num_queries: int,
    color_temperature: float = 50.0,
    shape_temperature: float = 0.35,
    template_hw: Tuple[int, int] = (7, 21),
) -> List[Tensor]:
    """Blend colour and shape attention bias with per-legend modality weights."""
    color_biases = compute_legend_color_biases(
        legend_lab_means,
        images_rgb,
        memory_hws,
        num_queries,
        temperature=color_temperature,
    )
    shape_biases = compute_legend_shape_biases(
        legend_patches,
        images_rgb,
        memory_hws,
        num_queries,
        temperature=shape_temperature,
        template_hw=template_hw,
    )

    combined: List[Tensor] = []
    for color_bias, shape_bias in zip(color_biases, shape_biases):
        level_bias = color_bias.new_zeros(color_bias.shape)
        for b_idx, weights in enumerate(legend_modality_weights):
            if weights is None or weights.shape[0] == 0:
                continue
            n_leg = min(weights.shape[0], color_bias.shape[1])
            color_mix = weights[:n_leg, 0] + 0.5 * weights[:n_leg, 2]
            shape_mix = weights[:n_leg, 1] + 0.5 * weights[:n_leg, 2]
            level_bias[b_idx, :n_leg] = (
                color_mix.unsqueeze(-1) * color_bias[b_idx, :n_leg]
                + shape_mix.unsqueeze(-1) * shape_bias[b_idx, :n_leg]
            )
        combined.append(level_bias)
    return combined


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
