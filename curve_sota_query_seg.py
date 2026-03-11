"""
Query-Based Instance Decoder & Training Objectives (Sec 3.2 & 3.3).

Builds on the Mamba-3 Spatial Encoder (Sec 3.1) with:
  - Mask2Former-style query decoder with BATO (DI-MaskDINO) and position
    relation encoding (Relation-DETR)
  - Hungarian + one-to-many matching with denoising queries
  - Legend-guided query initialisation, attention bias, contrastive loss, gate
  - 16-term training objective in 5 clear groups (+ 2 optional disabled by default):
      Group A — Instance Query Losses (7): cls, mask, dice, quality, aux, dn_mask, otm
      Group B — Pixel Topology Losses (5): centerline, crossing, boundary, direction, grid
      Group C — Connectivity & Contrastive (2): cape, pcc
      Group D — Snake Offset (1): snake_offset alignment
      Group E — Legend Contrastive (1): legend_contrastive (active when legend_patches passed)
      Optional: topograph (ICLR 2025), efd (ICML 2025)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

import math

from mamba3_curve_instance_seg import (
    CurveMambaEncoder,
    CurveSegConfig,
    FPNDecoder,
    GridSuppressionBranch,
    PredictionHead,
    _ensure_channel_first_mask,
    cape_connectivity_loss,
    dice_loss_from_logits,
    direction_cosine_loss,
    sigmoid_focal_loss,
    snake_offset_alignment_loss,
    topograph_loss,
)
from legend_encoder import (
    LegendPatchEncoder,
    LegendQueryGate,
    compute_legend_color_biases,
    legend_contrastive_loss,
)


class BATOModule(nn.Module):
    """
    Balance-Aware Token Optimization from DI-MaskDINO (NeurIPS 2024).

    The first decoder layer in query-based segmentation suffers from
    "detection-mask imbalance": the classification branch underfits early,
    creating a quality ceiling for all later layers. BATO fixes this by
    reweighting encoder memory tokens based on query-predicted relevance,
    boosting curve-specific regions in the cross-attention key/value space.

    Applied to the first decoder layer (highest impact); all subsequent
    layers also benefit because their input queries are now better calibrated.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.m_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        """
        queries: (B, Q, D)  — current decoder query features
        memory:  (B, N, D)  — encoder memory tokens (K/V source)
        Returns: (B, N, D)  — balance-enhanced memory tokens
        """
        # Compress all queries into a single "what are we looking for?" vector
        q_mean = self.q_proj(queries.mean(dim=1, keepdim=True))   # (B, 1, D)
        m_proj = self.m_proj(memory)                               # (B, N, D)

        # Per-memory-token relevance score (dot product with mean query)
        affinity = torch.einsum("bqd,bnd->bn", q_mean, m_proj).unsqueeze(-1)  # (B, N, 1)

        # Learned gate controls how strongly each token is enhanced
        gate_val = self.gate(memory)                               # (B, N, 1)

        # Additive residual enhancement of relevant memory tokens
        return memory + torch.sigmoid(affinity) * gate_val * memory


def query_pcc_loss(
    query_feats: Tensor,
    indices: List[Tuple[Tensor, Tensor]],
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> Tensor:
    """
    Query-level Prototypical Contrastive loss, inspired by CAVIS PCC (ICCV 2025).

    Groups decoder query feature vectors by their one-to-many (OTM) GT
    assignment, then applies SupCon:
      • Queries matched to the SAME GT instance → positives (pull together)
      • Queries matched to DIFFERENT GT instances → negatives (push apart)

    This directly prevents query confusion for visually similar (overlapping
    or crossing) curve instances, which pixel-level SupCon cannot address.

    query_feats: (B, Q, D)  — final decoder query feature vectors
    indices:     List[(src_idx, tgt_idx)] per batch item (OTM indices)
    temperature: contrastive temperature (lower = sharper separation)
    """
    total = query_feats.new_tensor(0.0)
    valid_batches = 0

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() < 2:
            continue

        # Float32 for numerical stability (AMP safety)
        matched = F.normalize(query_feats[b, src_idx].float(), dim=-1)  # (M, D)
        labels = tgt_idx  # (M,) — GT index per matched query

        n = matched.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=matched.device)

        sim = torch.matmul(matched, matched.t()) / temperature            # (M, M)
        sim = sim - sim.detach().max(dim=1, keepdim=True).values          # stability

        exp_sim = torch.exp(sim)
        denom = exp_sim.masked_fill(eye, 0.0).sum(dim=1, keepdim=True).clamp_min(eps)
        log_prob = sim - torch.log(denom)                                 # (M, M)

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye   # (M, M)
        n_pos = pos_mask.float().sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            continue

        loss_b = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp_min(1.0)
        total = total + loss_b[valid].mean()
        valid_batches += 1

    if valid_batches == 0:
        return query_feats.sum() * 0.0
    return total / valid_batches


def _rgb_to_hsv(rgb: Tensor, eps: float = 1e-6) -> Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    maxc, _ = rgb.max(dim=1, keepdim=True)
    minc, _ = rgb.min(dim=1, keepdim=True)
    delta = maxc - minc

    h = torch.zeros_like(maxc)
    is_r = (maxc == r) & (delta > eps)
    is_g = (maxc == g) & (delta > eps)
    is_b = (maxc == b) & (delta > eps)
    h = torch.where(is_r, ((g - b) / (delta + eps)) % 6.0, h)
    h = torch.where(is_g, ((b - r) / (delta + eps)) + 2.0, h)
    h = torch.where(is_b, ((r - g) / (delta + eps)) + 4.0, h)
    h = h / 6.0
    s = torch.where(maxc > eps, delta / (maxc + eps), torch.zeros_like(maxc))
    v = maxc
    return torch.cat([h, s, v], dim=1)


def _sobel_features(images: Tensor) -> Tensor:
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    gm = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
    return torch.cat([gx, gy, gm], dim=1)


class QueryPositionRelation(nn.Module):
    """Position relation encoding between queries (Relation-DETR, ECCV 2024 Oral).

    Encodes pairwise spatial relationships between query reference points
    as self-attention bias, enabling queries to reason about inter-instance
    spatial arrangements (parallel curves, crossings, spatial ordering).
    """

    def __init__(self, d_model: int, nhead: int, num_pos_feats: int = 64):
        super().__init__()
        self.nhead = nhead
        self.rel_enc = nn.Sequential(
            nn.Linear(4, num_pos_feats),
            nn.GELU(),
            nn.Linear(num_pos_feats, nhead),
        )

    def forward(self, ref_points: Tensor) -> Tensor:
        """
        ref_points: (B, Q, 2) normalized reference points in [0, 1]
        Returns: (B*nhead, Q, Q) attention bias for self-attention
        """
        # Pairwise relative positions
        dx = ref_points[:, :, 0:1] - ref_points[:, :, 0:1].transpose(1, 2)
        dy = ref_points[:, :, 1:2] - ref_points[:, :, 1:2].transpose(1, 2)
        dist = (dx ** 2 + dy ** 2 + 1e-8).sqrt()
        angle = torch.atan2(dy, dx) / math.pi  # normalize to [-1, 1]

        # (B, Q, Q, 4)
        rel_feats = torch.stack([dx.squeeze(-1), dy.squeeze(-1),
                                  dist.squeeze(-1).log(), angle.squeeze(-1)], dim=-1)

        # (B, Q, Q, nhead) -> (B, nhead, Q, Q) -> (B*nhead, Q, Q)
        bias = self.rel_enc(rel_feats).permute(0, 3, 1, 2)
        b, nh, q, _ = bias.shape
        return bias.reshape(b * nh, q, q)


class QueryDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        align_topk: int = 96,
        use_query_align: bool = True,
        cross_attn_topk: int = 0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.align_topk = align_topk if use_query_align else 0
        if self.align_topk > 0:
            self.align_proj_q = nn.Linear(d_model, d_model, bias=False)
            self.align_proj_m = nn.Linear(d_model, d_model, bias=False)
            self.align_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
        # Optional sparse cross-attention: reduce KV length by keeping only
        # globally most relevant memory tokens for current query set.
        self.cross_attn_topk = max(0, int(cross_attn_topk))
        if self.cross_attn_topk > 0:
            self.cross_proj_q = nn.Linear(d_model, d_model, bias=False)
            self.cross_proj_m = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _query_memory_align(self, q: Tensor, memory: Tensor) -> Tensor:
        qn = F.normalize(self.align_proj_q(q), dim=-1, eps=1e-6)
        mn = F.normalize(self.align_proj_m(memory), dim=-1, eps=1e-6)
        sim = torch.einsum("bqd,bkd->bqk", qn, mn)
        topk = min(self.align_topk, memory.shape[1])
        if topk < memory.shape[1]:
            vals, idx = torch.topk(sim, k=topk, dim=-1)
            batch_idx = torch.arange(memory.shape[0], device=memory.device)[:, None, None]
            selected = memory[batch_idx, idx]  # (B,Q,topk,D)
            weights = vals.softmax(dim=-1)
            aligned = torch.einsum("bqk,bqkd->bqd", weights, selected)
        else:
            weights = sim.softmax(dim=-1)
            aligned = torch.einsum("bqk,bkd->bqd", weights, memory)
        return aligned

    def _sparsify_cross_memory(
        self,
        q: Tensor,
        memory: Tensor,
        cross_attn_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Keep full memory when sparse mode is disabled or already short.
        if self.cross_attn_topk <= 0 or memory.shape[1] <= self.cross_attn_topk:
            return memory, cross_attn_bias

        qn = F.normalize(self.cross_proj_q(q), dim=-1, eps=1e-6)       # (B,Q,D)
        mn = F.normalize(self.cross_proj_m(memory), dim=-1, eps=1e-6)  # (B,K,D)
        sim = torch.einsum("bqd,bkd->bqk", qn, mn)                     # (B,Q,K)
        # Global token saliency from all queries
        importance = sim.max(dim=1).values                              # (B,K)
        k_keep = min(self.cross_attn_topk, memory.shape[1])
        idx = torch.topk(importance, k=k_keep, dim=-1).indices          # (B,k_keep)

        gather_idx = idx.unsqueeze(-1).expand(-1, -1, memory.shape[-1])
        memory_sel = torch.gather(memory, dim=1, index=gather_idx)

        bias_sel: Optional[Tensor] = None
        if cross_attn_bias is not None:
            # cross_attn_bias: (B, Q, K) -> gather K dimension by selected idx
            bias_idx = idx.unsqueeze(1).expand(-1, q.shape[1], -1)
            bias_sel = torch.gather(cross_attn_bias, dim=2, index=bias_idx)
        return memory_sel, bias_sel

    def forward(self, q: Tensor, q_pos: Tensor, memory: Tensor,
                attn_bias: Optional[Tensor] = None,
                cross_attn_bias: Optional[Tensor] = None) -> Tensor:
        if self.align_topk > 0:
            aligned = self._query_memory_align(q, memory)
            q = q + self.align_gate(torch.cat([q, aligned], dim=-1)) * aligned
        q2, _ = self.self_attn(q + q_pos, q + q_pos, q, attn_mask=attn_bias)
        q = self.norm1(q + self.drop(q2))

        memory_cross, cross_bias = self._sparsify_cross_memory(q, memory, cross_attn_bias)

        # LCAB: expand colour-similarity bias to (B*nhead, Q, HW) and inject
        # into cross-attention logits so every layer stays colour-guided.
        if cross_bias is not None:
            B, Q_tot, HW = cross_bias.shape
            nhead = self.cross_attn.num_heads
            bias_exp = (cross_bias
                        .unsqueeze(1)
                        .expand(-1, nhead, -1, -1)
                        .reshape(B * nhead, Q_tot, HW)
                        .to(q.dtype))
            q2, _ = self.cross_attn(q + q_pos, memory_cross, memory_cross,
                                    attn_mask=bias_exp)
        else:
            q2, _ = self.cross_attn(q + q_pos, memory_cross, memory_cross)
        q = self.norm2(q + self.drop(q2))
        q = self.norm3(q + self.drop(self.ffn(q)))
        return q


class CurveQueryDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_queries: int,
        num_styles: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_mult: int,
        align_topk: int,
        cross_attn_topk: int,
        use_query_routing: bool,
        use_bato: bool = True,
        use_query_align: bool = True,
        use_position_relation: bool = True,
        use_style_head: bool = False,
        use_legend_queries: bool = True,
    ):
        super().__init__()
        self.use_query_routing = use_query_routing
        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList(
            [
                QueryDecoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    align_topk=align_topk,
                    use_query_align=use_query_align,
                    cross_attn_topk=cross_attn_topk,
                )
                for _ in range(num_layers)
            ]
        )
        # BATO: applied to the first decoder layer to balance detection/mask branches
        self.bato = BATOModule(d_model) if use_bato else None
        # Position relation encoding between queries (Relation-DETR, ECCV 2024)
        self.position_relation = QueryPositionRelation(d_model, num_heads) if use_position_relation else None
        # Innovation E: adaptive blend gate for legend-absent robustness
        self.legend_gate = LegendQueryGate(d_model) if use_legend_queries else None
        self.ref_point_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),  # reference points in [0, 1]
        )
        self.route_heads = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1)) for _ in range(num_layers)]
        )
        self.class_head = nn.Linear(d_model, 2)  # background / curve
        # Style head is optional: only create when style annotations exist (use_style_head=True)
        self.style_head = nn.Linear(d_model, num_styles) if use_style_head else None
        self.quality_head = nn.Linear(d_model, 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # EFD head: predict Elliptical Fourier Descriptors per query
        # (EFDTR, ICML 2025) — 10 harmonics × 4 coefficients = 40 dims
        self.efd_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 40),
        )

    def _predict(self, q: Tensor, mask_feat: Tensor, out_hw: Tuple[int, int]) -> Dict[str, Tensor]:
        pred_logits = self.class_head(q)
        pred_quality = self.quality_head(q).squeeze(-1)
        pred_efd = self.efd_head(q)
        kernel = self.mask_embed(q)
        pred_masks = torch.einsum("bqc,bchw->bqhw", kernel, mask_feat)
        if pred_masks.shape[-2:] != out_hw:
            pred_masks = F.interpolate(pred_masks, size=out_hw, mode="bilinear", align_corners=False)
        out: Dict[str, Tensor] = {
            "pred_logits": pred_logits,
            "pred_quality": pred_quality,
            "pred_efd": pred_efd,
            "pred_masks": pred_masks,
        }
        if self.style_head is not None:
            out["pred_style_logits"] = self.style_head(q)
        return out

    def forward(
        self,
        memories: Sequence[Tensor],
        mask_feat: Tensor,
        out_hw: Tuple[int, int],
        dn_q: Optional[Tensor] = None,   # (B, N_dn, C) denoising queries, training only
        content_init: Optional[Tensor] = None,  # (B, Q, C) mask-enhanced query init
        legend_init: Optional[Tensor] = None,   # (B, N_leg, C) legend query encodings (A)
        legend_valid: Optional[Tensor] = None,  # (B, N_leg) bool — valid legend items (E)
        legend_color_biases: Optional[List[Tensor]] = None,  # LCAB: 3×(B,Q,HW_k)
    ) -> Dict[str, Tensor]:
        b = mask_feat.shape[0]
        q = self.query_feat.weight.unsqueeze(0).expand(b, -1, -1).clone()
        q_pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)

        # Mask-enhanced query initialization: mix learned queries with
        # content-aware features from preliminary mask predictions
        if content_init is not None:
            q = q + content_init

        # Innovation A + E: legend query initialisation with adaptive gate.
        # Gate collapses to 0 for missing legends → safe fallback to learned q.
        if legend_init is not None and self.legend_gate is not None:
            valid = (legend_valid if legend_valid is not None
                     else legend_init.new_ones(b, legend_init.shape[1], dtype=torch.bool))
            q = self.legend_gate(legend_init, valid, q)

        # Prepend denoising queries (with zero positional encoding)
        n_dn = 0
        if dn_q is not None:
            n_dn = dn_q.shape[1]
            dn_pos = torch.zeros_like(dn_q)
            q = torch.cat([dn_q, q], dim=1)
            q_pos = torch.cat([dn_pos, q_pos], dim=1)

        # Pad LCAB with zeros for DN query positions so bias shape matches Q_total
        lcab = legend_color_biases
        if n_dn > 0 and lcab is not None:
            lcab = [
                torch.cat([b_.new_zeros(b_.shape[0], n_dn, b_.shape[2]), b_], dim=1)
                for b_ in lcab
            ]

        aux: List[Dict[str, Tensor]] = []
        for i, layer in enumerate(self.layers):
            memory = memories[i % len(memories)]
            # BATO: reweight memory tokens in the first layer (if enabled)
            if i == 0 and self.bato is not None:
                memory = self.bato(q, memory)
            # Position relation bias (if enabled)
            ref_pts = self.ref_point_head(q)  # (B, N_dn+Q, 2)
            attn_bias = self.position_relation(ref_pts) if self.position_relation is not None else None
            # LCAB: select bias for this layer's memory level
            cross_bias = lcab[i % len(lcab)] if lcab is not None else None
            q_new = layer(q, q_pos, memory, attn_bias=attn_bias,
                          cross_attn_bias=cross_bias)
            if self.use_query_routing:
                # Gated residual: each query autonomously decides how much of
                # the new layer output to incorporate (learned end-to-end)
                route = torch.sigmoid(self.route_heads[i](q))  # (B, N_dn+Q, 1)
                q = route * q_new + (1.0 - route) * q
            else:
                q = q_new
            aux.append(self._predict(q, mask_feat, out_hw))

        out = aux[-1]
        out["aux_outputs"] = aux[:-1]

        # Export final query feature vectors for query-level PCC contrastive loss
        out["query_feats"] = q  # (B, N_dn+Q, D)

        # Split denoising predictions from main predictions
        if n_dn > 0:
            _q_keys = ("pred_logits", "pred_masks", "pred_style_logits", "pred_quality", "pred_efd")
            out["dn_outputs"] = {k: out[k][:, :n_dn] for k in _q_keys if k in out}
            for k in _q_keys:
                if k in out:
                    out[k] = out[k][:, n_dn:]
            out["aux_outputs"] = [
                {k: (v[:, n_dn:] if k in _q_keys else v) for k, v in a.items()}
                for a in out["aux_outputs"]
            ]
            out["query_feats"] = out["query_feats"][:, n_dn:]  # strip DN queries

        return out


@dataclass
class CurveSOTAConfig:
    backbone: CurveSegConfig = field(default_factory=CurveSegConfig)
    use_hsv_features: bool = True
    use_gradient_features: bool = True
    num_queries: int = 240
    num_styles: int = 5
    query_layers: int = 6
    query_heads: int = 8
    query_dropout: float = 0.1
    query_ff_mult: int = 4
    align_topk: int = 96
    cross_attn_topk: int = 1024   # sparse KV selection for cross-attn (0=disable)
    use_query_routing: bool = True
    memory_bottleneck_ratio: float = 0.5
    dn_groups: int = 2          # number of denoising query groups per image
    dn_noise_scale: float = 0.4  # Gaussian noise std for DN query initialization
    # Ablation config flags (Improvement 3 & 11)
    use_bato: bool = True               # BATO memory reweighting (layer 0)
    use_query_align: bool = True         # Top-k query-memory alignment
    use_position_relation: bool = True   # Relation-DETR pairwise bias
    # Optional heads (disabled by default — enable when annotations exist)
    use_style_head: bool = False         # Per-instance curve style classification
    use_layering_head: bool = False      # Curve layering order prediction (needs layering GT)
    # Loss ablation flags — set False to ablate individual loss contributions
    use_cape_loss: bool = True           # CAPE gap + bridge connectivity (MICCAI 2025)
    use_pcc_loss: bool = True            # Query-level PCC contrastive (CAVIS, ICCV 2025)
    # Legend-guided innovations (A, LCAB, C, E)
    use_legend_queries: bool = True      # Enable all 4 legend-guided innovations
    lcab_temperature: float = 50.0       # LCAB: Lab colour similarity sharpness (larger=sharper)
    legend_contrastive_tau: float = 0.1  # C: InfoNCE temperature for legend–curve alignment


class CurveSOTAQueryNet(nn.Module):
    """
    Output (always present):
      - pred_logits: (B,Q,2)
      - pred_masks: (B,Q,H,W)
      - pred_quality: (B,Q)
      - aux_outputs: list of decoder layer predictions
      - centerline_logits, crossing_logits, boundary_logits, direction_vectors, grid_logits

    Output (optional, when enabled in CurveSOTAConfig):
      - pred_style_logits: (B,Q,S)   — requires use_style_head=True
      - layering_logits: (B,1,H,W)   — requires use_layering_head=True
    """

    def __init__(self, cfg: CurveSOTAConfig = CurveSOTAConfig()):
        super().__init__()
        self.cfg = cfg
        extra_in = (3 if cfg.use_hsv_features else 0) + (3 if cfg.use_gradient_features else 0)
        self.backbone_cfg = replace(cfg.backbone, in_channels=cfg.backbone.in_channels + extra_in)
        self.encoder = CurveMambaEncoder(self.backbone_cfg)
        self.decoder = FPNDecoder(
            self.backbone_cfg.encoder_dims, self.backbone_cfg.decoder_dim,
            stem_dim=self.backbone_cfg.encoder_dims[0] // 2,  # stem_half channels
        )
        self.grid_branch = GridSuppressionBranch(
            (
                self.backbone_cfg.encoder_dims[0],
                self.backbone_cfg.encoder_dims[1],
                self.backbone_cfg.encoder_dims[2],
            ),
            decoder_dim=self.backbone_cfg.decoder_dim,
        )
        # Learnable scale for grid additive bias (starts at 0 for stable init, M5)
        self.grid_scale = nn.Parameter(torch.zeros(1))
        self.query_decoder = CurveQueryDecoder(
            d_model=self.backbone_cfg.decoder_dim,
            num_queries=cfg.num_queries,
            num_styles=cfg.num_styles,
            num_layers=cfg.query_layers,
            num_heads=cfg.query_heads,
            dropout=cfg.query_dropout,
            ff_mult=cfg.query_ff_mult,
            align_topk=cfg.align_topk,
            cross_attn_topk=cfg.cross_attn_topk,
            use_query_routing=cfg.use_query_routing,
            use_bato=cfg.use_bato,
            use_query_align=cfg.use_query_align,
            use_position_relation=cfg.use_position_relation,
            use_style_head=cfg.use_style_head,
            use_legend_queries=cfg.use_legend_queries,
        )
        # Innovation A: legend patch encoder (Lab colour + FFT line-style → d_model)
        self.legend_encoder = (
            LegendPatchEncoder(self.backbone_cfg.decoder_dim)
            if cfg.use_legend_queries else None
        )
        bottleneck_dim = max(16, int(self.backbone_cfg.decoder_dim * cfg.memory_bottleneck_ratio))
        self.memory_proj = nn.ModuleList(
            [
                nn.Conv2d(self.backbone_cfg.encoder_dims[3], self.backbone_cfg.decoder_dim, 1, bias=False),
                nn.Conv2d(self.backbone_cfg.encoder_dims[2], self.backbone_cfg.decoder_dim, 1, bias=False),
                nn.Conv2d(self.backbone_cfg.encoder_dims[1], self.backbone_cfg.decoder_dim, 1, bias=False),
            ]
        )
        self.memory_bottleneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.backbone_cfg.decoder_dim, bottleneck_dim, 1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(bottleneck_dim, self.backbone_cfg.decoder_dim, 1, bias=False),
                )
                for _ in range(3)
            ]
        )
        self.centerline_head = PredictionHead(self.backbone_cfg.decoder_dim, 1)
        self.crossing_head = PredictionHead(self.backbone_cfg.decoder_dim, 1)
        self.boundary_head = PredictionHead(self.backbone_cfg.decoder_dim, 1)
        self.direction_head = PredictionHead(self.backbone_cfg.decoder_dim, 4)
        # Layering head is optional: only meaningful when layering_target annotations exist
        self.layering_head = PredictionHead(self.backbone_cfg.decoder_dim, 1) if cfg.use_layering_head else None

        # Mask-enhanced query initialization: predict preliminary attention
        # map over spatial features, then pool to produce content-aware queries
        d = self.backbone_cfg.decoder_dim
        self.query_init_scorer = nn.Sequential(
            nn.Conv2d(d, d // 2, 3, padding=1, bias=False),
            nn.GroupNorm(4, d // 2),
            nn.GELU(),
            nn.Conv2d(d // 2, cfg.num_queries, 1),
        )
        self.query_init_proj = nn.Linear(d, d)

    def _augment_input(self, images: Tensor) -> Tensor:
        feats = [images]
        if self.cfg.use_hsv_features:
            feats.append(_rgb_to_hsv(images))
        if self.cfg.use_gradient_features:
            feats.append(_sobel_features(images))
        return torch.cat(feats, dim=1)

    def _make_dn_queries(
        self,
        mask_feat: Tensor,                        # (B, C, H, W)
        instance_targets: List[Dict[str, Tensor]],
    ) -> Tuple[Optional[Tensor], Optional[dict]]:
        """Create denoising queries from GT masks via masked-avg-pool + Gaussian noise."""
        b, c, h, w = mask_feat.shape
        device = mask_feat.device
        n_gts = [t["masks"].shape[0] if "masks" in t and t["masks"].numel() > 0 else 0
                 for t in instance_targets]
        max_n = max(n_gts) if n_gts else 0
        if max_n == 0:
            return None, None

        groups = self.cfg.dn_groups
        scale = self.cfg.dn_noise_scale
        all_q: List[Tensor] = []

        for i in range(b):
            if n_gts[i] == 0:
                all_q.append(torch.zeros(groups * max_n, c, device=device))
                continue
            gt = instance_targets[i]["masks"].float().to(device)   # (N, H, W)
            if gt.shape[-2:] != (h, w):
                gt = F.interpolate(gt.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
            # Masked average pool: (N, C)
            m = gt.unsqueeze(1)                                     # (N, 1, H, W)
            feat = mask_feat[i].unsqueeze(0)                        # (1, C, H, W)
            base = (m * feat).flatten(2).sum(2) / m.flatten(2).sum(2).clamp_min(1.0)
            if n_gts[i] < max_n:
                pad = torch.zeros(max_n - n_gts[i], c, device=device)
                base = torch.cat([base, pad], dim=0)
            # Repeat with different noise for each group
            group_qs = [base + torch.randn_like(base) * scale for _ in range(groups)]
            all_q.append(torch.cat(group_qs, dim=0))               # (groups*max_n, C)

        dn_q = torch.stack(all_q, dim=0)                           # (B, groups*max_n, C)
        dn_meta = {"n_gts": n_gts, "max_n": max_n, "groups": groups}
        return dn_q, dn_meta

    def forward(
        self,
        images: Tensor,
        instance_targets: Optional[List[Dict[str, Tensor]]] = None,  # for DN during training
        legend_patches: Optional[List[Optional[Tensor]]] = None,     # B × Optional[(N_i,3,H,W)]
    ) -> Dict[str, Tensor]:
        out_hw = images.shape[-2:]
        x = self._augment_input(images)
        f1, f2, f3, f4, stem_half, snake_offsets = self.encoder(x)
        fused = self.decoder((f1, f2, f3, f4), stem_feat=stem_half)

        # Additive grid bias with learnable scale (M5)
        grid_bias, grid_logits_low = self.grid_branch((f1, f2, f3))
        if grid_bias.shape[-2:] != fused.shape[-2:]:
            grid_bias = F.interpolate(grid_bias, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        fused = fused + self.grid_scale * grid_bias

        memories = [
            rearrange(self.memory_bottleneck[0](self.memory_proj[0](f4)), "b c h w -> b (h w) c"),
            rearrange(self.memory_bottleneck[1](self.memory_proj[1](f3)), "b c h w -> b (h w) c"),
            rearrange(self.memory_bottleneck[2](self.memory_proj[2](f2)), "b c h w -> b (h w) c"),
        ]

        dn_q, dn_meta = None, None
        if self.training and instance_targets is not None and self.cfg.dn_groups > 0:
            dn_q, dn_meta = self._make_dn_queries(fused, instance_targets)

        # Mask-enhanced query initialization: compute per-query attention
        # maps over spatial features, then weighted-avg-pool to init queries
        attn_maps = self.query_init_scorer(fused)  # (B, Q, H', W')
        attn_maps = attn_maps.flatten(2).softmax(dim=2)  # (B, Q, H'W')
        feat_flat = rearrange(fused, "b c h w -> b c (h w)")  # (B, C, H'W')
        content_init = torch.einsum("bqn,bcn->bqc", attn_maps, feat_flat)
        content_init = self.query_init_proj(content_init)  # (B, Q, C)

        # ── Legend-guided innovations (A, LCAB, C, E) ────────────────────────
        legend_init: Optional[Tensor] = None
        legend_valid: Optional[Tensor] = None
        legend_color_biases: Optional[List[Tensor]] = None
        legend_feats_out: Optional[Tensor] = None   # exposed for criterion (loss C)

        if legend_patches is not None and self.legend_encoder is not None:
            B = images.shape[0]
            device = images.device
            d = self.backbone_cfg.decoder_dim
            # Maximum legend count across batch (for padding)
            n_legs = [p.shape[0] if p is not None and p.numel() > 0 else 0
                      for p in legend_patches]
            max_n = max(n_legs) if n_legs else 0

            if max_n > 0:
                leg_q = images.new_zeros(B, max_n, d)           # (B, N_max, D)
                leg_v = images.new_zeros(B, max_n, dtype=torch.bool)  # valid mask
                lab_means: List[Optional[Tensor]] = []

                # Batch device transfer once to minimise CPU↔GPU round-trips
                legend_patches = [
                    p.to(device) if p is not None else None
                    for p in legend_patches
                ]

                for b_idx, (patches_i, n_i) in enumerate(zip(legend_patches, n_legs)):
                    if n_i == 0:
                        lab_means.append(None)
                        continue
                    p = patches_i.float().clamp(0.0, 1.0)  # already on device
                    enc = self.legend_encoder(p)                 # (N_i, D)
                    leg_q[b_idx, :n_i] = enc
                    leg_v[b_idx, :n_i] = True
                    lab_means.append(
                        LegendPatchEncoder.extract_lab_mean(p)   # (N_i, 3)
                    )

                legend_init = leg_q
                legend_valid = leg_v
                legend_feats_out = leg_q                        # (B, N_max, D) for loss C

                # LCAB: precompute colour biases at 3 memory resolutions
                memory_hws = [
                    (f4.shape[-2], f4.shape[-1]),
                    (f3.shape[-2], f3.shape[-1]),
                    (f2.shape[-2], f2.shape[-1]),
                ]
                legend_color_biases = compute_legend_color_biases(
                    lab_means, images.clamp(0.0, 1.0),
                    memory_hws, self.cfg.num_queries,
                    self.cfg.lcab_temperature,
                )
        # ─────────────────────────────────────────────────────────────────────

        qout = self.query_decoder(
            memories, fused, out_hw, dn_q=dn_q,
            content_init=content_init,
            legend_init=legend_init,
            legend_valid=legend_valid,
            legend_color_biases=legend_color_biases,
        )
        result: Dict[str, Tensor] = {
            "pred_logits": qout["pred_logits"],
            "pred_masks": qout["pred_masks"],
            "pred_quality": qout["pred_quality"],
            "pred_efd": qout["pred_efd"],
            "aux_outputs": qout["aux_outputs"],
            "query_feats": qout["query_feats"],   # needed for PCC contrastive loss
            "centerline_logits": self.centerline_head(fused, out_hw),
            "crossing_logits": self.crossing_head(fused, out_hw),
            "boundary_logits": self.boundary_head(fused, out_hw),
            "direction_vectors": self.direction_head(fused, out_hw),
            "grid_logits": F.interpolate(grid_logits_low, size=out_hw, mode="bilinear", align_corners=False),
        }
        # Optional outputs: only included when the corresponding head is enabled
        if "pred_style_logits" in qout:
            result["pred_style_logits"] = qout["pred_style_logits"]
        if self.layering_head is not None:
            result["layering_logits"] = self.layering_head(fused, out_hw)
        # Expose snake offsets for offset alignment loss (DDP-safe: from return value)
        result["snake_offsets"] = snake_offsets
        if dn_meta is not None:
            result["dn_outputs"] = qout.get("dn_outputs", {})
            result["dn_meta"] = dn_meta   # type: ignore[assignment]
        # Legend feats for contrastive loss (Group E, Innovation C)
        # Included only when legend_patches was provided to this forward call
        if legend_feats_out is not None:
            result["legend_feats"] = legend_feats_out   # (B, N_max, D)
            result["legend_valid"] = legend_valid         # (B, N_max) bool
        return result


def _hungarian_assign(cost: Tensor) -> Tuple[Tensor, Tensor]:
    """Optimal bipartite matching via scipy (C implementation, ~50× faster than pure Python)."""
    if cost.numel() == 0:
        z = torch.empty(0, dtype=torch.long, device=cost.device)
        return z, z
    from scipy.optimize import linear_sum_assignment
    matrix = cost.detach().float().cpu().numpy()
    rows, cols = linear_sum_assignment(matrix)
    return (
        torch.tensor(rows, dtype=torch.long, device=cost.device),
        torch.tensor(cols, dtype=torch.long, device=cost.device),
    )


class OneToManyMatcher(nn.Module):
    """
    For each GT instance, selects the top-K scoring predicted queries as positives
    (one-to-many assignment). Used as an auxiliary training signal alongside the
    main one-to-one Hungarian matching to provide denser supervision.
    """

    def __init__(self, top_k: int = 4):
        super().__init__()
        self.top_k = top_k

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        pred_masks = torch.sigmoid(outputs["pred_masks"])          # (B, Q, H, W)
        cls_scores = outputs["pred_logits"].softmax(-1)[..., 1]    # (B, Q)
        results: List[Tuple[Tensor, Tensor]] = []
        for b_idx in range(pred_masks.shape[0]):
            gt = targets[b_idx]["masks"].float().to(pred_masks.device)
            if gt.numel() == 0:
                z = torch.empty(0, dtype=torch.long, device=pred_masks.device)
                results.append((z, z))
                continue
            if gt.shape[-2:] != pred_masks.shape[-2:]:
                gt = F.interpolate(gt.unsqueeze(1), size=pred_masks.shape[-2:], mode="nearest").squeeze(1)
            pm = pred_masks[b_idx].flatten(1)          # (Q, HW)
            tm = gt.flatten(1)                          # (N, HW)
            inter = torch.einsum("qd,nd->qn", pm, tm)        # (Q, N)
            union = pm.sum(1, keepdim=True) + tm.sum(1).unsqueeze(0) - inter  # (Q,1)+(1,N)-(Q,N)
            iou = inter / (union + 1e-6)               # (Q, N)
            combined = iou * cls_scores[b_idx].unsqueeze(1)
            src_list, tgt_list = [], []
            assigned: set = set()
            for g in range(gt.shape[0]):
                k = min(self.top_k, pred_masks.shape[1])
                topk_q = torch.topk(combined[:, g], k).indices
                # Each query assigned to at most one GT (keep first assignment)
                unique_q = [qi for qi in topk_q.tolist() if qi not in assigned]
                assigned.update(unique_q)
                if unique_q:
                    src_list.append(torch.tensor(unique_q, dtype=torch.long, device=pred_masks.device))
                    tgt_list.append(torch.full((len(unique_q),), g, dtype=torch.long, device=pred_masks.device))
            if src_list:
                results.append((torch.cat(src_list), torch.cat(tgt_list)))
            else:
                z = torch.empty(0, dtype=torch.long, device=pred_masks.device)
                results.append((z, z))
        return results


@dataclass
class MatcherWeights:
    cls: float = 2.0
    mask: float = 5.0
    dice: float = 5.0
    style: float = 1.0


class HungarianCurveMatcher(nn.Module):
    def __init__(self, weights: MatcherWeights = MatcherWeights()):
        super().__init__()
        self.weights = weights

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        pred_style = outputs.get("pred_style_logits", None)
        result: List[Tuple[Tensor, Tensor]] = []
        eps = 1e-6
        for b in range(pred_logits.shape[0]):
            tgt_masks = targets[b]["masks"].float().to(pred_masks.device)
            if tgt_masks.numel() == 0:
                z = torch.empty(0, dtype=torch.long, device=pred_masks.device)
                result.append((z, z))
                continue
            if tgt_masks.shape[-2:] != pred_masks.shape[-2:]:
                tgt_masks = F.interpolate(tgt_masks.unsqueeze(1), size=pred_masks.shape[-2:], mode="nearest").squeeze(1)

            cost_cls = -pred_logits[b].softmax(-1)[:, 1].unsqueeze(1).expand(-1, tgt_masks.shape[0])
            pm = torch.sigmoid(pred_masks[b]).flatten(1).clamp(eps, 1.0 - eps)
            tm = tgt_masks.flatten(1)
            pm_e = pm[:, None, :]
            tm_e = tm[None, :, :]
            cost_mask = -(tm_e * torch.log(pm_e) + (1 - tm_e) * torch.log(1 - pm_e)).mean(-1)
            inter = (pm_e * tm_e).sum(-1)
            denom = pm_e.sum(-1) + tm_e.sum(-1)
            cost_dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
            cost = self.weights.cls * cost_cls + self.weights.mask * cost_mask + self.weights.dice * cost_dice

            if pred_style is not None and "styles" in targets[b]:
                style_prob = pred_style[b].softmax(-1)
                style_ids = targets[b]["styles"].long().to(style_prob.device).clamp(0, style_prob.shape[-1] - 1)
                cost = cost + self.weights.style * (-style_prob[:, style_ids])

            cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)
            result.append(_hungarian_assign(cost))
        return result


def _mask_to_efd(mask: Tensor, n_harmonics: int = 10) -> Optional[Tensor]:
    """Extract Elliptical Fourier Descriptors from a binary mask contour.

    Uses cv2.findContours for efficient boundary extraction, then computes
    the first n_harmonics Fourier coefficient pairs (a_n, b_n, c_n, d_n)
    following the Kuhl-Giardina parameterization.

    Returns: (4*n_harmonics,) tensor or None if no valid contour found.
    """
    import cv2 as _cv2
    m = (mask > 0.5).cpu().numpy().astype("uint8")
    contours, _ = _cv2.findContours(m, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    coords = max(contours, key=len).squeeze(1).tolist()  # largest contour
    if len(coords) < 8:
        return None

    pts = torch.tensor(coords, dtype=torch.float32)  # (N, 2)
    n = pts.shape[0]

    # Compute cumulative arc-length parameter
    diffs = pts[1:] - pts[:-1]
    diffs = torch.cat([diffs, (pts[0] - pts[-1]).unsqueeze(0)], dim=0)
    dt = diffs.norm(dim=1).clamp_min(1e-6)
    t = torch.cumsum(dt, dim=0)
    T = t[-1]

    # Fourier coefficients
    coeffs = []
    for k in range(1, n_harmonics + 1):
        factor = T / (2.0 * k * k * math.pi * math.pi)
        cos_term = torch.cos(2.0 * k * math.pi * t / T)
        cos_prev = torch.cos(2.0 * k * math.pi * (t - dt) / T)
        sin_term = torch.sin(2.0 * k * math.pi * t / T)
        sin_prev = torch.sin(2.0 * k * math.pi * (t - dt) / T)

        dx = diffs[:, 0] / dt
        dy = diffs[:, 1] / dt

        a_k = factor * (dx * (cos_term - cos_prev)).sum()
        b_k = factor * (dx * (sin_term - sin_prev)).sum()
        c_k = factor * (dy * (cos_term - cos_prev)).sum()
        d_k = factor * (dy * (sin_term - sin_prev)).sum()
        coeffs.extend([a_k, b_k, c_k, d_k])

    result = torch.stack(coeffs)
    # Normalize by first harmonic magnitude for scale invariance
    mag = (result[0] ** 2 + result[1] ** 2 + result[2] ** 2 + result[3] ** 2).sqrt().clamp_min(1e-6)
    return result / mag


@dataclass
class SOTALossWeights:
    """Loss weights for 16-term objective, organised into 5 groups.

    Group A — Instance Query Losses (7): cls, mask, dice, quality, aux, dn_mask, otm
    Group B — Pixel Topology Losses (5): centerline, crossing, boundary, direction, grid
    Group C — Connectivity & Contrastive (2): cape, pcc
    Group D — Snake Offset (1): snake_offset alignment
    Group E — Legend Contrastive (1): legend_contrastive (active when legend_patches passed)
    """
    no_object: float = 0.1
    # --- Group A: Instance Query Losses ---
    cls: float = 2.0
    mask: float = 5.0
    dice: float = 5.0
    quality: float = 0.6
    aux: float = 0.5
    dn_mask: float = 1.0        # denoising query supervision
    otm: float = 0.5            # one-to-many auxiliary matching
    # --- Group B: Pixel Topology Losses ---
    centerline: float = 0.8
    crossing: float = 0.6
    boundary: float = 0.6
    direction: float = 0.3
    grid: float = 0.3          # grid/background auxiliary loss (supervises grid_logits_conv)
    # --- Group C: Connectivity & Contrastive ---
    cape: float = 0.2           # CAPE gap + bridge connectivity (MICCAI 2025)
    pcc: float = 0.3            # query-level PCC contrastive (CAVIS, ICCV 2025)
    # --- Group D: Snake Offset ---
    snake_offset: float = 0.1   # Snake offset alignment with GT tangent direction
    # --- Group E: Legend Contrastive ---
    legend_contrastive: float = 0.2  # InfoNCE legend–curve alignment; 0 when no legend data
    # --- Optional (disabled by default) ---
    topograph: float = 0.0      # Topograph multi-threshold topology (ICLR 2025); set >0 to enable
    efd: float = 0.0            # Elliptical Fourier Descriptor (EFDTR, ICML 2025); set >0 to enable


class CurveSOTACriterion(nn.Module):
    def __init__(
        self,
        weights: SOTALossWeights = SOTALossWeights(),
        matcher_weights: MatcherWeights = MatcherWeights(),
        legend_contrastive_tau: float = 0.1,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.weights = weights
        self.legend_contrastive_tau = legend_contrastive_tau
        self.matcher = HungarianCurveMatcher(matcher_weights)
        self.one_to_many_matcher = OneToManyMatcher(top_k=4)
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self._loss_keys = (
            "cls", "mask", "dice", "quality", "aux", "dn_mask", "otm",
            "centerline", "crossing", "boundary", "direction", "grid",
            "cape", "pcc", "snake_offset", "legend_contrastive",
            "topograph", "efd",
        )
        if self.use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1)) for k in self._loss_keys
            })

    @staticmethod
    def _dict_to_instance_targets(targets: Dict[str, Tensor], batch_size: int, device: torch.device) -> List[Dict[str, Tensor]]:
        if "instance_ids" not in targets:
            raise ValueError("targets dict mode requires `instance_ids` of shape (B,H,W)")
        ids = targets["instance_ids"].to(device).long()
        out: List[Dict[str, Tensor]] = []
        for b in range(batch_size):
            uids = torch.unique(ids[b])
            uids = uids[uids > 0]
            if uids.numel() == 0:
                h, w = ids.shape[-2:]
                out.append({"masks": torch.zeros(0, h, w, device=device), "labels": torch.zeros(0, dtype=torch.long, device=device)})
                continue
            masks = torch.stack([(ids[b] == uid).float() for uid in uids], dim=0)
            item = {"masks": masks, "labels": torch.ones(uids.numel(), dtype=torch.long, device=device)}
            if "instance_styles" in targets:
                style_map = targets["instance_styles"].to(device).long()
                styles = []
                for uid in uids:
                    pix = style_map[b][ids[b] == uid]
                    styles.append(torch.mode(pix).values if pix.numel() else torch.tensor(0, device=device))
                item["styles"] = torch.stack(styles).long()
            out.append(item)
        return out

    @staticmethod
    def _get_matched_masks(
        pred_masks: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        src, tgt = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src.append(pred_masks[b, src_idx])
            tgt.append(targets[b]["masks"][tgt_idx])
        if not src:
            return None, None
        src_t = torch.cat(src, dim=0)
        tgt_t = torch.cat(tgt, dim=0)
        if src_t.shape[-2:] != tgt_t.shape[-2:]:
            tgt_t = F.interpolate(tgt_t.unsqueeze(1), size=src_t.shape[-2:], mode="nearest").squeeze(1)
        return src_t, tgt_t

    def _loss_labels(self, pred_logits: Tensor, indices: List[Tuple[Tensor, Tensor]]) -> Tensor:
        bsz, q, _ = pred_logits.shape
        target_classes = torch.zeros((bsz, q), dtype=torch.long, device=pred_logits.device)
        for b, (src_idx, _) in enumerate(indices):
            if src_idx.numel():
                target_classes[b, src_idx] = 1
        class_weight = pred_logits.new_tensor([self.weights.no_object, 1.0])
        return F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=class_weight)

    def _loss_masks(
        self,
        pred_masks: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        src, tgt = self._get_matched_masks(pred_masks, targets, indices)
        if src is None or tgt is None:
            z = pred_masks.sum() * 0.0
            return z, z
        return sigmoid_focal_loss(src.unsqueeze(1), tgt.unsqueeze(1)), dice_loss_from_logits(src.unsqueeze(1), tgt.unsqueeze(1))

    def _loss_quality(
        self,
        pred_quality: Optional[Tensor],
        pred_masks: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        if pred_quality is None:
            return pred_masks.sum() * 0.0
        src_mask, tgt_mask = self._get_matched_masks(pred_masks, targets, indices)
        if src_mask is None or tgt_mask is None:
            return pred_quality.sum() * 0.0

        src_q: List[Tensor] = []
        for b, (src_idx, _) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            src_q.append(pred_quality[b, src_idx])
        if not src_q:
            return pred_quality.sum() * 0.0
        q_pred = torch.cat(src_q, dim=0)

        probs = torch.sigmoid(src_mask)
        tgt = tgt_mask.float()
        inter = (probs * tgt).flatten(1).sum(dim=1)
        union = (probs + tgt - probs * tgt).flatten(1).sum(dim=1)
        iou_target = ((inter + 1e-6) / (union + 1e-6)).clamp(0.0, 1.0)
        return F.binary_cross_entropy_with_logits(q_pred, iou_target)

    @staticmethod
    def _boundary_from_instance_ids(instance_ids: Tensor) -> Tensor:
        # boundary pixels are where adjacent labels differ.
        x = instance_ids.long()
        up = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0)
        down = F.pad(x[:, 1:, :], (0, 0, 0, 1), value=0)
        left = F.pad(x[:, :, :-1], (1, 0, 0, 0), value=0)
        right = F.pad(x[:, :, 1:], (0, 1, 0, 0), value=0)
        boundary = ((x != up) | (x != down) | (x != left) | (x != right)) & (x > 0)
        return boundary.float()

    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]] | Dict[str, Tensor]) -> Dict[str, Tensor]:
        device = outputs["pred_logits"].device
        bsz = outputs["pred_logits"].shape[0]
        instance_ids_map: Optional[Tensor] = None
        if isinstance(targets, dict):
            inst_targets = self._dict_to_instance_targets(targets, bsz, device)
            pixel_targets = {k: v.to(device) for k, v in targets.items() if k != "instance_ids"}
            if "instance_ids" in targets:
                instance_ids_map = targets["instance_ids"].to(device)
        else:
            inst_targets = targets
            pixel_targets = {}
            for key in (
                "centerline_mask",
                "crossing_mask",
                "junction_mask",
                "boundary_mask",
                "direction_vectors",
            ):
                if all(key in t for t in targets):
                    pixel_targets[key] = torch.stack([t[key].to(device) for t in targets], dim=0)

        indices = self.matcher(outputs, inst_targets)
        losses: Dict[str, Tensor] = {}

        # ═══ Group A: Instance Query Losses ═══

        losses["cls"] = self._loss_labels(outputs["pred_logits"], indices)
        losses["mask"], losses["dice"] = self._loss_masks(outputs["pred_masks"], inst_targets, indices)
        losses["quality"] = self._loss_quality(
            outputs.get("pred_quality"),
            outputs["pred_masks"],
            inst_targets,
            indices,
        )

        # Deep supervision from intermediate decoder layers (cls+mask+dice averaged)
        aux_cls = outputs["pred_logits"].sum() * 0.0
        aux_mask = outputs["pred_logits"].sum() * 0.0
        aux_dice = outputs["pred_logits"].sum() * 0.0
        n_aux = 0
        for aux in outputs.get("aux_outputs", []):
            aux_cls = aux_cls + self._loss_labels(aux["pred_logits"], indices)
            m, d = self._loss_masks(aux["pred_masks"], inst_targets, indices)
            aux_mask = aux_mask + m
            aux_dice = aux_dice + d
            n_aux += 1
        if n_aux > 0:
            aux_cls = aux_cls / n_aux
            aux_mask = aux_mask / n_aux
            aux_dice = aux_dice / n_aux
        losses["aux"] = aux_cls + aux_mask + aux_dice

        # One-to-many auxiliary matching (denser supervision per GT)
        otm_indices = self.one_to_many_matcher(outputs, inst_targets)
        otm_m, otm_d = self._loss_masks(outputs["pred_masks"], inst_targets, otm_indices)
        losses["otm"] = otm_m + otm_d

        # Denoising query direct supervision (no Hungarian matching needed)
        if "dn_outputs" in outputs and "dn_meta" in outputs:
            losses["dn_mask"] = self._compute_dn_loss(
                outputs["dn_outputs"], inst_targets, outputs["dn_meta"]
            )
        else:
            losses["dn_mask"] = outputs["pred_logits"].sum() * 0.0

        # ═══ Group B: Pixel Topology Losses ═══

        if "centerline_mask" in pixel_targets:
            center_t = _ensure_channel_first_mask(pixel_targets["centerline_mask"]).to(device)
            losses["centerline"] = dice_loss_from_logits(outputs["centerline_logits"], center_t) + sigmoid_focal_loss(outputs["centerline_logits"], center_t)
        else:
            losses["centerline"] = outputs["centerline_logits"].sum() * 0.0

        crossing_key = "crossing_mask" if "crossing_mask" in pixel_targets else "junction_mask"
        if crossing_key in pixel_targets:
            cross_t = _ensure_channel_first_mask(pixel_targets[crossing_key]).to(device)
            losses["crossing"] = dice_loss_from_logits(outputs["crossing_logits"], cross_t) + sigmoid_focal_loss(outputs["crossing_logits"], cross_t, alpha=0.75)
        else:
            losses["crossing"] = outputs["crossing_logits"].sum() * 0.0

        if "boundary_mask" in pixel_targets:
            boundary_t = _ensure_channel_first_mask(pixel_targets["boundary_mask"]).to(device)
            losses["boundary"] = dice_loss_from_logits(outputs["boundary_logits"], boundary_t) + sigmoid_focal_loss(outputs["boundary_logits"], boundary_t, alpha=0.75)
        elif instance_ids_map is not None:
            boundary_t = self._boundary_from_instance_ids(instance_ids_map).unsqueeze(1)
            losses["boundary"] = dice_loss_from_logits(outputs["boundary_logits"], boundary_t) + sigmoid_focal_loss(outputs["boundary_logits"], boundary_t, alpha=0.75)
        else:
            losses["boundary"] = outputs["boundary_logits"].sum() * 0.0

        if "direction_vectors" in pixel_targets and "centerline_mask" in pixel_targets:
            dir_t = pixel_targets["direction_vectors"].to(device).float()
            valid = (_ensure_channel_first_mask(pixel_targets["centerline_mask"]).to(device) > 0.5).float()
            losses["direction"] = direction_cosine_loss(
                outputs["direction_vectors"][:, :2], dir_t[:, :2], valid
            )
            if dir_t.shape[1] >= 4 and crossing_key in pixel_targets:
                valid_cross = (_ensure_channel_first_mask(pixel_targets[crossing_key]).to(device) > 0.5).float()
                losses["direction"] = losses["direction"] + 0.5 * direction_cosine_loss(
                    outputs["direction_vectors"][:, 2:], dir_t[:, 2:], valid_cross
                )
        else:
            losses["direction"] = outputs["direction_vectors"].sum() * 0.0

        # Grid auxiliary loss: use real grid_mask if available, else fall back to
        # background from instance_ids (unified with base model behavior, item 3).
        if "grid_logits" in outputs:
            if "grid_mask" in pixel_targets and pixel_targets["grid_mask"].sum() > 0:
                grid_target = _ensure_channel_first_mask(pixel_targets["grid_mask"]).to(device)
            elif instance_ids_map is not None:
                grid_target = (instance_ids_map == 0).float().unsqueeze(1)
            else:
                grid_target = None
            if grid_target is not None:
                losses["grid"] = sigmoid_focal_loss(
                    outputs["grid_logits"], grid_target, alpha=0.25
                )
            else:
                losses["grid"] = outputs["pred_logits"].sum() * 0.0
        else:
            losses["grid"] = outputs["pred_logits"].sum() * 0.0

        # ═══ Group C: Connectivity & Contrastive ═══

        # CAPE: skeleton recall (gap) + inter-instance bridge penalty (MICCAI 2025)
        if instance_ids_map is not None and "centerline_mask" in pixel_targets:
            center_t_cape = _ensure_channel_first_mask(pixel_targets["centerline_mask"]).to(device)
            pred_center_probs = torch.sigmoid(outputs["centerline_logits"])
            losses["cape"] = cape_connectivity_loss(
                pred_center_probs, center_t_cape, instance_ids_map
            )
        else:
            losses["cape"] = outputs["centerline_logits"].sum() * 0.0

        # Query-level PCC contrastive: pull same-GT queries, push different-GT (CAVIS, ICCV 2025)
        if "query_feats" in outputs and any(s.numel() > 0 for s, _ in otm_indices):
            losses["pcc"] = query_pcc_loss(outputs["query_feats"], otm_indices)
        else:
            losses["pcc"] = outputs["pred_logits"].sum() * 0.0

        # ═══ Group D: Snake Offset ═══

        # Snake offset alignment loss (Improvement 2)
        if (
            "snake_offsets" in outputs
            and self.weights.snake_offset > 0
            and "direction_vectors" in pixel_targets
            and "centerline_mask" in pixel_targets
        ):
            dir_t_snake = pixel_targets["direction_vectors"].to(device).float()
            valid_snake = (_ensure_channel_first_mask(pixel_targets["centerline_mask"]).to(device) > 0.5).float()
            losses["snake_offset"] = snake_offset_alignment_loss(
                outputs["snake_offsets"], dir_t_snake[:, :2], valid_snake
            )
        else:
            losses["snake_offset"] = outputs["pred_logits"].sum() * 0.0

        # ═══ Optional losses (disabled by default, weight=0.0) ═══

        # Topograph: multi-threshold topology consistency (ICLR 2025)
        if self.weights.topograph > 0 and "centerline_mask" in pixel_targets:
            center_t_topo = _ensure_channel_first_mask(pixel_targets["centerline_mask"]).to(device)
            losses["topograph"] = topograph_loss(
                torch.sigmoid(outputs["centerline_logits"]), center_t_topo
            )
        else:
            losses["topograph"] = outputs["centerline_logits"].sum() * 0.0

        # EFD: Elliptical Fourier Descriptor contour regularization (EFDTR, ICML 2025)
        if self.weights.efd > 0 and "pred_efd" in outputs:
            losses["efd"] = self._compute_efd_loss(
                outputs["pred_efd"], inst_targets, indices
            )
        else:
            losses["efd"] = outputs["pred_logits"].sum() * 0.0

        # ═══ Group E: Legend Contrastive (Innovation C) ═══

        # Bidirectional InfoNCE aligning legend patch encodings with matched
        # decoder query features.  Active only when legend_patches was passed
        # to the model forward; zero otherwise (no gradient, no overhead).
        if (
            "legend_feats" in outputs
            and "legend_valid" in outputs
            and "query_feats" in outputs
        ):
            leg_list: List[Tensor] = []
            qry_list: List[Tensor] = []
            leg_feats = outputs["legend_feats"]   # (B, N_max, D)
            leg_valid = outputs["legend_valid"]   # (B, N_max) bool
            qry_feats = outputs["query_feats"]    # (B, Q, D)

            for b, (src_idx, tgt_idx) in enumerate(indices):
                if src_idx.numel() == 0:
                    continue
                n_leg_b = int(leg_valid[b].sum().item())
                for sq, tq in zip(src_idx.tolist(), tgt_idx.tolist()):
                    if tq < n_leg_b and leg_valid[b, tq]:
                        leg_list.append(leg_feats[b, tq])
                        qry_list.append(qry_feats[b, sq])

            if len(leg_list) >= 2:
                losses["legend_contrastive"] = legend_contrastive_loss(
                    torch.stack(leg_list),
                    torch.stack(qry_list),
                    temperature=self.legend_contrastive_tau,
                )
            else:
                losses["legend_contrastive"] = outputs["pred_logits"].sum() * 0.0
        else:
            losses["legend_contrastive"] = outputs["pred_logits"].sum() * 0.0

        manual_weights = {
            "cls": self.weights.cls,
            "mask": self.weights.mask,
            "dice": self.weights.dice,
            "quality": self.weights.quality,
            "aux": self.weights.aux,
            "dn_mask": self.weights.dn_mask,
            "otm": self.weights.otm,
            "centerline": self.weights.centerline,
            "crossing": self.weights.crossing,
            "boundary": self.weights.boundary,
            "direction": self.weights.direction,
            "grid": self.weights.grid,
            "cape": self.weights.cape,
            "pcc": self.weights.pcc,
            "snake_offset": self.weights.snake_offset,
            "legend_contrastive": self.weights.legend_contrastive,
            "topograph": self.weights.topograph,
            "efd": self.weights.efd,
        }

        total = losses["cls"].new_tensor(0.0)
        for key in self._loss_keys:
            w = float(manual_weights[key])
            if w == 0.0:
                continue
            if self.use_uncertainty_weighting:
                log_var = self.log_vars[key]
                precision = torch.exp(-log_var)
                total = total + w * (precision * losses[key] + 0.5 * log_var.squeeze())
            else:
                total = total + w * losses[key]
        losses["total"] = total
        return losses

    def _compute_efd_loss(
        self,
        pred_efd: Tensor,
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Elliptical Fourier Descriptor loss (EFDTR, ICML 2025).

        Computes GT Fourier coefficients from GT mask contours and
        supervises the predicted coefficients via smooth-L1 loss.
        """
        total = pred_efd.sum() * 0.0
        count = 0
        n_harmonics = pred_efd.shape[-1] // 4

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            gt_masks = targets[b]["masks"][tgt_idx].float().to(pred_efd.device)
            pred_coeffs = pred_efd[b, src_idx]  # (M, 4*K)

            for m_idx in range(gt_masks.shape[0]):
                gt_coeffs = _mask_to_efd(gt_masks[m_idx], n_harmonics)
                if gt_coeffs is None:
                    continue
                gt_coeffs = gt_coeffs.to(pred_efd.device)
                total = total + F.smooth_l1_loss(pred_coeffs[m_idx], gt_coeffs)
                count += 1

        return total / max(count, 1)

    def _compute_dn_loss(
        self,
        dn_outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        dn_meta: dict,
    ) -> Tensor:
        """Direct focal+dice supervision for denoising queries (no Hungarian matching)."""
        pred_masks = dn_outputs["pred_masks"]   # (B, groups*max_n, H, W)
        n_gts: List[int] = dn_meta["n_gts"]
        max_n: int = dn_meta["max_n"]
        groups: int = dn_meta["groups"]

        total = pred_masks.sum() * 0.0
        count = 0
        for b_idx in range(pred_masks.shape[0]):
            n = n_gts[b_idx]
            if n == 0:
                continue
            gt = targets[b_idx]["masks"].float().to(pred_masks.device)   # (N, H, W)
            for g in range(groups):
                pred_g = pred_masks[b_idx, g * max_n: g * max_n + n]    # (N, H, W)
                gt_r = gt
                if pred_g.shape[-2:] != gt.shape[-2:]:
                    gt_r = F.interpolate(gt.unsqueeze(1), size=pred_g.shape[-2:],
                                         mode="nearest").squeeze(1)
                total = total + sigmoid_focal_loss(
                    pred_g.unsqueeze(1), gt_r.unsqueeze(1)
                ) + dice_loss_from_logits(pred_g.unsqueeze(1), gt_r.unsqueeze(1))
                count += 1
        return total / max(count, 1)


@dataclass
class InferenceConfig:
    score_thresh: float = 0.35
    mask_thresh: float = 0.5
    top_k: int = 120
    nms_iou: float = 0.7
    min_pixels: int = 24
    quality_power: float = 1.0
    crossing_iou_override: float = 0.92  # allow overlap up to this IoU when crossing detected
    crossing_conf_thresh: float = 0.4
    crossing_min_overlap: int = 5


def _binary_iou(a: Tensor, b: Tensor, eps: float = 1e-6) -> float:
    inter = (a & b).float().sum()
    union = (a | b).float().sum()
    return float((inter + eps) / (union + eps))


def _should_suppress(
    new_mask: Tensor,
    kept_masks: List[Tensor],
    crossing_map: Optional[Tensor],    # (1, H, W) sigmoid crossing prob
    nms_iou: float,
    crossing_iou_override: float,
    crossing_conf_thresh: float,
    crossing_min_overlap: int,
) -> bool:
    """
    Returns True when new_mask should be suppressed.

    For high-IoU pairs, checks whether the overlap region has strong crossing
    signal. If so, the IoU threshold is raised to crossing_iou_override,
    allowing two crossing curves to coexist even when their masks partially
    overlap.
    """
    for km in kept_masks:
        iou = _binary_iou(new_mask, km)
        if iou <= nms_iou:
            continue
        # Above base threshold: check for crossing evidence in overlap
        if crossing_map is not None:
            overlap = (new_mask & km).float()
            if int(overlap.sum()) >= int(crossing_min_overlap):
                cross_conf = float(
                    (crossing_map.squeeze(0) * overlap).sum() / overlap.sum()
                )
                # Strong crossing → raise threshold, keep both masks
                if cross_conf > crossing_conf_thresh and iou < crossing_iou_override:
                    continue
        return True  # suppress
    return False  # keep


@torch.no_grad()
def postprocess_curve_instances(
    outputs: Dict[str, Tensor], cfg: InferenceConfig = InferenceConfig()
) -> List[Dict[str, Tensor]]:
    cls_scores = outputs["pred_logits"].softmax(-1)[..., 1]
    if "pred_quality" in outputs:
        cls_scores = cls_scores * torch.sigmoid(outputs["pred_quality"]).pow(cfg.quality_power)
    masks = torch.sigmoid(outputs["pred_masks"])
    style_ids = outputs.get("pred_style_logits")
    if style_ids is not None:
        style_ids = style_ids.argmax(-1)
    results: List[Dict[str, Tensor]] = []
    for b in range(cls_scores.shape[0]):
        # Per-image crossing map for ordering-aware NMS
        crossing_map: Optional[Tensor] = None
        if "crossing_logits" in outputs:
            crossing_map = torch.sigmoid(outputs["crossing_logits"][b])   # (1, H, W)

        order = torch.argsort(cls_scores[b], descending=True)[: cfg.top_k]
        kept_masks, kept_scores, kept_styles = [], [], []
        for q in order:
            s = cls_scores[b, q]
            if float(s) < cfg.score_thresh:
                continue
            mb = masks[b, q] > cfg.mask_thresh
            if int(mb.sum()) < cfg.min_pixels:
                continue
            if _should_suppress(mb, kept_masks, crossing_map,
                                 cfg.nms_iou, cfg.crossing_iou_override,
                                 cfg.crossing_conf_thresh, cfg.crossing_min_overlap):
                continue
            kept_masks.append(mb)
            kept_scores.append(s)
            if style_ids is not None:
                kept_styles.append(style_ids[b, q])
        if kept_masks:
            out_masks = torch.stack(kept_masks, 0)
            out_scores = torch.stack(kept_scores, 0)
            out_styles = (torch.stack(kept_styles, 0) if kept_styles
                          else torch.zeros(out_masks.shape[0], dtype=torch.long, device=out_masks.device))
        else:
            h, w = masks.shape[-2:]
            out_masks = torch.zeros(0, h, w, dtype=torch.bool, device=masks.device)
            out_scores = torch.zeros(0, device=masks.device)
            out_styles = torch.zeros(0, dtype=torch.long, device=masks.device)
        results.append({"masks": out_masks, "scores": out_scores, "styles": out_styles})
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = CurveSOTAConfig(
        backbone=CurveSegConfig(
            encoder_dims=(32, 64, 128, 256),
            blocks_per_stage=(1, 1, 1, 1),
            decoder_dim=64,
            embed_dim=16,
            mamba_d_state=16,
            mamba_headdim=16,
            mamba_chunk_size=16,
        ),
        num_queries=16,
        query_layers=2,
        query_heads=4,
        num_styles=3,
        align_topk=24,
    )
    model = CurveSOTAQueryNet(cfg)
    model.eval()
    images = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        outputs = model(images)
    print("pred_logits:", tuple(outputs["pred_logits"].shape))
    print("pred_masks:", tuple(outputs["pred_masks"].shape))
    print("pred_quality:", tuple(outputs["pred_quality"].shape))
    if "snake_offsets" in outputs:
        print(f"snake_offsets: {len(outputs['snake_offsets'])} entries")

    # Print progressive branch info
    for i, stage in enumerate(model.encoder.stages):
        n_branches = stage[0].active_branches
        print(f"Stage {i}: {len(n_branches)} branches = {n_branches}")

    # Print ablation flags
    print(f"use_bato={cfg.use_bato}, use_query_align={cfg.use_query_align}, "
          f"use_position_relation={cfg.use_position_relation}")

    targets = {
        "instance_ids": torch.randint(0, 4, (1, 64, 64)),
        "centerline_mask": (torch.rand(1, 64, 64) > 0.85).float(),
        "crossing_mask": (torch.rand(1, 64, 64) > 0.95).float(),
        "direction_vectors": torch.cat([
            F.normalize(torch.randn(1, 2, 64, 64), dim=1),
            F.normalize(torch.randn(1, 2, 64, 64), dim=1),
        ], dim=1),
        "layering_target": torch.randint(-1, 2, (1, 64, 64)).float(),
        "grid_mask": (torch.rand(1, 64, 64) > 0.7).float(),
    }

    # Run with grad for loss
    model.train()
    outputs = model(images)
    criterion = CurveSOTACriterion()
    losses = criterion(outputs, targets)
    print("Loss terms:", sorted(losses.keys()))
    print("total loss:", float(losses["total"]))
