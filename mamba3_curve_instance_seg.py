"""
Mamba-3 Spatial Encoder for chart curve instance segmentation (Sec 3.1).

Core contribution: Progressive multi-branch 2D spatial fusion via Mamba-3 SSM,
designed for thin curvilinear structures in chemical chart images.

Progressive scan branches (per stage):
  Stage 0 (H/4):  3 branches — local + row + col
  Stage 1 (H/8):  5 branches — + snake_h + snake_v
  Stage 2 (H/16): 9 branches — + row_rev + col_rev + snake_d45 + snake_d135
  Stage 3 (H/32): 9 branches — same as Stage 2

Branch types:
  - Bidirectional row/column Mamba scans (4 directions)
  - Structure-aware snake scans with learned deformable offsets (4 directions)
  - Local depthwise convolution for fine detail

Also provides:
  - Multi-scale FPN decoder with H/2 stem skip connection
  - SE-attention weighted branch fusion with DropPath & LayerScale
  - Multi-task heads with task-specific depthwise refinement
  - Grid suppression via learnable additive bias
  - Topology-preserving loss functions (CAPE, cbDice, topograph)
  - Uncertainty-weighted multi-loss balancing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from mamba3 import Mamba3, Mamba3Config


def _group_count(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return max(groups, 1)


def _resolve_headdim(d_model: int, preferred: int) -> int:
    d_inner = 2 * d_model
    if d_inner % preferred == 0:
        return preferred
    for candidate in (64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1):
        if d_inner % candidate == 0:
            return candidate
    return 1


def _ensure_channel_first_mask(mask: Tensor) -> Tensor:
    if mask.dim() == 3:
        return mask.unsqueeze(1).float()
    if mask.dim() == 4:
        return mask.float()
    raise ValueError(f"Expected mask with 3D/4D shape, got {tuple(mask.shape)}")


def _pad_sequence_to_chunk(x: Tensor, chunk_size: int) -> Tuple[Tensor, int]:
    """Pad sequence length (dim=1) to a multiple of chunk_size."""
    seq_len = x.shape[1]
    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len == 0:
        return x, 0
    pad = torch.zeros(
        x.shape[0],
        pad_len,
        x.shape[2],
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat([x, pad], dim=1), pad_len


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


CHART_CURVE_BRANCHES: Tuple[Tuple[str, ...], ...] = (
    ("local", "row", "col"),
    ("local", "row", "col", "snake_h", "snake_v"),
    ("local", "row", "col", "row_rev", "col_rev", "snake_h", "snake_v"),
    ("local", "row", "col", "row_rev", "col_rev", "snake_h", "snake_v"),
)

LEGACY_BRANCHES: Tuple[Tuple[str, ...], ...] = (
    ("local", "row", "col"),
    ("local", "row", "col", "snake_h", "snake_v"),
    ("local", "row", "col", "row_rev", "col_rev", "snake_h", "snake_v", "snake_d45", "snake_d135"),
    ("local", "row", "col", "row_rev", "col_rev", "snake_h", "snake_v", "snake_d45", "snake_d135"),
)


@dataclass
class CurveSegConfig:
    in_channels: int = 3
    encoder_dims: Tuple[int, int, int, int] = (64, 128, 224, 384)
    blocks_per_stage: Tuple[int, int, int, int] = (2, 2, 3, 2)
    mamba_d_state: int = 64
    mamba_headdim: int = 32
    mamba_chunk_size: int = 64
    decoder_dim: int = 128
    embed_dim: int = 16
    # Progressive branch design: controls which branches are active per stage
    branches_per_stage: Tuple[Tuple[str, ...], ...] = CHART_CURVE_BRANCHES

    # Architectural switches used by both base and SOTA variants
    use_stem_skip: bool = True           # FPN H/2 stem skip connection
    use_grid_suppression: bool = True    # additive grid bias branch
    use_grad_checkpoint: bool = False    # gradient checkpointing for memory savings

    def __post_init__(self):
        if len(self.encoder_dims) != len(self.blocks_per_stage):
            raise ValueError("encoder_dims and blocks_per_stage must have the same length")
        if len(self.branches_per_stage) != len(self.blocks_per_stage):
            raise ValueError("branches_per_stage and blocks_per_stage must have the same length")

    @classmethod
    def chart_preset(cls, **kwargs) -> "CurveSegConfig":
        return cls(**kwargs)

    @classmethod
    def legacy_preset(cls, **kwargs) -> "CurveSegConfig":
        defaults = dict(
            encoder_dims=(64, 128, 256, 512),
            blocks_per_stage=(2, 2, 4, 2),
            branches_per_stage=LEGACY_BRANCHES,
        )
        defaults.update(kwargs)
        return cls(**defaults)


def _build_mamba_args(d_model: int, cfg: CurveSegConfig) -> Mamba3Config:
    return Mamba3Config(
        d_model=d_model,
        n_layer=1,
        d_state=cfg.mamba_d_state,
        expand=2,
        headdim=_resolve_headdim(d_model, cfg.mamba_headdim),
        chunk_size=cfg.mamba_chunk_size,
        vocab_size=32,
        use_mimo=False,
    )


class SinusoidalPosEmb2D(nn.Module):
    """2D sinusoidal positional embedding, added to features before Mamba scans.

    Generates fixed sin/cos embeddings for (H, W) spatial positions (half channels
    encode row position, half encode column position), then projects to the target
    channel dimension via a learned linear layer.  The embedding is cached and
    recomputed only when the spatial size changes.
    """

    def __init__(self, channels: int, temperature: float = 10000.0):
        super().__init__()
        self.channels = channels
        self.temperature = temperature
        # project concatenated (row_emb ‖ col_emb) → channels
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self._cache_hw: Tuple[int, int] = (0, 0)
        self._cache_emb: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) → returns (B, C, H, W) with positional encoding added."""
        _, _, H, W = x.shape
        if self._cache_hw != (H, W) or self._cache_emb is None or self._cache_emb.device != x.device or self._cache_emb.dtype != x.dtype:
            self._cache_emb = self._build(H, W, x.device, x.dtype)
            self._cache_hw = (H, W)
        return x + self.proj(self._cache_emb)

    def _build(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        half = self.channels // 2
        # frequency bands
        dim_t = torch.arange(0, half, 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (dim_t / max(half, 1))
        # row positions
        pos_h = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)  # (H,1)
        # col positions
        pos_w = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)  # (W,1)

        sin_h = torch.sin(pos_h / dim_t)  # (H, half//2)
        cos_h = torch.cos(pos_h / dim_t)  # (H, half//2)
        emb_h = torch.stack([sin_h, cos_h], dim=2).reshape(H, -1)  # (H, half)
        emb_h = emb_h[:, :half]  # truncate if odd

        sin_w = torch.sin(pos_w / dim_t)
        cos_w = torch.cos(pos_w / dim_t)
        emb_w = torch.stack([sin_w, cos_w], dim=2).reshape(W, -1)
        emb_w = emb_w[:, :self.channels - half]

        # broadcast to (C, H, W)
        emb = torch.cat([
            emb_h.unsqueeze(1).expand(-1, W, -1),   # (H, W, half)
            emb_w.unsqueeze(0).expand(H, -1, -1),   # (H, W, C-half)
        ], dim=-1).permute(2, 0, 1)  # (C, H, W)
        return emb.unsqueeze(0).to(dtype)  # (1, C, H, W)


class SnakeScanBranch(nn.Module):
    """Structure-aware snake scanning inspired by SCSegamba (CVPR 2025).

    Instead of fixed scanning patterns (row/column/diagonal), predicts
    per-pixel offsets perpendicular to the scan direction so the scan
    path adaptively follows curvilinear structures.

    Supports four directions:
      - horizontal: scans left-to-right with vertical offsets
      - vertical: scans top-to-bottom with horizontal offsets
      - diag45: scans along 45° anti-diagonals with perpendicular offsets
      - diag135: scans along 135° diagonals with perpendicular offsets
    """

    def __init__(self, channels: int, cfg: CurveSegConfig, direction: str = "horizontal"):
        super().__init__()
        assert direction in ("horizontal", "vertical", "diag45", "diag135")
        self.direction = direction
        self.channels = channels
        self.chunk_size = cfg.mamba_chunk_size

        # Lightweight offset predictor (depthwise 5×5 → pointwise 1×1)
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )

        mamba_args = _build_mamba_args(channels, cfg)
        self.mamba = Mamba3(mamba_args)

        # Zero-init offset_net last layer for near-zero offsets at init (M3)
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # Cache for diagonal scan indices (avoid Python loop every forward)
        self._diag_cache_hw: Tuple[int, int] = (0, 0)
        self._diag_cache_idx: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (output_2d, offset) where offset is (B, 1, H, W)."""
        b, c, h, w = x.shape

        # Predict perpendicular offsets bounded by tanh
        offset = torch.tanh(self.offset_net(x)) * min(8.0, max(h, w) * 0.1)

        # Build normalized sampling grid
        gy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        gx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        base_y, base_x = torch.meshgrid(gy, gx, indexing="ij")
        base_y = base_y.unsqueeze(0).expand(b, -1, -1)
        base_x = base_x.unsqueeze(0).expand(b, -1, -1)

        if self.direction == "horizontal":
            offset_norm = offset.squeeze(1) * (2.0 / max(h - 1, 1))
            grid = torch.stack([base_x, (base_y + offset_norm).clamp(-1, 1)], dim=-1)
        elif self.direction == "vertical":
            offset_norm = offset.squeeze(1) * (2.0 / max(w - 1, 1))
            grid = torch.stack([(base_x + offset_norm).clamp(-1, 1), base_y], dim=-1)
        elif self.direction == "diag45":
            # 45° scan: offset perpendicular to anti-diagonal direction
            scale = 0.7071  # 1/sqrt(2)
            offset_norm_x = offset.squeeze(1) * (2.0 / max(w - 1, 1)) * scale
            offset_norm_y = offset.squeeze(1) * (2.0 / max(h - 1, 1)) * scale
            grid = torch.stack([
                (base_x + offset_norm_x).clamp(-1, 1),
                (base_y + offset_norm_y).clamp(-1, 1),
            ], dim=-1)
        else:  # diag135
            # 135° scan: offset perpendicular to main diagonal direction
            scale = 0.7071
            offset_norm_x = offset.squeeze(1) * (2.0 / max(w - 1, 1)) * scale
            offset_norm_y = offset.squeeze(1) * (2.0 / max(h - 1, 1)) * (-scale)
            grid = torch.stack([
                (base_x + offset_norm_x).clamp(-1, 1),
                (base_y + offset_norm_y).clamp(-1, 1),
            ], dim=-1)

        # Sample features along deformed grid
        deformed = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        # Scan along primary direction with Mamba
        if self.direction == "horizontal":
            seq = rearrange(deformed, "b c h w -> (b h) w c")
            seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
            out, _ = self.mamba(seq, None)
            if pad_len > 0:
                out = out[:, :-pad_len]
            out_2d = rearrange(out, "(b h) w c -> b c h w", b=b, h=h)
        elif self.direction == "vertical":
            seq = rearrange(deformed, "b c h w -> (b w) h c")
            seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
            out, _ = self.mamba(seq, None)
            if pad_len > 0:
                out = out[:, :-pad_len]
            out_2d = rearrange(out, "(b w) h c -> b c h w", b=b, w=w)
        elif self.direction == "diag45":
            # Scan along anti-diagonals: flatten by anti-diagonal index
            out_2d = self._diag_scan(deformed, b, h, w, anti=True)
        else:  # diag135
            # Scan along main diagonals
            out_2d = self._diag_scan(deformed, b, h, w, anti=False)

        return out_2d, offset

    @staticmethod
    def _build_diag_indices(h: int, w: int, anti: bool) -> Tensor:
        """Pre-compute diagonal scan indices (pure integer arithmetic, no grad)."""
        indices = []
        if anti:
            for s in range(h + w - 1):
                for i in range(max(0, s - w + 1), min(h, s + 1)):
                    indices.append(i * w + (s - i))
        else:
            for d in range(-(w - 1), h):
                for i in range(max(0, d), min(h, d + w)):
                    indices.append(i * w + (i - d))
        return torch.tensor(indices, dtype=torch.long)

    def _get_diag_indices(self, h: int, w: int, anti: bool, device: torch.device) -> Tensor:
        """Return cached diagonal indices, rebuilding only when (H, W) changes."""
        key = (h, w)
        if self._diag_cache_hw != key or self._diag_cache_idx is None:
            self._diag_cache_idx = self._build_diag_indices(h, w, anti)
            self._diag_cache_hw = key
        return self._diag_cache_idx.to(device)

    def _diag_scan(self, deformed: Tensor, b: int, h: int, w: int, anti: bool) -> Tensor:
        """Scan along diagonal lines and reconstruct 2D output."""
        idx = self._get_diag_indices(h, w, anti, deformed.device)
        flat = deformed.flatten(2)  # (B, C, H*W)
        seq = flat[:, :, idx].permute(0, 2, 1)  # (B, H*W, C)
        seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
        out, _ = self.mamba(seq, None)
        if pad_len > 0:
            out = out[:, :-pad_len]
        out_flat = torch.zeros_like(flat)  # (B, C, H*W)
        out_flat[:, :, idx] = out.permute(0, 2, 1)
        return out_flat.view(b, -1, h, w)


class SEBranchFusion(nn.Module):
    """SE-attention weighted branch fusion.

    Each branch output (B, C, H, W) is stacked → (B, N_branch, C, H, W).
    Per-branch global avg pool → (B, N) → small MLP(N, max(N//r,2), N) → sigmoid
    produces lightweight per-branch importance weights.
    Weighted sum + concat projection residual → (B, C, H, W).
    """

    def __init__(self, channels: int, n_branches: int, reduction: int = 4):
        super().__init__()
        self.n_branches = n_branches
        # Lightweight SE: operates on N-dim branch descriptor (not N*C)
        hidden = max(n_branches // reduction, 2)
        self.se = nn.Sequential(
            nn.Linear(n_branches, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_branches),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * n_branches, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        # Zero-init proj last conv for stable residual at init (M3)
        nn.init.zeros_(self.proj[0].weight)

    def forward(self, branches: list) -> Tensor:
        """branches: list of (B, C, H, W) tensors."""
        if len(branches) == 1:
            return branches[0]
        stacked = torch.stack(branches, dim=1)  # (B, N, C, H, W)
        b, n, c, h, w = stacked.shape
        # Per-branch channel mean → global spatial avg → (B, N)
        branch_desc = stacked.mean(dim=2).mean(dim=(2, 3))  # (B, N)
        weights = self.se(branch_desc).view(b, n, 1, 1, 1)  # (B, N, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, C, H, W)
        concat_feat = stacked.reshape(b, n * c, h, w)  # (B, N*C, H, W)
        return fused + self.proj(concat_feat)


class DropPath(nn.Module):
    """Stochastic Depth (drop path) regularization (Huang et al., 2016).

    During training, randomly drops the entire residual branch with probability
    `drop_prob`, forcing the network to be robust to missing sub-networks.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # Proper Bernoulli stochastic depth: add keep_prob BEFORE floor so that
        # P(floor(rand + keep_prob) = 1) = keep_prob (timm / torchvision convention)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep_prob).floor_()
        return x / keep_prob * mask


class SpatialMambaFusionBlock(nn.Module):
    """
    2D block with progressive branch design:
    - 2D positional encoding (sinusoidal + learned projection)
    - Subset of branches determined by active_branches parameter
    - SE-attention weighted fusion + channel MLP
    - DropPath (stochastic depth) on both residual paths
    - LayerScale (per-channel learnable scaling) on both residual paths
    """

    def __init__(self, channels: int, cfg: CurveSegConfig,
                 active_branches: Tuple[str, ...] = (
                     "local", "row", "col", "row_rev", "col_rev", "snake_h", "snake_v",
                 ),
                 drop_path: float = 0.0,
                 layer_scale_init: float = 1e-4):
        super().__init__()
        self.channels = channels
        self.chunk_size = cfg.mamba_chunk_size
        self.active_branches = active_branches

        # 2D positional encoding: injected before Mamba scans
        self.pos_emb = SinusoidalPosEmb2D(channels)

        self.norm1 = nn.GroupNorm(_group_count(channels), channels)

        # Conditionally create branches based on active_branches
        if "local" in active_branches:
            self.local_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
            self.local_pw = nn.Conv2d(channels, channels, 1, bias=False)

        mamba_args = _build_mamba_args(channels, cfg)
        if "row" in active_branches:
            self.row_mamba = Mamba3(mamba_args)
        if "row_rev" in active_branches:
            self.row_mamba_rev = Mamba3(mamba_args)
        if "col" in active_branches:
            self.col_mamba = Mamba3(mamba_args)
        if "col_rev" in active_branches:
            self.col_mamba_rev = Mamba3(mamba_args)
        if "snake_h" in active_branches:
            self.snake_h = SnakeScanBranch(channels, cfg, direction="horizontal")
        if "snake_v" in active_branches:
            self.snake_v = SnakeScanBranch(channels, cfg, direction="vertical")
        if "snake_d45" in active_branches:
            self.snake_d45 = SnakeScanBranch(channels, cfg, direction="diag45")
        if "snake_d135" in active_branches:
            self.snake_d135 = SnakeScanBranch(channels, cfg, direction="diag135")

        self.fuse = SEBranchFusion(channels, n_branches=len(active_branches))

        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
        )

        # LayerScale: per-channel learnable scaling (CaiT / ConvNeXt V2)
        self.gamma1 = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))
        self.gamma2 = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))

        # DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Zero-init MLP last layer for stable residual (M3)
        nn.init.zeros_(self.mlp[-1].weight)

    def _scan_rows(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        seq = rearrange(x, "b c h w -> (b h) w c")
        seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
        out, _ = self.row_mamba(seq, None)
        if pad_len > 0:
            out = out[:, :-pad_len]
        return rearrange(out, "(b h) w c -> b c h w", b=b, h=h)

    def _scan_cols(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        seq = rearrange(x, "b c h w -> (b w) h c")
        seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
        out, _ = self.col_mamba(seq, None)
        if pad_len > 0:
            out = out[:, :-pad_len]
        return rearrange(out, "(b w) h c -> b c h w", b=b, w=w)

    def _scan_rows_rev(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        seq = rearrange(x, "b c h w -> (b h) w c").flip(1)
        seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
        out, _ = self.row_mamba_rev(seq, None)
        if pad_len > 0:
            out = out[:, :-pad_len]
        return rearrange(out.flip(1), "(b h) w c -> b c h w", b=b, h=h)

    def _scan_cols_rev(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        seq = rearrange(x, "b c h w -> (b w) h c").flip(1)
        seq, pad_len = _pad_sequence_to_chunk(seq, self.chunk_size)
        out, _ = self.col_mamba_rev(seq, None)
        if pad_len > 0:
            out = out[:, :-pad_len]
        return rearrange(out.flip(1), "(b w) h c -> b c h w", b=b, w=w)

    def forward(self, x: Tensor) -> Tuple[Tensor, list]:
        """Returns (output, snake_offsets) where snake_offsets is a list of (name, offset) pairs."""
        res = x
        x_n = self.norm1(x)
        x_n = self.pos_emb(x_n)

        parts = []
        snake_offsets = []

        if hasattr(self, "local_dw"):
            parts.append(self.local_pw(self.local_dw(x_n)))
        if hasattr(self, "row_mamba"):
            parts.append(self._scan_rows(x_n))
        if hasattr(self, "col_mamba"):
            parts.append(self._scan_cols(x_n))
        if hasattr(self, "row_mamba_rev"):
            parts.append(self._scan_rows_rev(x_n))
        if hasattr(self, "col_mamba_rev"):
            parts.append(self._scan_cols_rev(x_n))
        if hasattr(self, "snake_h"):
            sh_out, sh_offset = self.snake_h(x_n)
            parts.append(sh_out)
            snake_offsets.append(("snake_h", sh_offset))
        if hasattr(self, "snake_v"):
            sv_out, sv_offset = self.snake_v(x_n)
            parts.append(sv_out)
            snake_offsets.append(("snake_v", sv_offset))
        if hasattr(self, "snake_d45"):
            sd45_out, sd45_offset = self.snake_d45(x_n)
            parts.append(sd45_out)
            snake_offsets.append(("snake_d45", sd45_offset))
        if hasattr(self, "snake_d135"):
            sd135_out, sd135_offset = self.snake_d135(x_n)
            parts.append(sd135_out)
            snake_offsets.append(("snake_d135", sd135_offset))

        mixed = self.fuse(parts)
        x = res + self.drop_path(self.gamma1 * mixed)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x, snake_offsets


class EncoderOutput(NamedTuple):
    """Named return type for CurveMambaEncoder for clarity and structured access."""
    f1: Tensor       # Stage 0 features (H/4)
    f2: Tensor       # Stage 1 features (H/8)
    f3: Tensor       # Stage 2 features (H/16)
    f4: Tensor       # Stage 3 features (H/32)
    stem_half: Tensor  # Stem features (H/2) for high-res skip
    snake_offsets: list  # List of (name, offset_tensor) pairs


class CurveMambaEncoder(nn.Module):
    def __init__(self, cfg: CurveSegConfig):
        super().__init__()
        dims = cfg.encoder_dims
        blocks = cfg.blocks_per_stage

        # Stem split into two layers for high-res skip (Improvement 10)
        self.stem_layer1 = ConvGNAct(cfg.in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1)  # → H/2
        self.stem_layer2 = ConvGNAct(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1)           # → H/4

        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    ConvGNAct(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                )
                for i in range(len(dims) - 1)
            ]
        )

        # Progressive branch design: each stage gets its own branch subset
        # DropPath rates increase linearly from 0 to max_drop_path across all blocks
        branches = cfg.branches_per_stage
        total_blocks = sum(blocks)
        max_drop_path = 0.2
        dp_rates = [x.item() for x in torch.linspace(0, max_drop_path, total_blocks)]
        block_idx = 0
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage_blocks = nn.ModuleList()
            for _ in range(blocks[i]):
                stage_blocks.append(
                    SpatialMambaFusionBlock(
                        dims[i], cfg,
                        active_branches=branches[i],
                        drop_path=dp_rates[block_idx],
                    )
                )
                block_idx += 1
            self.stages.append(stage_blocks)

        self.use_grad_checkpoint = cfg.use_grad_checkpoint

    def forward(self, x: Tensor) -> EncoderOutput:
        """Returns EncoderOutput(f1, f2, f3, f4, stem_half, snake_offsets)."""
        stem_half = self.stem_layer1(x)   # (B, dims[0]//2, H/2, W/2)
        x = self.stem_layer2(stem_half)    # (B, dims[0], H/4, W/4)

        feats = []
        all_offsets = []
        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                x = self.downsamples[stage_idx - 1](x)
            for block in stage:
                if self.use_grad_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint
                    x, block_offsets = checkpoint(block, x, use_reentrant=False)
                else:
                    x, block_offsets = block(x)
                all_offsets.extend(block_offsets)
            feats.append(x)

        return EncoderOutput(feats[0], feats[1], feats[2], feats[3], stem_half, all_offsets)


class FPNDecoder(nn.Module):
    def __init__(self, in_dims: Tuple[int, int, int, int], out_dim: int,
                 stem_dim: Optional[int] = None):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_dim, 1, bias=False) for c in in_dims])
        self.smooth = nn.ModuleList([ConvGNAct(out_dim, out_dim, 3, 1, 1) for _ in in_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * len(in_dims), out_dim, 1, bias=False),
            nn.GroupNorm(_group_count(out_dim), out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_dim), out_dim),
            nn.GELU(),
        )
        # High-res stem skip: projects H/2 stem features → out_dim,
        # then FPN output is upsampled to H/2 resolution to receive fine details.
        if stem_dim is not None:
            self.stem_skip = nn.Sequential(
                nn.Conv2d(stem_dim, out_dim, 1, bias=False),
                nn.GroupNorm(_group_count(out_dim), out_dim),
            )

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor],
                stem_feat: Optional[Tensor] = None) -> Tensor:
        p = [lat(f) for lat, f in zip(self.laterals, features)]

        for i in range(len(p) - 2, -1, -1):
            p[i] = p[i] + F.interpolate(
                p[i + 1], size=p[i].shape[-2:], mode="bilinear", align_corners=False
            )

        p = [smooth(x) for smooth, x in zip(self.smooth, p)]
        target_hw = p[0].shape[-2:]
        all_scales = [p[0]] + [
            F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False) for x in p[1:]
        ]
        out = self.fuse(torch.cat(all_scales, dim=1))

        # High-res stem skip: upsample FPN output to H/2 (stem resolution)
        # so that fine-grained edge information from the stem is preserved.
        if hasattr(self, "stem_skip") and stem_feat is not None:
            stem_proj = self.stem_skip(stem_feat)  # (B, out_dim, H/2, W/2)
            out = F.interpolate(
                out, size=stem_proj.shape[-2:], mode="bilinear", align_corners=False
            )
            out = out + stem_proj

        return out


class PredictionHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_refine: bool = True):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),  # depthwise
            nn.Conv2d(in_ch, in_ch, 1, bias=False),  # pointwise
            nn.GroupNorm(_group_count(in_ch), in_ch),
            nn.GELU(),
        ) if use_refine else nn.Identity()
        self.head = nn.Sequential(
            ConvGNAct(in_ch, in_ch, 3, 1, 1),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def forward(self, x: Tensor, out_hw: Tuple[int, int]) -> Tensor:
        x = self.refine(x)
        y = self.head(x)
        if y.shape[-2:] != out_hw:
            y = F.interpolate(y, size=out_hw, mode="bilinear", align_corners=False)
        return y


class GridSuppressionBranch(nn.Module):
    """
    Estimates grid/background bias using multi-scale features (f1, f2, f3).
    Coarser features are projected and upsampled to f1 resolution before fusion.
    Outputs decoder_dim-channel additive bias (Improvement 9: additive instead of multiplicative).
    Also provides 1-channel grid logits for visualization/loss via grid_logits_conv.
    """

    def __init__(self, in_chs: Tuple[int, int, int], decoder_dim: int):
        super().__init__()
        ch1, ch2, ch3 = in_chs
        self.proj2 = nn.Conv2d(ch2, ch1, 1, bias=False)
        self.proj3 = nn.Conv2d(ch3, ch1, 1, bias=False)
        self.branch = nn.Sequential(
            nn.Conv2d(ch1, ch1, 5, padding=2, groups=ch1, bias=False),
            nn.Conv2d(ch1, ch1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch1, decoder_dim, 1),
        )
        # 1-channel logits for visualization / auxiliary loss
        self.grid_logits_conv = nn.Conv2d(decoder_dim, 1, 1)
        # Zero-init last conv for stable additive bias at init (M3)
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, feats: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Returns (grid_bias, grid_logits) where grid_bias is (B, decoder_dim, H, W)."""
        f1, f2, f3 = feats
        f2_up = F.interpolate(self.proj2(f2), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f3_up = F.interpolate(self.proj3(f3), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        bias = self.branch(f1 + f2_up + f3_up)
        logits = self.grid_logits_conv(bias)
        return bias, logits


class CurveInstanceMamba3Net(nn.Module):
    """
    Output dictionary keys:
      - composed_mask:       (B, 1, H, W)  — skeleton + width composed mask [0,1]
      - instance_embeddings: (B, E, H, W)
      - centerline_logits:   (B, 1, H, W)
      - width_logits:        (B, 1, H, W)  — predicted half-width field
      - direction_vectors:   (B, 4, H, W)  — primary + secondary direction
      - crossing_logits:     (B, 1, H, W)
      - grid_logits:         (B, 1, H, W)
      - snake_offsets:       list of (name, offset) pairs for offset loss
    """

    def __init__(self, cfg: CurveSegConfig = CurveSegConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = CurveMambaEncoder(cfg)
        self.decoder = FPNDecoder(
            cfg.encoder_dims, cfg.decoder_dim,
            stem_dim=(cfg.encoder_dims[0] // 2) if cfg.use_stem_skip else None,
        )
        self.grid_branch = GridSuppressionBranch(
            (cfg.encoder_dims[0], cfg.encoder_dims[1], cfg.encoder_dims[2]),
            decoder_dim=cfg.decoder_dim,
        )

        # Learnable scale for grid additive bias (starts at 0 for stable init, M5)
        self.grid_scale = nn.Parameter(torch.zeros(1)) if cfg.use_grid_suppression else None

        self.embedding_head = PredictionHead(cfg.decoder_dim, cfg.embed_dim)
        self.centerline_head = PredictionHead(cfg.decoder_dim, 1)
        self.width_head = PredictionHead(cfg.decoder_dim, 1)
        self.direction_head = PredictionHead(cfg.decoder_dim, 4)
        self.crossing_head = PredictionHead(cfg.decoder_dim, 1)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        out_hw = images.shape[-2:]

        f1, f2, f3, f4, stem_half, snake_offsets = self.encoder(images)
        fused = self.decoder(
            (f1, f2, f3, f4),
            stem_feat=stem_half if self.cfg.use_stem_skip else None,
        )

        grid_bias, grid_logits_low = self.grid_branch((f1, f2, f3))
        if self.grid_scale is not None:
            if grid_bias.shape[-2:] != fused.shape[-2:]:
                grid_bias = F.interpolate(
                    grid_bias, size=fused.shape[-2:], mode="bilinear", align_corners=False
                )
                assert grid_bias.shape == fused.shape, (
                    f"grid_bias shape {grid_bias.shape} != fused shape {fused.shape} "
                    "after interpolation; check FPN decoder output resolution"
                )
            fused = fused + self.grid_scale * grid_bias

        centerline_logits = self.centerline_head(fused, out_hw)
        width_logits = self.width_head(fused, out_hw)

        return {
            "composed_mask": skeleton_to_mask(centerline_logits, width_logits),
            "instance_embeddings": self.embedding_head(fused, out_hw),
            "centerline_logits": centerline_logits,
            "width_logits": width_logits,
            "direction_vectors": self.direction_head(fused, out_hw),
            "crossing_logits": self.crossing_head(fused, out_hw),
            "grid_logits": F.interpolate(
                grid_logits_low, size=out_hw, mode="bilinear", align_corners=False
            ),
            "snake_offsets": snake_offsets,
        }


def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    targets = targets.float()
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def dice_loss_from_logits(logits: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    targets = targets.float()
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = 2.0 * (probs * targets).sum(dims)
    den = probs.sum(dims) + targets.sum(dims)
    return (1.0 - ((inter + eps) / (den + eps))).mean()


def dice_loss_from_probs(probs: Tensor, targets: Tensor, eps: float = 1e-6) -> Tensor:
    """Dice loss directly from probability values in [0, 1] (no sigmoid applied)."""
    targets = targets.float()
    dims = (1, 2, 3)
    inter = 2.0 * (probs * targets).sum(dims)
    den = probs.sum(dims) + targets.sum(dims)
    return (1.0 - ((inter + eps) / (den + eps))).mean()


def direction_cosine_loss(pred_vec: Tensor, gt_vec: Tensor, valid_mask: Tensor) -> Tensor:
    pred_unit = F.normalize(pred_vec, dim=1, eps=1e-6)
    gt_unit = F.normalize(gt_vec, dim=1, eps=1e-6)
    cos_sim = (pred_unit * gt_unit).sum(dim=1, keepdim=True)
    loss_map = (1.0 - cos_sim.abs()) * valid_mask.float()
    denom = valid_mask.float().sum().clamp_min(1.0)
    return loss_map.sum() / denom


def snake_offset_alignment_loss(
    snake_offsets: list,
    direction_gt: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Align snake scan offsets with GT curve tangent direction (Improvement 2).

    For horizontal snake: offset is vertical → should correlate with |dy|
    For vertical snake: offset is horizontal → should correlate with |dx|
    For diagonal snakes: similar alignment along perpendicular direction.

    snake_offsets: list of (name, offset) where offset is (B, 1, H, W)
    direction_gt: (B, 2, H, W) GT tangent direction (dx, dy)
    valid_mask: (B, 1, H, W) where GT direction is valid
    """
    if not snake_offsets:
        return direction_gt.sum() * 0.0

    dx_gt = direction_gt[:, 0:1]  # (B, 1, H, W) — signed
    dy_gt = direction_gt[:, 1:2]  # (B, 1, H, W) — signed
    mag_gt = (dx_gt.pow(2) + dy_gt.pow(2)).sqrt().clamp_min(1e-6)  # (B, 1, H, W)
    total = direction_gt.new_tensor(0.0)
    count = 0

    for name, offset in snake_offsets:
        # Resize offset to match GT if needed
        off = offset  # (B, 1, H, W) — keep sign
        if off.shape[-2:] != valid_mask.shape[-2:]:
            off = F.interpolate(off, size=valid_mask.shape[-2:], mode="bilinear", align_corners=False)

        # Normalize offset to [-1, 1] range by max magnitude.
        # Gate on significance to avoid 1e-6 denominator amplifying gradients
        # when predictions are near zero early in training.
        off_max = off.abs().amax(dim=(2, 3), keepdim=True)
        valid_off = (off_max > 1e-4).float()
        off_norm = torch.where(
            off_max > 1e-4,
            off / off_max.clamp_min(1e-6),
            torch.zeros_like(off),
        )

        if name == "snake_h":
            # Horizontal scan offsets vertically → align signed offset with dy/|d|
            target = dy_gt / mag_gt
        elif name == "snake_v":
            # Vertical scan offsets horizontally → align signed offset with dx/|d|
            target = dx_gt / mag_gt
        elif name == "snake_d45":
            # 45° anti-diagonal: perpendicular is (1,1)/√2 direction
            # Offset along perp → project tangent onto (1,1)/√2
            target = (dy_gt - dx_gt) / (mag_gt * 1.4142)
        elif name == "snake_d135":
            # 135° main diagonal: perpendicular is (1,-1)/√2 direction
            target = (dy_gt + dx_gt) / (mag_gt * 1.4142)
        else:
            continue

        # Smooth-L1 alignment loss weighted by valid mask and significant-offset gate
        diff = F.smooth_l1_loss(off_norm, target, reduction="none") * valid_mask.float() * valid_off
        denom = (valid_mask.float() * valid_off).sum().clamp_min(1.0)
        total = total + diff.sum() / denom
        count += 1

    if count == 0:
        return direction_gt.sum() * 0.0
    return total / count


def _soft_erode(x: Tensor) -> Tensor:
    """Differentiable soft morphological erosion via negated max-pooling."""
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: Tensor) -> Tensor:
    """Soft morphological opening = dilate(erode(x))."""
    return F.max_pool2d(_soft_erode(x), kernel_size=3, stride=1, padding=1)


def _soft_skel(x: Tensor, iters: int = 5) -> Tensor:
    """Differentiable soft skeletonization via iterative erosion-opening residuals."""
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters - 1):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred_probs: Tensor, gt: Tensor, iters: int = 5, eps: float = 1e-6) -> Tensor:
    """
    Soft clDice loss (CVPR 2021) for topology-preserving curve segmentation.

    Computes a Dice F1 score on soft skeletons of both prediction and GT,
    directly penalizing connectivity breaks that pixel-level Dice ignores.

    pred_probs: (B, 1, H, W) in [0, 1]
    gt:         (B, 1, H, W) binary
    """
    gt = gt.float()
    skel_pred = _soft_skel(pred_probs, iters)
    skel_gt = _soft_skel(gt, iters)
    # Precision: skeleton of prediction covered by GT mask
    tprec = ((skel_pred * gt).sum() + eps) / (skel_pred.sum() + eps)
    # Recall: skeleton of GT covered by predicted mask
    tsens = ((skel_gt * pred_probs).sum() + eps) / (skel_gt.sum() + eps)
    return 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + eps)


def width_smoothness_loss(width_logits: Tensor, centerline_probs: Tensor) -> Tensor:
    """
    Total-variation smoothness regularization on the predicted width field,
    weighted by the predicted centerline probability.

    Real chart curves have near-constant stroke width, so the width field
    should vary slowly along the curve. This provides direct supervision for
    the width head even when explicit GT width annotations are unavailable.

    width_logits:      (B, 1, H, W)
    centerline_probs:  (B, 1, H, W) in [0, 1], used as spatial weight
    """
    width = F.softplus(width_logits)  # positive half-width
    dy = (width[:, :, 1:, :] - width[:, :, :-1, :]).abs()
    dx = (width[:, :, :, 1:] - width[:, :, :, :-1]).abs()
    # Average centerline confidence of neighbouring pixel pairs as spatial weight
    w_y = (centerline_probs[:, :, 1:, :] + centerline_probs[:, :, :-1, :]) * 0.5
    w_x = (centerline_probs[:, :, :, 1:] + centerline_probs[:, :, :, :-1]) * 0.5
    tv = (dy * w_y).sum() + (dx * w_x).sum()
    return tv / (w_y.sum() + w_x.sum()).clamp_min(1.0)


def skeleton_recall_loss(pred_probs: Tensor, skel_gt: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Skeleton Recall Loss (ECCV 2024, DKFZ).

    Measures how well the predicted mask covers the GT centerline/skeleton.
    Directly penalizes gaps and breaks in predicted curves without requiring
    expensive persistent homology. 90% faster than PH-based methods.

    pred_probs: (B, 1, H, W) in [0, 1]
    skel_gt:    (B, 1, H, W) GT skeleton / centerline mask (binary)
    """
    skel = skel_gt.float()
    recall = ((pred_probs * skel).sum() + eps) / (skel.sum() + eps)
    return 1.0 - recall


def cbdice_loss(
    pred_probs: Tensor,
    gt: Tensor,
    width_logits: Tensor,
    iters: int = 5,
    eps: float = 1e-6,
) -> Tensor:
    """
    Centerline Boundary Dice Loss (MICCAI 2024).

    Extends clDice by weighting skeleton-overlap by the predicted half-width
    (radius), so thin and thick curve segments contribute equally.
    Without this, clDice under-penalises errors on thin curves because their
    skeleton has fewer pixels.

    pred_probs:   (B, 1, H, W) in [0, 1]
    gt:           (B, 1, H, W) binary GT mask
    width_logits: (B, 1, H, W) predicted half-width logits
    """
    gt = gt.float()
    # Skip loss for empty GT (no curves in image) — avoids tsens=1 with zero gt skeleton
    if gt.sum() < 1:
        return pred_probs.sum() * 0.0
    # Detach radius so width gradient comes only through width_smoothness_loss
    radius = F.softplus(width_logits).detach()  # (B, 1, H, W) positive half-width

    skel_pred = _soft_skel(pred_probs, iters)
    skel_gt = _soft_skel(gt, iters)

    # Per-pixel radius at GT skeleton; normalize to a probability distribution
    r_gt = radius * skel_gt                                               # (B, 1, H, W)
    r_norm = r_gt / (r_gt.sum(dim=(1, 2, 3), keepdim=True) + 1e-4)       # 1e-4: thin-line weights > eps

    # Radius-weighted precision: skeleton of pred covered by GT, weighted by GT radius
    tprec = ((skel_pred * gt * r_norm).sum() + eps) / (skel_pred.sum() + eps)
    # Radius-weighted recall: skeleton of GT covered by pred, weighted by GT radius
    tsens = ((skel_gt * pred_probs * r_norm).sum() + eps) / (skel_gt.sum() + eps)
    return 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + eps)


def cape_connectivity_loss(
    pred_probs: Tensor,
    centerline_mask: Tensor,
    instance_ids: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Simplified CAPE Connectivity Loss (MICCAI 2025).

    Enforces two complementary constraints simultaneously:

    1. GAP penalty (skeleton recall): the predicted mask must cover every
       pixel of the GT centerline — direct supervision against broken curves.
    2. BRIDGE penalty: the predicted mask must be suppressed at boundaries
       between two different curve instances, preventing false connections
       (pixel bridges) that would merge distinct curve series.

    pred_probs:      (B, 1, H, W) in [0, 1]
    centerline_mask: (B, 1, H, W) GT skeleton / centerline (binary)
    instance_ids:    (B, H, W) integer instance labels, 0 = background
    """
    # 1. Gap penalty: skeleton recall
    gap_loss = skeleton_recall_loss(pred_probs, centerline_mask, eps)

    # 2. Bridge penalty: find pixel pairs where adjacent pixels belong to
    #    different (both non-background) instances → suppress prediction there
    ids = instance_ids.long()  # (B, H, W)
    h_diff = (ids[:, :, 1:] != ids[:, :, :-1]) & (ids[:, :, 1:] > 0) & (ids[:, :, :-1] > 0)
    v_diff = (ids[:, 1:, :] != ids[:, :-1, :]) & (ids[:, 1:, :] > 0) & (ids[:, :-1, :] > 0)

    # Grow the boundary by one pixel on each side so nearby pixels are also penalised
    bridge = torch.zeros_like(ids, dtype=torch.float)
    bridge[:, :, 1:]  += h_diff.float()
    bridge[:, :, :-1] += h_diff.float()
    bridge[:, 1:, :]  += v_diff.float()
    bridge[:, :-1, :] += v_diff.float()
    bridge = (bridge > 0).float().unsqueeze(1)  # (B, 1, H, W)

    n_bridge = bridge.sum().clamp_min(1.0)
    bridge_loss = (pred_probs * bridge).sum() / n_bridge

    return gap_loss + bridge_loss


def topograph_loss(
    pred_probs: Tensor,
    gt: Tensor,
    n_thresholds: int = 5,
    iters: int = 3,
    eps: float = 1e-6,
) -> Tensor:
    """
    Multi-confidence skeleton recall loss.

    Evaluates skeleton recall at multiple confidence thresholds (0.3 → 0.7)
    applied to the **prediction**. This forces the model to maintain correct
    connectivity across its full confidence range — preventing "thin bridges"
    that only appear at high confidence and break at lower thresholds.

    At each threshold t, the prediction is soft-binarized via a steep sigmoid
    gate centered at t, then skeleton recall is measured against the GT
    skeleton. Since GT is binary, the same GT skeleton is used at all levels;
    the multi-threshold mechanism regularizes the prediction's confidence
    calibration, not the GT topology.

    Inspired by Topograph (ICLR 2025) but simplified: avoids persistent
    homology and operates purely through differentiable soft-skeletonization.

    pred_probs:   (B, 1, H, W) in [0, 1]
    gt:           (B, 1, H, W) binary
    n_thresholds: number of confidence levels to evaluate
    iters:        soft-skeletonisation iterations per level
    """
    gt = gt.float()
    if gt.sum() < 1:
        return pred_probs.sum() * 0.0

    # GT skeleton is computed once (binary GT is threshold-invariant)
    skel_gt = _soft_skel(gt, iters)
    total = pred_probs.new_tensor(0.0)
    thresholds = torch.linspace(0.3, 0.7, n_thresholds, device=pred_probs.device)

    for t in thresholds:
        # Soft thresholding of the prediction at confidence level t
        pred_t = torch.sigmoid((pred_probs - t) * 30.0)
        recall_t = ((pred_t * skel_gt).sum() + eps) / (skel_gt.sum() + eps)
        total = total + (1.0 - recall_t)

    return total / n_thresholds


def supcon_instance_loss(
    embeddings: Tensor,
    instance_ids: Tensor,
    temperature: float = 0.07,
    max_pixels: int = 512,
) -> Tensor:
    """
    Supervised Contrastive Loss for instance pixel embeddings.

    Replaces the fixed-margin discriminative push-pull loss with an
    InfoNCE-style SupCon loss that adapts to any number of instances
    without requiring explicit margin tuning.

    instance_ids: (B, H, W), 0 = background, positive integers = instance IDs
    """
    b, e, h, w = embeddings.shape
    emb_flat = rearrange(embeddings, "b e h w -> b (h w) e")
    ids_flat = rearrange(instance_ids, "b h w -> b (h w)")

    total_loss = embeddings.new_tensor(0.0)
    valid_batches = 0

    for i in range(b):
        ids = ids_flat[i]
        # L2-normalize embeddings onto unit sphere
        emb = F.normalize(emb_flat[i].float(), dim=-1)   # (HW, E) — float32 for AMP safety

        fg = ids > 0
        if fg.sum() < 2:
            continue

        fg_ids = ids[fg]
        fg_emb = emb[fg]   # (N_fg, E)

        # Random subsample to keep memory bounded
        if fg_emb.shape[0] > max_pixels:
            idx = torch.randperm(fg_emb.shape[0], device=emb.device)[:max_pixels]
            fg_emb = fg_emb[idx]
            fg_ids = fg_ids[idx]

        n = fg_emb.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=emb.device)

        # Cosine-similarity matrix scaled by temperature
        sim = torch.matmul(fg_emb, fg_emb.t()) / temperature     # (N, N)
        # Subtract row-max for numerical stability (no gradient through max)
        sim = sim - sim.detach().max(dim=1, keepdim=True).values

        exp_sim = torch.exp(sim)
        # Denominator: all pairs except self
        denom = exp_sim.masked_fill(eye, 0.0).sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_prob = sim - torch.log(denom)          # (N, N)

        # Positive pairs: same instance, excluding self
        pos_mask = (fg_ids.unsqueeze(0) == fg_ids.unsqueeze(1)) & ~eye   # (N, N)
        n_pos = pos_mask.float().sum(dim=1)        # (N,)
        valid = n_pos > 0
        if not valid.any():
            continue

        loss_i = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos.clamp_min(1.0)
        total_loss = total_loss + loss_i[valid].mean()
        valid_batches += 1

    if valid_batches == 0:
        return embeddings.sum() * 0.0
    return total_loss / valid_batches


def skeleton_to_mask(
    centerline_logits: Tensor,
    width_logits: Tensor,
    dilation_steps: int = 6,
) -> Tensor:
    """
    Compose centerline probability + predicted width into a soft binary mask
    via differentiable iterative dilation.

    At each step s, pixels within distance s of the skeleton are included
    if the predicted half-width >= s (gated by a soft sigmoid step).

    centerline_logits: (B, 1, H, W)
    width_logits:      (B, 1, H, W)  →  softplus → positive half-width in pixels
    Returns:           (B, 1, H, W) in [0, 1]
    """
    skel = torch.sigmoid(centerline_logits)
    width = F.softplus(width_logits) + 0.5       # minimum half-width 0.5 px
    mask = skel
    for step in range(1, dilation_steps + 1):
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        # Soft gate: probability that predicted width covers this step
        gate = torch.sigmoid(5.0 * (width - step))
        mask = mask + (dilated - mask) * gate
    return mask.clamp(0.0, 1.0)


@dataclass
class CurveLossWeights:
    """Loss weights for base model, organized to parallel SOTA model groups.

    Group A — Mask & Instance (3): mask, embedding, width
    Group B — Pixel Topology (4): centerline, crossing, direction, grid
    Group C — Connectivity (2): topology (cbDice), cape
    Group D — Snake Offset (1): snake_offset alignment
    """
    mask: float = 1.0          # composed mask (skeleton→width) supervision
    embedding: float = 0.8     # SupCon instance embedding
    width: float = 0.2         # width-field TV smoothness regularization
    centerline: float = 0.7
    crossing: float = 0.4
    direction: float = 0.3
    grid: float = 0.3          # grid/background auxiliary loss (supervises grid_logits_conv)
    topology: float = 0.3      # cbDice: radius-weighted skeleton connectivity
    cape: float = 0.25         # CAPE: gap + inter-instance bridge penalty
    snake_offset: float = 0.1  # Snake offset alignment with GT tangent direction


class CurveInstanceLoss(nn.Module):
    """10-term loss for base model, organized into 4 groups.

    Group A — Mask & Instance (3): mask, embedding, width
    Group B — Pixel Topology (4): centerline, crossing, direction, grid
    Group C — Connectivity (2): topology (cbDice), cape
    Group D — Snake Offset (1): snake_offset alignment

    Expected targets:
      - curve_mask:       (B, H, W) or (B, 1, H, W)
      - centerline_mask:  (B, H, W) or (B, 1, H, W)
      - instance_ids:     (B, H, W), 0 = background
      - direction_vectors:(B, 4, H, W) primary + secondary direction
      - crossing_mask:    (B, H, W) or (B, 1, H, W)
    """

    # All loss term names for uncertainty weighting
    _LOSS_KEYS = (
        "mask", "embedding", "width", "centerline", "crossing",
        "direction", "grid", "topology", "cape", "snake_offset",
    )

    def __init__(
        self,
        weights: CurveLossWeights = CurveLossWeights(),
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        embed_dim: int = 16,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.weights = weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_uncertainty_weighting = use_uncertainty_weighting
        # 2-layer 1×1 conv projection head for SupCon (improves representation learning)
        self.supcon_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
        )
        # Adaptive uncertainty weighting (Kendall et al., 2018):
        # learned log(σ²) per loss term; effective weight = w_manual / (2·σ²) + log(σ)
        if use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                k: nn.Parameter(torch.zeros(1)) for k in self._LOSS_KEYS
            })

    def forward(self, outputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        device = outputs["composed_mask"].device

        curve_target = _ensure_channel_first_mask(targets["curve_mask"]).to(device)
        center_target = _ensure_channel_first_mask(targets["centerline_mask"]).to(device)
        crossing_target = _ensure_channel_first_mask(targets["crossing_mask"]).to(device)

        center_logits = outputs["centerline_logits"]
        crossing_logits = outputs["crossing_logits"]

        # Primary mask supervision via skeleton→width composed mask (probs, not logits)
        composed = outputs["composed_mask"]
        losses["mask"] = dice_loss_from_probs(composed, curve_target) + F.binary_cross_entropy(
            composed.clamp(1e-6, 1 - 1e-6), curve_target
        )

        losses["centerline"] = dice_loss_from_logits(
            center_logits, center_target
        ) + sigmoid_focal_loss(
            center_logits, center_target, alpha=self.focal_alpha, gamma=self.focal_gamma
        )

        direction_target = targets["direction_vectors"].to(device).float()
        valid_dir = (center_target > 0.5).float()
        # Primary direction: all centerline pixels
        losses["direction"] = direction_cosine_loss(
            outputs["direction_vectors"][:, :2], direction_target[:, :2], valid_dir
        )
        # Secondary direction: crossing pixels only (when 4-channel target provided)
        if direction_target.shape[1] >= 4:
            losses["direction"] = losses["direction"] + 0.5 * direction_cosine_loss(
                outputs["direction_vectors"][:, 2:], direction_target[:, 2:],
                (crossing_target > 0.5).float(),
            )

        # Crossing pixels are sparse positives → alpha=0.75 to up-weight them
        losses["crossing"] = dice_loss_from_logits(
            crossing_logits, crossing_target
        ) + sigmoid_focal_loss(
            crossing_logits, crossing_target, alpha=0.75, gamma=self.focal_gamma
        )

        instance_ids = targets["instance_ids"].to(device).long()
        # Apply projection head before SupCon for better representation learning
        proj_emb = self.supcon_proj(outputs["instance_embeddings"])
        losses["embedding"] = supcon_instance_loss(proj_emb, instance_ids)

        # cbDice: radius-weighted skeleton topology (replaces soft-clDice)
        losses["topology"] = cbdice_loss(
            outputs["composed_mask"], curve_target, outputs["width_logits"]
        )

        # CAPE: gap penalty (covers GT skeleton) + bridge penalty (no pred between instances)
        losses["cape"] = cape_connectivity_loss(
            outputs["composed_mask"], center_target, instance_ids
        )

        # Width TV: encourage locally smooth width field along predicted curves
        skel_probs = torch.sigmoid(center_logits).detach()
        losses["width"] = width_smoothness_loss(outputs["width_logits"], skel_probs)

        # Grid auxiliary loss: use real grid_mask annotation if available (non-zero),
        # otherwise fall back to inverted curve_mask as proxy (grid = 1 - curve).
        if "grid_mask" in targets and _ensure_channel_first_mask(targets["grid_mask"]).sum() > 0:
            grid_target = _ensure_channel_first_mask(targets["grid_mask"]).to(device)
        else:
            grid_target = 1.0 - curve_target  # proxy: background is non-curve
        losses["grid"] = sigmoid_focal_loss(
            outputs["grid_logits"], grid_target, alpha=0.25, gamma=self.focal_gamma
        )

        # Snake offset alignment loss (Improvement 2)
        if "snake_offsets" in outputs and self.weights.snake_offset > 0:
            direction_target = targets["direction_vectors"].to(device).float()
            valid_dir = (center_target > 0.5).float()
            losses["snake_offset"] = snake_offset_alignment_loss(
                outputs["snake_offsets"], direction_target[:, :2], valid_dir
            )
        else:
            losses["snake_offset"] = outputs["composed_mask"].sum() * 0.0

        # Compute total with optional uncertainty weighting
        manual_weights = {
            "mask": self.weights.mask, "embedding": self.weights.embedding,
            "width": self.weights.width, "centerline": self.weights.centerline,
            "crossing": self.weights.crossing, "direction": self.weights.direction,
            "grid": self.weights.grid, "topology": self.weights.topology,
            "cape": self.weights.cape, "snake_offset": self.weights.snake_offset,
        }
        total = losses["mask"].new_tensor(0.0)
        for key in self._LOSS_KEYS:
            w = manual_weights[key]
            if w == 0.0:
                continue
            if self.use_uncertainty_weighting:
                # Adaptive: w / (2·exp(log_var)) · loss + 0.5·log_var
                log_var = self.log_vars[key]
                precision = torch.exp(-log_var)
                total = total + w * (precision * losses[key] + 0.5 * log_var.squeeze())
            else:
                total = total + w * losses[key]
        losses["total"] = total
        return losses


if __name__ == "__main__":
    torch.manual_seed(0)

    model = CurveInstanceMamba3Net(
        CurveSegConfig(
            encoder_dims=(64, 128, 256, 512),
            blocks_per_stage=(1, 1, 2, 1),
            decoder_dim=128,
            embed_dim=16,
            mamba_chunk_size=32,
        )
    )
    model.eval()

    images = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        outputs = model(images)

    print("Output keys:", sorted(k for k in outputs.keys() if k != "snake_offsets"))
    for k, v in outputs.items():
        if k == "snake_offsets":
            print(f"snake_offsets: {len(v)} entries")
            continue
        print(k, tuple(v.shape))

    # Print progressive branch info
    for i, stage in enumerate(model.encoder.stages):
        n_branches = stage[0].active_branches
        print(f"Stage {i}: {len(n_branches)} branches = {n_branches}")

    criterion = CurveInstanceLoss()
    targets = {
        "curve_mask": (torch.rand(2, 256, 256) > 0.7).float(),
        "centerline_mask": (torch.rand(2, 256, 256) > 0.85).float(),
        "instance_ids": torch.randint(0, 6, (2, 256, 256)),
        "direction_vectors": torch.cat([
            F.normalize(torch.randn(2, 2, 256, 256), dim=1),
            F.normalize(torch.randn(2, 2, 256, 256), dim=1),
        ], dim=1),
        "crossing_mask": (torch.rand(2, 256, 256) > 0.95).float(),
        "layering_target": torch.randint(-1, 2, (2, 256, 256)).float(),
    }

    # Need to re-run with grad for loss computation
    model.train()
    outputs = model(images)
    losses = criterion(outputs, targets)
    print("Loss terms:", sorted(losses.keys()))
    print("Total loss:", float(losses["total"]))
