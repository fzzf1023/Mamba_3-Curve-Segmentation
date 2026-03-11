"""
mamba3-minimal
==============

A minimal, single-file implementation of the Mamba-3 model in PyTorch.

> **MAMBA-3: IMPROVED SEQUENCE MODELING USING STATE SPACE PRINCIPLES**
> Paper: Under review at ICLR 2026

Key innovations over Mamba-2:
  1) Trapezoidal Discretization (Eq. 4) — second-order accurate state update
  2) Complex-valued SSM / Data-Dependent RoPE (Eq. 9) — enables state-tracking
  3) MIMO formulation (Appendix D) — improves hardware utilization during decode
  4) QK-Normalization on B, C — replaces post-SSD norm
  5) Learnable BC Bias — head-specific, channel-wise, initialized to ones
  6) No short convolution — trapezoidal + bias makes conv1d unnecessary

Architecture follows Llama design: alternating Mamba-3 SSM blocks and SwiGLU MLP
blocks with pre-normalization (RMSNorm).
"""

import json
import math
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = str | torch.device | None

# ──────────────────────────────────────────────────────────────────────────────
# Hardware-agnostic device selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Detect the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Mamba3Config:
    d_model: int          # Model dimension (D)
    n_layer: int = 24     # Number of Mamba-3 layers (each = SSM mixer + SwiGLU MLP)
    d_state: int = 128    # SSM state dimension (N). Must be even for RoPE pairing.
    expand: int = 2       # Expansion factor (E). d_inner = expand * d_model
    headdim: int = 64     # Head dimension (P)
    chunk_size: int = 64  # Matrix partition size for chunked SSD (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16
    use_mimo: bool = False  # Toggle MIMO variant
    mimo_rank: int = 4      # MIMO rank (r) when use_mimo=True

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        assert self.d_state % 2 == 0, "d_state must be even for complex SSM / RoPE pairing"

        # SwiGLU inner dimension (Llama convention: ≈ 8/3 * d_model, rounded to 256)
        self.d_mlp_inner = 256 * ((int(2 * (4 * self.d_model) / 3) + 255) // 256)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


# ──────────────────────────────────────────────────────────────────────────────
# Inference Cache
# ──────────────────────────────────────────────────────────────────────────────

class InferenceCache(NamedTuple):
    ssm_state: Tensor   # (batch, nheads, headdim, d_state) — SSM hidden state h_t
    prev_Bx: Tensor     # (batch, nheads, headdim, d_state) — B̄_{t-1} ⊗ x_{t-1} for β term
    cum_angle: Tensor   # (batch, nheads, d_state // 2) — cumulative RoPE angle Σ Δ_i * θ_i
                        # Initialized as (batch, 1, d_state // 2) for broadcasting; becomes
                        # (batch, nheads, d_state // 2) after the first forward/step call.

    @staticmethod
    def alloc(batch_size: int, args: Mamba3Config, device: Device = None):
        return InferenceCache(
            ssm_state=torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
            prev_Bx=torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
            cum_angle=torch.zeros(
                batch_size, 1, args.d_state // 2, device=device
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Language Model
# ──────────────────────────────────────────────────────────────────────────────

class Mamba3LMHeadModel(nn.Module):
    """Full language model: Embedding → N × [RMSNorm → Mamba3 → RMSNorm → SwiGLU] → RMSNorm → LM Head."""

    def __init__(self, args: Mamba3Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        self.backbone = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                # Llama-style: pre-norm → mixer → residual, pre-norm → MLP → residual
                                mixer_norm=RMSNorm(args.d_model, device=device),
                                mixer=Mamba3(args, device=device),
                                mlp_norm=RMSNorm(args.d_model, device=device),
                                mlp=SwiGLU(args.d_model, args.d_mlp_inner, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )
        # Weight tying (standard practice)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) token IDs
            h: hidden states for inference step. If present, the constant-time
               (wrt sequence length) inference path is taken; input_ids should
               have shape (batch, 1).

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing input_ids
        """
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        # Shape: (batch, seqlen, d_model)
        x = self.backbone.embedding(input_ids)

        for i, layer in enumerate(self.backbone.layers):
            # --- Mamba-3 SSM mixer with pre-normalization ---
            y, h[i] = layer.mixer(layer.mixer_norm(x), h[i])
            x = y + x  # Residual connection

            # --- SwiGLU MLP with pre-normalization ---
            x = x + layer.mlp(layer.mlp_norm(x))

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        """Auto-regressive generation, yielding one token at a time."""
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # ── Process prompt ──
        # The input to the forward (non-inference) path must have length that is
        # a multiple of chunk_size. We process the bulk via forward and the tail
        # token-by-token via the inference path.
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # ── Generate tokens ──
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if next_token.item() == eos_token_id:
                return
            tokens = next_token.unsqueeze(0)
            yield cast(int, next_token.item()), h


# ──────────────────────────────────────────────────────────────────────────────
# Mamba-3 SSM Block (the core mixer)
# ──────────────────────────────────────────────────────────────────────────────

class Mamba3(nn.Module):
    """
    Mamba-3 SSM block implementing:
      • Trapezoidal discretization (Proposition 1, Eq. 4)
      • Complex-valued SSM via data-dependent RoPE (Proposition 4, Eq. 9)
      • QK-normalization on B, C (Section 3.4)
      • Learnable BC bias (Section 3.4, Appendix G)
      • Optional MIMO (Section 3.3, Appendix D)
    """

    def __init__(self, args: Mamba3Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # ── Projection dimensions depend on SISO vs MIMO ──
        # MIMO (Appendix D): B, C become rank-R matrices → d_state * mimo_rank
        self.bc_dim = args.d_state * args.mimo_rank if args.use_mimo else args.d_state

        # ── Input projection ──
        # Projects d_model → z + x + B + C + dt + λ + θ
        #   z:     d_inner           — gate for output (SiLU activation)
        #   x:     d_inner           — SSM input value
        #   B:     bc_dim            — SSM input projection (d_state for SISO, d_state*R for MIMO)
        #   C:     bc_dim            — SSM output projection
        #   dt:    nheads            — step size Δ (one per head)
        #   λ:     nheads            — trapezoidal interpolation parameter (one per head)
        #   θ:     d_state // 2      — rotation angles for data-dependent RoPE
        d_in_proj = (
            2 * args.d_inner     # z + x
            + 2 * self.bc_dim    # B + C (expanded by mimo_rank when MIMO)
            + 2 * args.nheads    # dt + λ
            + args.d_state // 2  # θ
        )
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        # ── SSM parameters ──
        # A_log: log of negative diagonal (one scalar per head, always negative after exp)
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        # D: skip connection coefficient (one per head)
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        # dt_bias: bias added to dt before softplus
        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))

        # ── Stable initialization (prevents NaN from exp on garbage memory) ──
        # A_log: follow Mamba-2 convention — log(1..nheads) so A ≈ −[1..nheads] initially
        # D: skip connection starts at 1.0
        # dt_bias: inv-softplus of uniform dt in [0.001, 0.1]
        with torch.no_grad():
            self.A_log.copy_(
                torch.log(torch.arange(1, args.nheads + 1, dtype=torch.float32))
            )
            self.D.fill_(1.0)
            dt_init = torch.exp(
                torch.rand(args.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp_min(1e-4)
            # inv_softplus(y) = log(exp(y) − 1) = y + log(1 − exp(−y))
            self.dt_bias.copy_(dt_init + torch.log(-torch.expm1(-dt_init)))

        # ── QK-Normalization on B, C (Section 3.4) ──
        # Applied after projection, before bias addition — mirrors QK-Norm in Transformers
        self.B_norm = RMSNorm(self.bc_dim, device=device)
        self.C_norm = RMSNorm(self.bc_dim, device=device)

        # ── BC Bias (Section 3.4, Appendix G) ──
        # Head-specific, channel-wise, initialized to ones, data-independent, trainable
        if args.use_mimo:
            R = args.mimo_rank
            # MIMO: bias is (nheads, d_state, mimo_rank) — per-head, per-channel, per-rank
            self.B_bias = nn.Parameter(torch.ones(args.nheads, args.d_state, R, device=device))
            self.C_bias = nn.Parameter(torch.ones(args.nheads, args.d_state, R, device=device))
            # MIMO rank expansion (Appendix D): X_t = X'_t ⊙ W_X (element-wise broadcast)
            self.mimo_x_proj = nn.Parameter(torch.ones(args.nheads, args.headdim, R, device=device))
            self.mimo_z_proj = nn.Parameter(torch.ones(args.nheads, args.headdim, R, device=device))
            # MIMO rank down-projection: (P, R) → (P) via learned weighted sum
            self.mimo_down = nn.Parameter(torch.ones(args.nheads, args.headdim, R, device=device) / R)
        else:
            # SISO: bias is (nheads, d_state)
            self.B_bias = nn.Parameter(torch.ones(args.nheads, args.d_state, device=device))
            self.C_bias = nn.Parameter(torch.ones(args.nheads, args.d_state, device=device))

        # ── Output projection ──
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: InferenceCache | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) — input (already pre-normed by layer norm)
            h: hidden state for inference. If present, uses constant-time step.

        Return (y, h)
            y: (batch, seqlen, d_model) — output
            h: updated inference cache
        """
        if h is not None:
            return self.step(u, h)

        batch, seqlen, _ = u.shape
        args = self.args

        # ── Negative diagonal of A (always < 0) ──
        A = -torch.exp(self.A_log)  # Shape: (nheads,)

        # ── Project input to all SSM components ──
        proj = self.in_proj(u)  # Shape: (batch, seqlen, d_in_proj)
        z, x, B, C, dt, lam, theta = torch.split(
            proj,
            [
                args.d_inner,        # z: gate
                args.d_inner,        # x: SSM input
                self.bc_dim,         # B: input projection (d_state for SISO, d_state*R for MIMO)
                self.bc_dim,         # C: output projection
                args.nheads,         # dt: step size per head
                args.nheads,         # λ: trapezoidal parameter per head
                args.d_state // 2,   # θ: rotation angles
            ],
            dim=-1,
        )

        # ── Discretization parameters ──
        # Eq. 4: dt controls both forget-gate (α) and input-gate (γ)
        dt = F.softplus(dt + self.dt_bias)  # Shape: (batch, seqlen, nheads)
        # λ_t = σ(u_t): data-dependent trapezoidal interpolation parameter (Appendix B.3)
        lam = torch.sigmoid(lam)  # Shape: (batch, seqlen, nheads)

        # ── QK-Normalization on B, C (no gating, just RMSNorm) ──
        B = self.B_norm(B)  # Shape: (batch, seqlen, bc_dim)
        C = self.C_norm(C)  # Shape: (batch, seqlen, bc_dim)

        # ── Data-Dependent RoPE angles (shared between SISO and MIMO) ──
        # Compute rotation angles per (head, state-pair):
        #   raw_angle[t, h, j] = Δ_t[h] * θ_t[j]
        #   cum_angle[t, h, j] = −Σ_{i=0}^{t} raw_angle[i, h, j]
        # θ is shared across heads; dt is per-head → angles are per-head
        raw_angles = (
            dt.unsqueeze(-1)                                  # (b, l, nheads, 1)
            * rearrange(theta, "b l n -> b l 1 n")           # (b, l, 1, d_state//2)
        )  # Shape: (batch, seqlen, nheads, d_state // 2)
        cum_angles = -torch.cumsum(raw_angles, dim=1)  # Shape: (batch, seqlen, nheads, d_state // 2)

        # ── Trapezoidal coefficients (Proposition 1, Eq. 4) ──
        #   α_t = exp(Δ_t * A_t)                    — decay
        #   β_t = (1 − λ_t) * Δ_t * exp(Δ_t * A_t)  — left-endpoint (previous input)
        #   γ_t = λ_t * Δ_t                          — right-endpoint (current input)
        dA = dt * rearrange(A, "h -> 1 1 h")  # (batch, seqlen, nheads)
        alpha = torch.exp(dA)
        beta = (1 - lam) * dt * alpha          # (batch, seqlen, nheads)
        gamma = lam * dt                        # (batch, seqlen, nheads)

        # ── Reshape x for SSD: (batch, seqlen, nheads, headdim) ──
        x = rearrange(x, "b l (h p) -> b l h p", p=args.headdim)

        if args.use_mimo:
            R = args.mimo_rank

            # ── MIMO: Reshape B, C from (b, l, d_state*R) → (b, l, d_state, R) ──
            B = rearrange(B, "b l (n r) -> b l n r", r=R)
            C = rearrange(C, "b l (n r) -> b l n r", r=R)

            # ── Broadcast to heads + bias: (b, l, 1, n, r) + (h, n, r) ──
            B = rearrange(B, "b l n r -> b l 1 n r") + self.B_bias
            C = rearrange(C, "b l n r -> b l 1 n r") + self.C_bias

            # ── Apply RoPE on d_state dim for each rank independently ──
            # Move d_state to last position, apply, move back
            B = rearrange(B, "b l h n r -> b l h r n")
            C = rearrange(C, "b l h n r -> b l h r n")
            B = apply_rope(B, cum_angles.unsqueeze(3))  # broadcast over rank dim
            C = apply_rope(C, cum_angles.unsqueeze(3))
            B = rearrange(B, "b l h r n -> b l h n r")  # (b, l, h, d_state, R)
            C = rearrange(C, "b l h r n -> b l h n r")

            # ── Expand x to rank R (Appendix D): X_t = X'_t ⊙ W_X ──
            x_mimo = x.unsqueeze(-1) * self.mimo_x_proj  # (b, l, h, p, R)

            # ── Trapezoidal SSD (two-SSD decomposition, MIMO variant) ──
            y_gamma, state_gamma = ssd_mimo(
                x_mimo * rearrange(gamma, "b l h -> b l h 1 1"),  # scaled by γ
                dA, B, C, args.chunk_size, device=self.device,
            )

            # Pre-shift B and x at the FULL SEQUENCE level for β term
            B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))       # (b, l, h, n, R)
            x_mimo_prev = F.pad(x_mimo[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))  # (b, l, h, p, R)

            y_beta, state_beta = ssd_mimo(
                x_mimo_prev * rearrange(beta, "b l h -> b l h 1 1"),
                dA, B_prev, C, args.chunk_size, device=self.device,
            )

            # Sum the two contributions
            y = y_gamma + y_beta            # (batch, seqlen, nheads, headdim, R)
            ssm_state = state_gamma + state_beta  # (batch, nheads, headdim, d_state)

            # ── Skip connection in rank-R space ──
            y = y + (x * self.D.unsqueeze(-1)).unsqueeze(-1)  # broadcast over R

            # ── Gate in rank-R space (Appendix D) ──
            z_heads = rearrange(z, "b l (h p) -> b l h p", p=args.headdim)
            z_mimo = z_heads.unsqueeze(-1) * self.mimo_z_proj  # (b, l, h, p, R)
            y = y * silu(z_mimo)

            # ── Down-project rank: (b, l, h, p, R) → (b, l, h, p) ──
            y = (y * self.mimo_down).sum(dim=-1)
            y = rearrange(y, "b l h p -> b l (h p)")
            y = self.out_proj(y)

            # ── Build inference cache ──
            # B⊗x after contracting rank R → same shape as SISO state
            last_Bx = torch.einsum(
                "bhnr, bhpr -> bhpn",
                B[:, -1], x_mimo[:, -1],
            )  # (batch, nheads, headdim, d_state)

        else:
            # ── SISO path ──
            # ── Add head-specific BC bias ──
            B = rearrange(B, "b l n -> b l 1 n") + self.B_bias  # (b, l, nheads, d_state)
            C = rearrange(C, "b l n -> b l 1 n") + self.C_bias

            # ── Apply cumulative rotation to B and C (Eq. 8 / Eq. 9) ──
            B = apply_rope(B, cum_angles)
            C = apply_rope(C, cum_angles)

            # ── Trapezoidal SSD (two-SSD decomposition) ──
            y_gamma, state_gamma = ssd(
                x * gamma.unsqueeze(-1), dA, B, C,
                args.chunk_size, device=self.device,
            )

            B_prev = F.pad(B[:, :-1], (0, 0, 0, 0, 1, 0))  # shifted right
            x_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))

            y_beta, state_beta = ssd(
                x_prev * beta.unsqueeze(-1), dA, B_prev, C,
                args.chunk_size, device=self.device,
            )

            # Sum the two contributions
            y = y_gamma + y_beta            # (batch, seqlen, nheads, headdim)
            ssm_state = state_gamma + state_beta

            # ── Skip connection: y = y + D * x ──
            y = y + x * self.D.unsqueeze(-1)

            # ── Gate and project output ──
            y = rearrange(y, "b l h p -> b l (h p)")
            y = y * silu(z)
            y = self.out_proj(y)

            # ── Build inference cache ──
            last_Bx = torch.einsum(
                "bhn, bhp -> bhpn",
                B[:, -1], x[:, -1],
            )  # (batch, nheads, headdim, d_state)

        last_cum_angle = cum_angles[:, -1:]  # (batch, 1, nheads, d_state//2)
        h_new = InferenceCache(
            ssm_state=ssm_state,
            prev_Bx=last_Bx,
            cum_angle=last_cum_angle.squeeze(1),  # (batch, nheads, d_state//2)
        )
        return y, h_new

    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """
        Single-token inference step implementing the Mamba-3 recurrence (Eq. 9):
            h_t = α_t h_{t-1} + β_t B̄_{t-1} x_{t-1} + γ_t B̄_t x_t
            y_t = C̄_t^T h_t

        Arguments
            u: (batch, 1, d_model)
            h: current inference cache (ssm_state, prev_Bx, cum_angle)

        Return (y, h_new)
            y: (batch, 1, d_model)
            h_new: updated inference cache
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"
        args = self.args

        A = -torch.exp(self.A_log)  # (nheads,)

        # ── Project ──
        proj = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, x, B, C, dt, lam, theta = torch.split(
            proj,
            [
                args.d_inner,
                args.d_inner,
                self.bc_dim,
                self.bc_dim,
                args.nheads,
                args.nheads,
                args.d_state // 2,
            ],
            dim=-1,
        )

        # ── Discretization ──
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        lam = torch.sigmoid(lam)             # (batch, nheads)

        # ── QK-Norm ──
        B = self.B_norm(B)  # (batch, bc_dim)
        C = self.C_norm(C)

        # ── Data-Dependent RoPE ──
        # Update cumulative angle: Θ_{t}[h,j] = Θ_{t-1}[h,j] − Δ_t[h] * θ_t[j]
        raw_angle = (
            dt.unsqueeze(-1)                         # (batch, nheads, 1)
            * theta.unsqueeze(1)                     # (batch, 1, d_state//2)
        )  # (batch, nheads, d_state//2)
        new_cum_angle = h.cum_angle - raw_angle  # (batch, nheads, d_state//2)

        # ── Trapezoidal coefficients (Eq. 4) ──
        dA = dt * A  # (batch, nheads)
        alpha = torch.exp(dA)  # decay
        beta = (1 - lam) * dt * alpha  # left-endpoint coefficient
        gamma = lam * dt  # right-endpoint coefficient

        # ── Reshape x: (batch, nheads, headdim) ──
        x = rearrange(x, "b (h p) -> b h p", p=args.headdim)

        if args.use_mimo:
            R = args.mimo_rank

            # ── Reshape B, C: (batch, d_state*R) → (batch, d_state, R) ──
            B = rearrange(B, "b (n r) -> b n r", r=R)
            C = rearrange(C, "b (n r) -> b n r", r=R)
            # Broadcast to heads + bias: (batch, 1, d_state, R) + (nheads, d_state, R)
            B = B.unsqueeze(1) + self.B_bias  # (batch, nheads, d_state, R)
            C = C.unsqueeze(1) + self.C_bias

            # ── Apply RoPE on d_state dim for each rank ──
            B = rearrange(B, "b h n r -> b h r n")
            C = rearrange(C, "b h n r -> b h r n")
            B = apply_rope(B, new_cum_angle.unsqueeze(2))  # broadcast over rank
            C = apply_rope(C, new_cum_angle.unsqueeze(2))
            B = rearrange(B, "b h r n -> b h n r")  # (batch, nheads, d_state, R)
            C = rearrange(C, "b h r n -> b h n r")

            # ── Expand x to rank R ──
            x_mimo = x.unsqueeze(-1) * self.mimo_x_proj  # (batch, nheads, headdim, R)

            # ── MIMO state update: B @ X^T contracts rank R ──
            BX = torch.einsum("bhnr, bhpr -> bhpn", B, x_mimo)  # (batch, nheads, P, N)

            new_ssm_state = (
                h.ssm_state * rearrange(alpha, "b h -> b h 1 1")
                + h.prev_Bx * rearrange(beta, "b h -> b h 1 1")
                + BX * rearrange(gamma, "b h -> b h 1 1")
            )

            # ── Output: H^T @ C → (headdim, R) per head ──
            y = torch.einsum("bhpn, bhnr -> bhpr", new_ssm_state, C)

            # ── Skip connection in rank-R space ──
            y = y + (x * rearrange(self.D, "h -> 1 h 1")).unsqueeze(-1)

            # ── Gate in rank-R space ──
            z_heads = rearrange(z, "b (h p) -> b h p", p=args.headdim)
            z_mimo = z_heads.unsqueeze(-1) * self.mimo_z_proj  # (batch, nheads, P, R)
            y = y * silu(z_mimo)

            # ── Down-project rank ──
            y = (y * self.mimo_down).sum(dim=-1)  # (batch, nheads, headdim)
            y = rearrange(y, "b h p -> b (h p)")
            y = self.out_proj(y)

            h_new = InferenceCache(
                ssm_state=new_ssm_state,
                prev_Bx=BX,
                cum_angle=new_cum_angle,
            )
            return y.unsqueeze(1), h_new

        else:
            # ── SISO path ──
            B = B.unsqueeze(1) + self.B_bias  # (batch, nheads, d_state)
            C = C.unsqueeze(1) + self.C_bias
            B = apply_rope(B, new_cum_angle)
            C = apply_rope(C, new_cum_angle)

            Bx = torch.einsum("bhn, bhp -> bhpn", B, x)  # outer product

            new_ssm_state = (
                h.ssm_state * rearrange(alpha, "b h -> b h 1 1")
                + h.prev_Bx * rearrange(beta, "b h -> b h 1 1")
                + Bx * rearrange(gamma, "b h -> b h 1 1")
            )

            y = torch.einsum("bhpn, bhn -> bhp", new_ssm_state, C)
            y = y + rearrange(self.D, "h -> h 1") * x

            y = rearrange(y, "b h p -> b (h p)")
            y = y * silu(z)
            y = self.out_proj(y)

            h_new = InferenceCache(
                ssm_state=new_ssm_state,
                prev_Bx=Bx,
                cum_angle=new_cum_angle,
            )
            return y.unsqueeze(1), h_new


# ──────────────────────────────────────────────────────────────────────────────
# SwiGLU MLP Block (Llama-style)
# ──────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network as used in the Llama architecture.

    SwiGLU(x) = W_down(SiLU(W_gate(x)) ⊙ W_up(x))
    """

    def __init__(self, d_model: int, d_inner: int, device: Device = None):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_inner, bias=False, device=device)
        self.w_up = nn.Linear(d_model, d_inner, bias=False, device=device)
        self.w_down = nn.Linear(d_inner, d_model, bias=False, device=device)

    def forward(self, x: Tensor) -> Tensor:
        # Shape: (batch, seqlen, d_model) → (batch, seqlen, d_model)
        return self.w_down(silu(self.w_gate(x)) * self.w_up(x))


# ──────────────────────────────────────────────────────────────────────────────
# Data-Dependent RoPE (Proposition 3, Section 3.2)
# ──────────────────────────────────────────────────────────────────────────────

def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Apply rotary position embedding with data-dependent angles.

    The complex SSM's block-diagonal rotation matrix R_t is absorbed into B and C
    via the "RoPE trick" (Proposition 3). The cumulative rotation angles are
    data-dependent (projected from input), unlike standard RoPE which uses fixed
    frequency schedules.

    Arguments
        x: (..., d_state)           — B or C projection to rotate
        angles: (..., d_state // 2) — cumulative rotation angles

    Return
        rotated x with same shape
    """
    # Split into pairs: even dims (2j) and odd dims (2j+1)
    x1 = x[..., 0::2]  # (..., d_state // 2) — even indices
    x2 = x[..., 1::2]  # (..., d_state // 2) — odd indices

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    # 2D rotation per pair: R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]
    x_rot_even = cos_a * x1 - sin_a * x2
    x_rot_odd = sin_a * x1 + cos_a * x2

    # Interleave even and odd back together
    # Stack along last dim then flatten: (..., d_state//2, 2) → (..., d_state)
    return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)


# ──────────────────────────────────────────────────────────────────────────────
# Structured State Space Duality (SSD) — Core Algorithm
# ──────────────────────────────────────────────────────────────────────────────
# The SSD algorithm is the same as in Mamba-2. The trapezoidal modification is
# handled externally via the two-SSD decomposition (see Mamba3.forward).
# ──────────────────────────────────────────────────────────────────────────────

def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    exp(segsum(A)) produces a 1-semiseparable matrix (lower-triangular decay mask),
    which is equivalent to a scalar SSM's cumulative decay products.

    segsum(x)[..., i, j] = Σ_{k=j+1}^{i} x[..., k]   for i ≥ j, else −∞

    Source: Mamba-2 (Dao & Gu, 2024)
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structured State Space Duality (SSD) — the chunked parallel algorithm.

    This is the same SSD algorithm from Mamba-2. In Mamba-3, the trapezoidal
    discretization is handled externally via a two-SSD decomposition: the caller
    runs this function twice (once for the γ term, once for the β term) with
    appropriately pre-scaled and pre-shifted inputs, then sums the results.

    Arguments
        x: (batch, seqlen, n_heads, d_head) — pre-scaled SSM input
        A: (batch, seqlen, n_heads) — log-decay rates (Δ * A, already negative)
        B: (batch, seqlen, n_heads, d_state) — input projection (per-head in Mamba-3)
        C: (batch, seqlen, n_heads, d_state) — output projection (per-head in Mamba-3)
        chunk_size: int — partition size Q

    Return
        y: (batch, seqlen, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)

    Source: Mamba-2 blog post & reference implementation
        https://tridao.me/blog/2024/mamba2-part3-algorithm/
    """
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    # ── Rearrange into chunks ──
    # (batch, seqlen, ...) → (batch, n_chunks, chunk_size, ...)
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # ── Step 1: Intra-chunk output (diagonal blocks) ──
    # Quadratic attention-like computation within each chunk of size Q
    L = torch.exp(segsum(A, device=device))  # (batch, nheads, n_chunks, Q, Q) decay mask
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # ── Step 2: Compute per-chunk states (B-terms for low-rank factorization) ──
    # Decay from each position to the end of its chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # ── Step 3: Inter-chunk SSM recurrence (A-terms) ──
    # Propagate accumulated states across chunk boundaries
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # ── Step 4: State-to-output per chunk (C-terms) ──
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # ── Combine intra-chunk and inter-chunk outputs ──
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


def ssd_mimo(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structured State Space Duality — MIMO variant (Appendix D).

    The MIMO formulation generalises SISO by replacing the outer-product-based
    state update (b ⊗ x) with a matrix-product-based update (B @ X^T). The
    MIMO rank R is orthogonal to the sequence dimension, so the Mamba-2 chunked
    SSD algorithm applies with modified einsums that contract over R.

    State shape is unchanged from SISO: (batch, n_heads, d_head, d_state).

    Arguments
        x: (batch, seqlen, n_heads, d_head, mimo_rank) — rank-expanded input
        A: (batch, seqlen, n_heads) — log-decay rates (Δ * A)
        B: (batch, seqlen, n_heads, d_state, mimo_rank) — input projection
        C: (batch, seqlen, n_heads, d_state, mimo_rank) — output projection
        chunk_size: int — partition size Q

    Return
        y: (batch, seqlen, n_heads, d_head, mimo_rank)
        final_state: (batch, n_heads, d_head, d_state)
    """
    assert x.shape[1] % chunk_size == 0, (
        f"seqlen ({x.shape[1]}) must be divisible by chunk_size ({chunk_size})"
    )

    # ── Rearrange into chunks ──
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # ── Step 1: Intra-chunk output (MIMO version) ──
    # Contracts input rank r between B and x; C provides output rank q
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhnq, bcshnr, bhcls, bcshpr -> bclhpq", C, B, L, x)

    # ── Step 2: Per-chunk states (contract input rank r between B and x) ──
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhnr, bhcl, bclhpr -> bchpn", B, decay_states, x)

    # ── Step 3: Inter-chunk SSM recurrence (unchanged from SISO) ──
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device)
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # ── Step 4: State-to-output per chunk (C has output rank) ──
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhnr, bchpn, bhcl -> bclhpr", C, states, state_decay_out)

    # ── Combine intra-chunk and inter-chunk outputs ──
    Y = rearrange(Y_diag + Y_off, "b c l h p r -> b (c l) h p r")

    return Y, final_state


# ──────────────────────────────────────────────────────────────────────────────
# Utility Modules
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    In Mamba-3 this serves two roles:
      1. Pre-normalization before each block (standard RMSNorm, no gating)
      2. QK-Normalization on B, C projections (standard RMSNorm, no gating)

    Paper: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x: Tensor) -> Tensor:
    """Sigmoid Linear Unit (SiLU) / Swish activation.

    Defined manually for compatibility with MPS (Apple Silicon).
    """
    return x * F.sigmoid(x)


# ──────────────────────────────────────────────────────────────────────────────
# Model Creation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def create_toy_model(
    d_model: int = 128,
    n_layer: int = 4,
    vocab_size: int = 256,
    device: Device = None,
    use_mimo: bool = False,
    mimo_rank: int = 4,
) -> Mamba3LMHeadModel:
    """Create a small Mamba-3 model for testing and debugging.

    Default configuration: ~3M parameters, suitable for 18GB M3 MacBook.
    Set use_mimo=True to create a MIMO variant with the specified rank.
    """
    if device is None:
        device = get_device()
    args = Mamba3Config(
        d_model=d_model,
        n_layer=n_layer,
        d_state=64,       # Smaller state for toy model
        headdim=32,        # Smaller heads
        chunk_size=32,
        vocab_size=vocab_size,
        use_mimo=use_mimo,
        mimo_rank=mimo_rank,
    )
    model = Mamba3LMHeadModel(args, device=device)
    # Initialize parameters
    for name, p in model.named_parameters():
        if "A_log" in name:
            nn.init.uniform_(p, -4, -1)  # A is negative, log(A) ∈ [-4, -1]
        elif "D" in name and p.dim() == 1:
            nn.init.ones_(p)
        elif "dt_bias" in name:
            nn.init.uniform_(p, 0.001, 0.1)
        elif "B_bias" in name or "C_bias" in name:
            pass  # Already initialized to ones
        elif "mimo" in name:
            pass  # Already initialized in __init__ (ones or ones/R)
        elif p.dim() >= 2:
            nn.init.normal_(p, std=0.02)
    return model

