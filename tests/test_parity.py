"""
test_parity.py — Parity State-Tracking Test
============================================

Tests the "killer feature" of Mamba-3: state-tracking via data-dependent RoPE.

The Mamba-3 paper (Table 4b) shows that Mamba-2 completely fails at parity
(random guessing ≈ 50%), while Mamba-3 achieves 100%. This is because the
complex-valued SSM (Proposition 2) introduces rotational dynamics that can
represent modular counting — something real-valued scalar transitions
fundamentally cannot.

If this test reaches ~100% accuracy, the Complex SSM / RoPE implementation
is mathematically correct.

Reference: Section 4.3, Table 4b, Appendix E (State-Tracking Synthetics)
"""

import torch
import torch.nn.functional as F
from mamba3 import Mamba3Config, Mamba3LMHeadModel, get_device

torch.manual_seed(42)
device = get_device()

# ── Model Configuration ──
# Paper (Appendix E): 1-layer for Parity, d_state=64, d_model ∈ {32, 64}
# chunk_size=8 allows short-sequence curriculum training
NUM_TOKENS = 3  # 0=unused, 1=bit-zero, 2=bit-one (padded to 16 internally)

args = Mamba3Config(
    d_model=64,
    n_layer=1,
    d_state=64,
    headdim=32,
    chunk_size=8,        # small chunks to allow short-sequence curriculum
    vocab_size=NUM_TOKENS,
)
model = Mamba3LMHeadModel(args, device=device)

# ── Initialize Weights (following mamba3.py conventions) ──
for name, p in model.named_parameters():
    if "A_log" in name:
        torch.nn.init.uniform_(p, -4, -1)
    elif "D" in name and p.dim() == 1:
        torch.nn.init.ones_(p)
    elif "dt_bias" in name:
        torch.nn.init.uniform_(p, 0.001, 0.1)
    elif "B_bias" in name or "C_bias" in name:
        pass  # Already initialized to ones (paper: Appendix G)
    elif p.dim() >= 2:
        torch.nn.init.normal_(p, std=0.02)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
batch_size = 64
n_params = sum(p.numel() for p in model.parameters())

print("=" * 62)
print("  Parity State-Tracking Test (Mamba-3 Paper, Table 4b)")
print("=" * 62)
print(f"  Device:   {device}")
print(f"  Model:    1-layer, d_model=64, d_state=64 ({n_params:,} params)")
print(f"  Task:     predict running parity of binary input")
print(f"  Expected: Mamba-3 → ~100%  |  Mamba-2 (no RoPE) → ~50%")
print()

# ── Curriculum Training ──
# Paper (Appendix E): "sequence length curriculum from 3→40 to 160"
# Simplified curriculum: 8 → 16 → 32
curriculum = [
    (8,  1000),
    (16, 1000),
    (32, 1500),
]

for seq_len, n_steps in curriculum:
    print(f"── Curriculum: seq_len={seq_len}, {n_steps} steps ──")
    model.train()
    for step in range(n_steps):
        # Random binary input: tokens ∈ {1, 2}
        x = torch.randint(1, 3, (batch_size, seq_len), device=device)

        # Running parity target:
        #   y[t] = 1  if even number of 2s in x[0..t]
        #   y[t] = 2  if odd  number of 2s in x[0..t]
        y = 1 + (x == 2).cumsum(dim=1) % 2

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits[:, :, :NUM_TOKENS].reshape(-1, NUM_TOKENS),
            y.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 250 == 0 or step == n_steps - 1:
            with torch.no_grad():
                preds = logits[:, :, :NUM_TOKENS].argmax(dim=-1)
                acc = (preds == y).float().mean().item() * 100
            print(f"  Step {step:4d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%")
    print()

# ── Length Generalization ──
# Paper evaluates at lengths longer than training to test extrapolation
print("── Length Generalization (trained up to 32) ──")
model.eval()
for test_len in [32, 40, 48, 64]:
    x = torch.randint(1, 3, (256, test_len), device=device)
    y = 1 + (x == 2).cumsum(dim=1) % 2
    with torch.no_grad():
        logits, _ = model(x)
        preds = logits[:, :, :NUM_TOKENS].argmax(dim=-1)
        acc = (preds == y).float().mean().item() * 100
    print(f"  Length {test_len:3d}: {acc:.1f}%")

print()
print("Accuracy ≈ 100% proves the Complex SSM / RoPE enables state-tracking.")
print("Mamba-2 (no RoPE) would score ≈ 50% (random guessing).")
