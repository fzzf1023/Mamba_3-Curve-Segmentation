"""
demo.py — Mamba-3 Minimal Demo
================================

Demonstrates training and generation with a toy Mamba-3 language model.
Trains a small model on a dummy dataset to verify the architecture works
end-to-end, including:

  1. Forward pass with chunked SSD (trapezoidal discretization)
  2. Autoregressive generation via inference step
  3. Comparison of forward vs step-by-step consistency
  4. Training loop with loss backpropagation
  5. MIMO (Multiple Input Multiple Output) forward pass

Designed to run on any hardware: CUDA, Apple Silicon (MPS), or CPU.
"""

import gc
import time

import torch
import torch.nn.functional as F

from mamba3 import (
    Mamba3Config,
    Mamba3LMHeadModel,
    InferenceCache,
    get_device,
)


def demo_architecture():
    """Demonstrate the Mamba-3 architecture with detailed shape annotations."""
    device = get_device()
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Mamba-3 Minimal — Architecture Demo                       ║")
    print(f"║  Device: {str(device):<51}║")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")

    # ── Model Configuration ──
    args = Mamba3Config(
        d_model=128,
        n_layer=4,
        d_state=64,
        headdim=32,
        chunk_size=32,
        vocab_size=256,
    )
    model = Mamba3LMHeadModel(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    print("Model Configuration:")
    print(f"  d_model      = {args.d_model}")
    print(f"  n_layer      = {args.n_layer}")
    print(f"  d_state      = {args.d_state}")
    print(f"  d_inner      = {args.d_inner} (expand={args.expand} × d_model)")
    print(f"  nheads       = {args.nheads}")
    print(f"  headdim      = {args.headdim}")
    print(f"  chunk_size   = {args.chunk_size}")
    print(f"  d_mlp_inner  = {args.d_mlp_inner}")
    print(f"  vocab_size   = {args.vocab_size}")
    print(f"  Parameters   = {n_params:,}\n")

    # ── Initialize parameters ──
    for name, p in model.named_parameters():
        if "A_log" in name:
            torch.nn.init.uniform_(p, -4, -1)
        elif "D" in name and p.dim() == 1:
            torch.nn.init.ones_(p)
        elif "dt_bias" in name:
            torch.nn.init.uniform_(p, 0.001, 0.1)
        elif p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.02)

    # ── Layer Architecture ──
    print("Layer Architecture (Llama-style, per layer):")
    print("  ┌─ RMSNorm ─→ Mamba3 SSM Block ─→ Residual Add")
    print("  │    (pre-norm)    ↓")
    print("  │    Input Projection → split (z, x, B, C, Δ, λ, θ)")
    print("  │    B, C → QK-Norm → +Bias → RoPE rotation")
    print("  │    Trapezoidal SSD (γ term + β term)")
    print("  │    y + D·x (skip) → y · SiLU(z) (gate) → Out Projection")
    print("  │")
    print("  └─ RMSNorm ─→ SwiGLU MLP ─→ Residual Add")
    print(f"       (pre-norm)   W_gate: {args.d_model}→{args.d_mlp_inner}")
    print()

    return model, args


def demo_forward_pass(model, args):
    """Test the forward pass (training mode with chunked SSD)."""
    device = next(model.parameters()).device
    print("─── Forward Pass (Training Mode) ─────────────────────────────")

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    print(f"  Input shape:  ({batch_size}, {seq_len})")

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, h = model(input_ids)
    dt = time.perf_counter() - t0

    print(f"  Output shape: {logits.shape}")
    print(f"  Time: {dt*1000:.1f}ms")

    # Verify gradient flow
    model.train()
    input_ids_grad = torch.randint(0, args.vocab_size, (1, 32), device=device)
    logits_grad, _ = model(input_ids_grad)
    loss = logits_grad.sum()
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    print(f"  Gradient check: {len(grad_norms)} parameters have gradients ✓")
    model.zero_grad()
    model.eval()
    print()
    return h


def demo_inference_step(model, args):
    """Test single-token inference (autoregressive decoding)."""
    device = next(model.parameters()).device
    print("─── Inference Step (Decode Mode) ──────────────────────────────")

    batch_size = 1
    h = [InferenceCache.alloc(batch_size, args, device=device) for _ in range(args.n_layer)]

    # Process 10 tokens one at a time
    t0 = time.perf_counter()
    for t in range(10):
        token = torch.randint(0, args.vocab_size, (batch_size, 1), device=device)
        with torch.no_grad():
            logits, h = model(token, h)
    dt = time.perf_counter() - t0

    print(f"  10 inference steps: {dt*1000:.1f}ms ({dt*100:.1f}ms/token)")
    print(f"  SSM state shape: {h[0].ssm_state.shape}")
    print(f"  Prev B⊗x shape: {h[0].prev_Bx.shape}")
    print(f"  Cum angle shape: {h[0].cum_angle.shape}")
    print()


def demo_consistency(model, args):
    """Verify that chunked forward and step-by-step inference produce identical results."""
    device = next(model.parameters()).device
    print("─── Forward vs Step-by-Step Consistency ───────────────────────")

    torch.manual_seed(42)
    seq_len = 64
    test_seq = torch.randint(0, args.vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        # Chunked forward (training path)
        logits_fwd, _ = model(test_seq)

        # Step-by-step (inference path)
        h_step = [InferenceCache.alloc(1, args, device=device) for _ in range(args.n_layer)]
        logits_list = []
        for t in range(seq_len):
            out, h_step = model(test_seq[:, t:t+1], h_step)
            logits_list.append(out)
        logits_seq = torch.cat(logits_list, dim=1)

    max_diff = (logits_fwd - logits_seq).abs().max().item()
    mean_diff = (logits_fwd - logits_seq).abs().mean().item()

    print(f"  Sequence length: {seq_len}")
    print(f"  Max absolute difference:  {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    if max_diff < 1e-2:
        print("  ✓ Outputs are consistent!")
    else:
        print("  ⚠ Outputs differ — check implementation.")
    print()


def demo_training_loop(model, args, n_steps=50):
    """Train the model on a simple next-token prediction task."""
    device = next(model.parameters()).device
    print("─── Training Loop Demo ───────────────────────────────────────")

    # Create a simple repeating pattern for the model to learn
    torch.manual_seed(123)
    pattern = torch.randint(0, args.vocab_size, (8,), device=device)
    train_data = pattern.repeat(args.chunk_size * 2 // len(pattern) + 1)[:args.chunk_size * 2]
    train_data = train_data.unsqueeze(0)  # (1, seq_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    model.train()

    print(f"  Training on repeating pattern of length {len(pattern)}")
    print(f"  Sequence length: {train_data.shape[1]}")
    print(f"  Steps: {n_steps}")
    print()

    losses = []
    t0 = time.perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad()
        logits, _ = model(train_data)

        # Next-token prediction loss
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, args.vocab_size),
            train_data[:, 1:].reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 10 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    dt = time.perf_counter() - t0
    print(f"\n  Training time: {dt:.1f}s ({dt/n_steps*1000:.0f}ms/step)")
    print(f"  Initial loss: {losses[0]:.4f} → Final loss: {losses[-1]:.4f}")
    if losses[-1] < losses[0]:
        print("  ✓ Model is learning!")
    else:
        print("  ⚠ Loss did not decrease.")

    model.eval()
    print()


def demo_generation(model, args):
    """Generate text autoregressively from a random prompt."""
    device = next(model.parameters()).device
    print("─── Autoregressive Generation ────────────────────────────────")

    prompt = torch.randint(0, args.vocab_size, (5,), device=device)
    print(f"  Prompt tokens: {prompt.tolist()}")
    print(f"  Generating 20 tokens...")

    generated = prompt.tolist()
    for token, h in model.generate(prompt, max_new_length=20, temperature=0.8, top_k=10):
        generated.append(token)

    print(f"  Generated: {generated}")
    print(f"  Total length: {len(generated)} tokens")
    print()


def demo_mps_memory(model, args):
    """Profile GPU memory across chunk sizes to demonstrate bounded memory.

    Shows how the SSD chunk_size parameter trades off memory for compute:
    larger chunks → more memory (quadratic intra-chunk attention) but fewer
    inter-chunk communication steps.
    """
    device = next(model.parameters()).device
    print("─── GPU Memory Profiling ─────────────────────────────────────")

    # --- Set up backend-specific functions ---
    if device.type == "mps":
        sync = torch.mps.synchronize
        get_alloc = lambda: torch.mps.current_allocated_memory() / (1024 ** 2)
        get_reserved = lambda: torch.mps.driver_allocated_memory() / (1024 ** 2)
        clear = torch.mps.empty_cache
        backend = "MPS (Apple Silicon)"
    elif device.type == "cuda":
        sync = lambda: torch.cuda.synchronize(device)
        get_alloc = lambda: torch.cuda.memory_allocated(device) / (1024 ** 2)
        get_reserved = lambda: torch.cuda.memory_reserved(device) / (1024 ** 2)
        clear = torch.cuda.empty_cache
        backend = "CUDA"
    else:
        print("  Skipped — requires MPS or CUDA for GPU memory profiling.\n")
        return

    print(f"  Backend: {backend}")

    gc.collect()
    clear()
    sync()
    baseline = get_alloc()
    reserved = get_reserved()
    print(f"  Model footprint (allocated): {baseline:.2f} MB")
    print(f"  Driver reserved:             {reserved:.2f} MB")
    print()

    # --- Sweep chunk_size to show memory scaling ---
    seq_len = 1024
    original_cs = args.chunk_size

    print(f"  Memory vs chunk_size (batch=1, seq_len={seq_len}):")
    print()
    print(f"  {'chunk_size':>12} | {'Allocated':>11} | {'Δ from base':>12}")
    print(f"  {'-'*12}-+-{'-'*11}-+-{'-'*12}")

    for cs in [16, 32, 64, 128, 256]:
        if seq_len % cs != 0:
            continue

        args.chunk_size = cs
        gc.collect()
        clear()
        sync()

        input_ids = torch.randint(0, args.vocab_size, (1, seq_len), device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        sync()

        peak = get_alloc()
        delta = peak - baseline

        del logits, input_ids
        gc.collect()
        clear()
        sync()

        tag = " ← default" if cs == original_cs else ""
        print(f"  {cs:>12} | {peak:>8.2f} MB | {delta:>9.2f} MB{tag}")

    args.chunk_size = original_cs
    print()
    print("  Larger chunk_size → more memory (O(Q²) intra-chunk attention).")
    print("  Smaller chunk_size → less memory, more inter-chunk overhead.")
    print()


def demo_mimo():
    """Demonstrate the MIMO (Multiple Input Multiple Output) forward pass."""
    device = get_device()
    print("─── MIMO (Multiple Input Multiple Output) ───────────────────────")
    print("  Reference: Appendix D (Rank-R Matrix Product State)")
    
    # ── Model Configuration ──
    # MIMO adds a rank-R dimension to B and C, enabling higher expressivity
    # without increasing the state size (d_state).
    rank = 4
    args = Mamba3Config(
        d_model=128,
        n_layer=1,
        d_state=64,
        headdim=32,
        chunk_size=32,
        vocab_size=256,
        use_mimo=True,
        mimo_rank=rank,
    )
    model = Mamba3LMHeadModel(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"  Configuration: d_model={args.d_model}, d_state={args.d_state}, MIMO Rank={rank}")
    print(f"  Parameters:    {n_params:,} (vs SISO ~86k)")

    # ── Forward Pass ──
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        output, _ = model(input_ids)
    t1 = time.perf_counter()

    print(f"  Input shape:   ({batch_size}, {seq_len})")
    print(f"  Output shape:  {tuple(output.shape)}")
    print(f"  Latency:       {(t1 - t0) * 1000:.2f} ms")
    print("  Status:        MIMO Forward Pass Successful ✓")
    print()


def main():
    print()
    model, args = demo_architecture()
    demo_forward_pass(model, args)
    demo_inference_step(model, args)
    demo_consistency(model, args)
    demo_training_loop(model, args)
    demo_generation(model, args)
    demo_mps_memory(model, args)
    demo_mimo()

    print("═══════════════════════════════════════════════════════════════")
    print("  All demos completed successfully! ✓")
    print("═══════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
