"""Test MIMO implementation: forward/step parity + regression check on SISO."""

import torch
from mamba3 import create_toy_model, get_device

device = get_device()
torch.manual_seed(42)

seqlen = 64
tokens = torch.randint(0, 32, (1, seqlen), device=device)
prefix_len = 32
threshold = 1e-3


def test_fwd_step_parity(label, model):
    """Compare full forward vs prefix-forward + step-by-step."""
    model.eval()
    with torch.no_grad():
        logits_fwd, cache_fwd = model(tokens)

        logits_prefix, h = model(tokens[:, :prefix_len])
        step_logits = [logits_prefix]
        for t in range(prefix_len, seqlen):
            logit_t, h = model(tokens[:, t : t + 1], h)
            step_logits.append(logit_t)
        logits_step = torch.cat(step_logits, dim=1)

    diff = (logits_fwd - logits_step).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # cache_fwd is a list of InferenceCache (one per layer)
    c0 = cache_fwd[0] if isinstance(cache_fwd, list) else cache_fwd
    print(f"=== {label} Forward/Step Consistency ===")
    print(f"  Logits shape:     {logits_fwd.shape}")
    print(f"  State shape:      {c0.ssm_state.shape}")
    print(f"  prev_Bx shape:    {c0.prev_Bx.shape}")
    print(f"  cum_angle shape:  {c0.cum_angle.shape}")
    print(f"  Max abs diff:     {max_diff:.2e}")
    print(f"  Mean abs diff:    {mean_diff:.2e}")
    ok = max_diff < threshold
    print(f"  {'PASS' if ok else 'FAIL'}: max diff {max_diff:.2e} {'<' if ok else '>='} {threshold}")
    print()
    return ok


# ── MIMO test ──
model_mimo = create_toy_model(
    d_model=64, n_layer=1, vocab_size=32, use_mimo=True, mimo_rank=4
)
ok_mimo = test_fwd_step_parity("MIMO (R=4)", model_mimo)

# ── SISO regression ──
model_siso = create_toy_model(
    d_model=64, n_layer=1, vocab_size=32, use_mimo=False
)
ok_siso = test_fwd_step_parity("SISO", model_siso)

# ── Parameter count comparison ──
n_mimo = sum(p.numel() for p in model_mimo.parameters())
n_siso = sum(p.numel() for p in model_siso.parameters())
print(f"Parameter count: SISO={n_siso:,}  MIMO(R=4)={n_mimo:,}  delta={n_mimo - n_siso:,}")

if ok_mimo and ok_siso:
    print("\nAll tests passed!")
else:
    print("\nSome tests FAILED!")
    exit(1)
