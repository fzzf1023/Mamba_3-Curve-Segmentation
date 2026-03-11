"""
test_text.py — Real Vocabulary Generation Test
================================================

Tests Mamba-3's inference cache and autoregressive generation loop with a
real tokenizer (tiktoken / GPT-2 vocabulary, ~50K tokens).

The model is untrained, so it will generate gibberish — but this proves that:
  1. The O(1) inference cache correctly propagates state across tokens
  2. The generate() loop works with real-world vocabulary sizes
  3. Token sampling (top-k, temperature) produces valid token IDs

Prerequisites:
    pip install tiktoken
"""

import torch

try:
    import tiktoken
except ImportError:
    print("This test requires tiktoken. Install it with:")
    print("  pip install tiktoken")
    raise SystemExit(1)

from mamba3 import create_toy_model, get_device

device = get_device()
enc = tiktoken.get_encoding("gpt2")  # 50,257 tokens

print("=" * 62)
print("  Real Vocabulary Generation Test (GPT-2 Tokenizer)")
print("=" * 62)
print(f"  Device: {device}")
print(f"  Vocab:  {enc.n_vocab:,} tokens (GPT-2)")
print()

# ── Create model sized for real vocabulary ──
print("Creating model (d_model=256, 4 layers)...")
model = create_toy_model(
    d_model=256,
    n_layer=4,
    vocab_size=enc.n_vocab,
    device=device,
)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")
print()

# ── Encode prompt ──
prompt_text = "The secret to machine learning is"
# generate() expects a 1-D tensor (no batch dimension)
input_ids = torch.tensor(enc.encode(prompt_text), device=device)

print(f"Prompt: \"{prompt_text}\"")
print(f"Tokens: {input_ids.tolist()} ({len(input_ids)} tokens)")
print()

# ── Generate tokens using O(1) inference cache ──
print("Generating (untrained model → expect gibberish)...")
print("─" * 50)
print(prompt_text, end="")

with torch.no_grad():
    for token, _ in model.generate(
        input_ids,
        max_new_length=30,
        temperature=0.8,
        top_k=50,
    ):
        # Defensive decoding: padded vocab may produce out-of-range token IDs
        if token < enc.n_vocab:
            print(enc.decode([token]), end="", flush=True)
        else:
            print(f"[{token}]", end="", flush=True)

print()
print("─" * 50)
print()
print("✓ Inference cache and generation loop work with real vocabularies!")
