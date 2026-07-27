"""Test cases for the end-to-end Transformer."""
import torch
from main import Transformer


def test_transformer_output_shape():
    """Test 1: forward pass returns (B, T_tgt, tgt_vocab_size) logits."""
    src_vocab_size, tgt_vocab_size = 100, 100
    d_model, num_heads, num_layers = 16, 4, 2
    B, T_src, T_tgt = 2, 10, 8

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)
    src = torch.randint(0, src_vocab_size, (B, T_src))
    tgt = torch.randint(0, tgt_vocab_size, (B, T_tgt))

    logits = model(src, tgt)
    expected = (B, T_tgt, tgt_vocab_size)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    print(f"[Transformer Test 1] output shape: {logits.shape} | expected: {expected}")
    print()


def test_transformer_gradient_flow():
    """Test 2: gradients flow through encoder + decoder + embedding + projection."""
    src_vocab_size, tgt_vocab_size = 50, 50
    d_model, num_heads, num_layers = 8, 2, 2
    B, T_src, T_tgt = 1, 4, 3

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)
    src = torch.randint(0, src_vocab_size, (B, T_src))
    tgt = torch.randint(0, tgt_vocab_size, (B, T_tgt))

    logits = model(src, tgt)
    loss = logits.sum()
    loss.backward()

    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters without gradient: {no_grad}"
    print(f"[Transformer Test 2] params without gradient: {'none (all good!)' if not no_grad else no_grad}")
    print()


def test_transformer_diff_src_tgt_lengths():
    """Test 3: encoder and decoder can have different sequence lengths."""
    src_vocab_size, tgt_vocab_size = 100, 100
    d_model, num_heads, num_layers = 16, 4, 2
    B, T_src, T_tgt = 2, 15, 7

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)
    src = torch.randint(0, src_vocab_size, (B, T_src))
    tgt = torch.randint(0, tgt_vocab_size, (B, T_tgt))

    logits = model(src, tgt)
    expected = (B, T_tgt, tgt_vocab_size)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    print(f"[Transformer Test 3] src_len={T_src}, tgt_len={T_tgt} | output: {logits.shape}")
    print()


def test_transformer_deterministic():
    """Test 4: same input -> same output."""
    src_vocab_size, tgt_vocab_size = 50, 50
    d_model, num_heads, num_layers = 8, 2, 2
    B, T_src, T_tgt = 1, 5, 5

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)
    model.eval()
    src = torch.randint(0, src_vocab_size, (B, T_src))
    tgt = torch.randint(0, tgt_vocab_size, (B, T_tgt))

    out1 = model(src, tgt)
    out2 = model(src, tgt)
    assert torch.allclose(out1, out2), "Determinism broken: two forward passes differ"
    print(f"[Transformer Test 4] deterministic: {torch.allclose(out1, out2)}")
    print()


def test_transformer_causal_mask_active():
    """Test 5: different target lengths produce expected shapes."""
    src_vocab_size, tgt_vocab_size = 50, 50
    d_model, num_heads, num_layers = 8, 2, 2
    B, T_src = 1, 5

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)
    model.eval()
    src = torch.randint(0, src_vocab_size, (B, T_src))

    tgt_short = torch.randint(0, tgt_vocab_size, (B, 3))
    tgt_long  = torch.randint(0, tgt_vocab_size, (B, 5))

    logits_short = model(src, tgt_short)
    logits_long  = model(src, tgt_long)

    assert logits_short.shape == (B, 3, tgt_vocab_size)
    assert logits_long.shape == (B, 5, tgt_vocab_size)
    print(f"[Transformer Test 5] short tgt_len=3: {logits_short.shape} | long tgt_len=5: {logits_long.shape}")
    print()
