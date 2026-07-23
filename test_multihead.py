"""Test cases for Multi-Head Attention."""
import torch
from main import MultiHeadAttention


def test_05_mha_output_shape():
    """Test 5: output shape is (B, T, d_model)."""
    B, T, d_model, num_heads = 2, 7, 16, 4
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out, attn = mha(x, x, x)
    print(f"[Test 5] input: ({B}, {T}, {d_model}) | num_heads={num_heads}")
    print(f"        output: {out.shape} | expected: ({B}, {T}, {d_model})")
    print()


def test_06_mha_attention_shape():
    """Test 6: attention weights per head, all rows sum to 1."""
    B, T, d_model, num_heads = 2, 6, 12, 3
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out, attn = mha(x, x, x)
    print(f"[Test 6] attn shape: {attn.shape} | expected: ({B}, {num_heads}, {T}, {T})")
    row_sums = attn.sum(dim=-1)
    all_one = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    print(f"        all rows sum to 1: {all_one}")
    print()


def test_07_mha_causal_mask():
    """Test 7: causal mask blocks future tokens (upper triangle all zeros)."""
    B, T, d_model, num_heads = 1, 5, 8, 2
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
    out, attn = mha(x, x, x, mask)
    upper_tri = attn[0, 0].triu(diagonal=1)
    print(f"[Test 7] causal mask, upper triangle sum: {upper_tri.sum().item():.2e}")
    print(f"        expected: 0.0")
    print()


def test_08_mha_cross_attention():
    """Test 8: cross-attention (Q from one seq, K/V from another)."""
    B, T_q, T_kv, d_model, num_heads = 2, 5, 7, 16, 4
    mha = MultiHeadAttention(d_model, num_heads)
    q = torch.randn(B, T_q, d_model)
    kv = torch.randn(B, T_kv, d_model)
    out, attn = mha(q, kv, kv)
    print(f"[Test 8] cross-attention: q=(B,{T_q},d) + kv=(B,{T_kv},d)")
    print(f"        output: {out.shape} | expected: ({B}, {T_q}, {d_model})")
    print(f"        attn:   {attn.shape} | expected: ({B}, {num_heads}, {T_q}, {T_kv})")
    print()


def test_09_mha_gradient_flow():
    """Test 9: all parameters receive gradients after backward."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out, _ = mha(x, x, x)
    loss = out.sum()
    loss.backward()
    no_grad = [name for name, p in mha.named_parameters() if p.grad is None]
    print(f"[Test 9] params without gradient: {no_grad if no_grad else 'none (all good!)'}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  MultiHeadAttention tests")
    print("=" * 50)
    print()
    test_05_mha_output_shape()
    test_06_mha_attention_shape()
    test_07_mha_causal_mask()
    test_08_mha_cross_attention()
    test_09_mha_gradient_flow()
