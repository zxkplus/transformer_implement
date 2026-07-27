"""Test cases for Encoder Block and Transformer Encoder."""
import torch
from main import EncoderBlock, TransformerEncoder


def test_encoder_output_shape():
    """Test 1: EncoderBlock preserves (B, T, d_model)."""
    B, T, d_model, num_heads = 2, 10, 16, 4
    enc = EncoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out = enc(x)
    assert out.shape == (B, T, d_model), f"Expected ({B}, {T}, {d_model}), got {out.shape}"
    print(f"[Encoder Test 1] output shape: {out.shape} | expected: ({B}, {T}, {d_model})")
    print()


def test_encoder_gradient_flow():
    """Test 2: gradients flow through all parameters."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    enc = EncoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    no_grad = [name for name, p in enc.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters without gradient: {no_grad}"
    print(f"[Encoder Test 2] params without gradient: {'none (all good!)' if not no_grad else no_grad}")
    print()


def test_encoder_multi_layer_shape():
    """Test 3: stacked TransformerEncoder preserves shape."""
    B, T, d_model, num_heads, num_layers = 2, 10, 16, 4, 3
    enc = TransformerEncoder(d_model, num_heads, num_layers)
    x = torch.randn(B, T, d_model)
    out = enc(x)
    assert out.shape == (B, T, d_model), f"Expected ({B}, {T}, {d_model}), got {out.shape}"
    print(f"[Encoder Test 3] TransformerEncoder ({num_layers} layers) output: {out.shape} | expected: ({B}, {T}, {d_model})")
    print()


def test_encoder_self_attention_mixes_tokens():
    """Test 4: self-attention propagates info across token positions."""
    B, T, d_model, num_heads = 1, 5, 8, 2
    enc = EncoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out1 = enc(x)
    # Spike token 0 significantly
    x2 = x.clone()
    x2[:, 0, :] = 100.0
    out2 = enc(x2)
    # Other tokens' outputs should differ (self-attention propagates)
    diff = (out1[:, 1:, :] - out2[:, 1:, :]).abs().max().item()
    assert diff > 1e-8, f"Self-attention didn't propagate: max diff = {diff:.2e}"
    print(f"[Encoder Test 4] self-attention propagates: max diff at other tokens = {diff:.2e} | expected: > 1e-8")
    print()


def test_encoder_residual_not_identity():
    """Test 5: output differs from input (transformation happens)."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    enc = EncoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out = enc(x)
    diff = (out - x).abs().max().item()
    assert diff > 1e-8, f"Output identical to input, residual may be broken: diff = {diff:.2e}"
    print(f"[Encoder Test 5] |out - in| max: {diff:.4f} | expected: > 1e-8")
    print()


def test_encoder_deterministic():
    """Test 6: same input → same output in eval mode."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    enc = EncoderBlock(d_model, num_heads)
    enc.eval()
    x = torch.randn(B, T, d_model)
    out1 = enc(x)
    out2 = enc(x)
    same = torch.allclose(out1, out2)
    assert same, "Encoder is not deterministic"
    print(f"[Encoder Test 6] deterministic: {same} | expected: True")
    print()


def test_encoder_multi_layer_deeper_than_single():
    """Test 7: 2-layer encoder output differs from 1-layer (stacking works)."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    enc1 = EncoderBlock(d_model, num_heads)
    enc2 = TransformerEncoder(d_model, num_heads, 2)
    x = torch.randn(B, T, d_model)
    # Copy weights from enc1 to the first layer of enc2
    enc2_block_0 = enc2.encodesequence[0]
    with torch.no_grad():
        enc2_block_0.multihead.Qline.weight.copy_(enc1.multihead.Qline.weight)
        enc2_block_0.multihead.Kline.weight.copy_(enc1.multihead.Kline.weight)
        enc2_block_0.multihead.Vline.weight.copy_(enc1.multihead.Vline.weight)
        enc2_block_0.multihead.outline.weight.copy_(enc1.multihead.outline.weight)
        enc2_block_0.ffn.line1.weight.copy_(enc1.ffn.line1.weight)
        enc2_block_0.ffn.line2.weight.copy_(enc1.ffn.line2.weight)
    out1 = enc1(x)
    out2 = enc2(x)
    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-8, f"Two layers should differ from one layer: diff = {diff:.2e}"
    print(f"[Encoder Test 7] |1-layer - 2-layer| max: {diff:.4f} | expected: > 1e-8")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  Encoder Block tests")
    print("=" * 50)
    print()
    test_encoder_output_shape()
    test_encoder_gradient_flow()
    test_encoder_multi_layer_shape()
    test_encoder_self_attention_mixes_tokens()
    test_encoder_residual_not_identity()
    test_encoder_deterministic()
    test_encoder_multi_layer_deeper_than_single()
