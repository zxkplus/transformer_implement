"""Test cases for Decoder Block and Transformer Decoder."""
import torch
from main import DecoderBlock, TransformerDecoder, TransformerEncoder


def test_decoder_output_shape():
    """Test 1: DecoderBlock preserves (B, T, d_model)."""
    B, T, d_model, num_heads = 2, 10, 16, 4
    dec = DecoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    enc_out = torch.randn(B, T, d_model)
    out = dec(x, enc_out)
    assert out.shape == (B, T, d_model), f"Expected ({B}, {T}, {d_model}), got {out.shape}"
    print(f"[Decoder Test 1] output shape: {out.shape} | expected: ({B}, {T}, {d_model})")
    print()


def test_decoder_gradient_flow():
    """Test 2: gradients flow through all parameters."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    dec = DecoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    enc_out = torch.randn(B, T, d_model)
    out = dec(x, enc_out)
    loss = out.sum()
    loss.backward()
    no_grad = [name for name, p in dec.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters without gradient: {no_grad}"
    print(f"[Decoder Test 2] params without gradient: {'none (all good!)' if not no_grad else no_grad}")
    print()


def test_decoder_multi_layer_shape():
    """Test 3: stacked TransformerDecoder preserves shape."""
    B, T, d_model, num_heads, num_layers = 2, 10, 16, 4, 3
    dec = TransformerDecoder(d_model, num_heads, num_layers)
    x = torch.randn(B, T, d_model)
    enc_out = torch.randn(B, T, d_model)
    out = dec(x, enc_out)
    assert out.shape == (B, T, d_model), f"Expected ({B}, {T}, {d_model}), got {out.shape}"
    print(f"[Decoder Test 3] TransformerDecoder ({num_layers} layers) output: {out.shape} | expected: ({B}, {T}, {d_model})")
    print()


def test_decoder_causal_mask():
    """Test 4: causal mask forward pass succeeds."""
    B, T, d_model, num_heads = 1, 6, 8, 2
    dec = DecoderBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    enc_out = torch.randn(B, T, d_model)
    mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
    out = dec(x, enc_out, mask)
    assert out.shape == (B, T, d_model), f"Causal mask broke shape: {out.shape}"
    print(f"[Decoder Test 4] causal mask forward: shape {out.shape} | expected: ({B}, {T}, {d_model})")
    print()


def test_decoder_cross_attention_diff_lengths():
    """Test 5: cross-attention with different target/source lengths."""
    B, T_tgt, T_src, d_model, num_heads = 2, 5, 9, 16, 4
    dec = DecoderBlock(d_model, num_heads)
    x = torch.randn(B, T_tgt, d_model)
    enc_out = torch.randn(B, T_src, d_model)
    out = dec(x, enc_out)
    assert out.shape == (B, T_tgt, d_model), f"Expected ({B}, {T_tgt}, {d_model}), got {out.shape}"
    print(f"[Decoder Test 5] cross-attention (tgt={T_tgt}, src={T_src}): {out.shape} | expected: ({B}, {T_tgt}, {d_model})")
    print()


def test_decoder_encoder_pipeline():
    """Test 6: end-to-end encoder → decoder pipeline."""
    B, T, d_model, num_heads, num_layers = 2, 8, 16, 4, 2
    enc = TransformerEncoder(d_model, num_heads, num_layers)
    dec = TransformerDecoder(d_model, num_heads, num_layers)
    src = torch.randn(B, T, d_model)
    tgt = torch.randn(B, T, d_model)
    enc_out = enc(src)
    out = dec(tgt, enc_out)
    assert out.shape == (B, T, d_model), f"Pipeline output mismatch: {out.shape}"
    loss = out.sum()
    loss.backward()
    print(f"[Decoder Test 6] encoder -> decoder pipeline: shape {out.shape}, gradients OK")
    print()


def test_decoder_deterministic():
    """Test 7: same input -> same output in eval mode."""
    B, T, d_model, num_heads = 1, 4, 8, 2
    dec = DecoderBlock(d_model, num_heads)
    dec.eval()
    x = torch.randn(B, T, d_model)
    enc_out = torch.randn(B, T, d_model)
    out1 = dec(x, enc_out)
    out2 = dec(x, enc_out)
    same = torch.allclose(out1, out2)
    assert same, "Decoder is not deterministic"
    print(f"[Decoder Test 7] deterministic: {same} | expected: True")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  Decoder Block tests")
    print("=" * 50)
    print()
    test_decoder_output_shape()
    test_decoder_gradient_flow()
    test_decoder_multi_layer_shape()
    test_decoder_causal_mask()
    test_decoder_cross_attention_diff_lengths()
    test_decoder_encoder_pipeline()
    test_decoder_deterministic()
