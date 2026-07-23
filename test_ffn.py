"""Test cases for Position-wise Feed-Forward Network."""
import torch
from main import PositionwiseFeedForward


def test_ffn_output_shape():
    """Test FFN preserves input shape."""
    B, T, d_model = 2, 10, 16
    ffn = PositionwiseFeedForward(d_model)
    x = torch.randn(B, T, d_model)
    out = ffn(x)
    print(f"[FFN Test 1] input: ({B}, {T}, {d_model}), output: {out.shape}")
    print(f"       expected: ({B}, {T}, {d_model})")
    print()


def test_ffn_position_wise():
    """Test FFN is truly position-wise: same weights, independent per token."""
    B, T, d_model = 1, 5, 8
    ffn = PositionwiseFeedForward(d_model)
    x = torch.randn(B, T, d_model)
    out = ffn(x)
    # Replace position 0 with zeros, keeping others. FFN should only change position 0.
    x2 = x.clone()
    x2[:, 0, :] = 0.0
    out2 = ffn(x2)
    diff_other = (out[:, 1:, :] - out2[:, 1:, :]).abs().max().item()
    print(f"[FFN Test 2] max diff at positions 1..T-1 after changing pos 0: {diff_other:.2e}")
    print(f"       expected: 0.0 (position-wise means independent per token)")
    print()


def test_ffn_hidden_dim():
    """Test FFN inner dimension is 4 * d_model."""
    d_model = 16
    ffn = PositionwiseFeedForward(d_model)
    w1_out = ffn.line1.weight.shape[0]
    w2_in = ffn.line2.weight.shape[1]
    print(f"[FFN Test 3] line1 output dim: {w1_out}, line2 input dim: {w2_in}")
    print(f"       expected: {4 * d_model} (= 4 * {d_model})")
    print()


def test_ffn_gradient_flow():
    """Test gradients flow through both linear layers."""
    B, T, d_model = 1, 4, 8
    ffn = PositionwiseFeedForward(d_model)
    x = torch.randn(B, T, d_model)
    out = ffn(x)
    loss = out.sum()
    loss.backward()
    no_grad = [name for name, p in ffn.named_parameters() if p.grad is None]
    print(f"[FFN Test 4] params without gradient: {no_grad if no_grad else 'none (all good!)'}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  PositionwiseFeedForward tests")
    print("=" * 50)
    print()
    test_ffn_output_shape()
    test_ffn_position_wise()
    test_ffn_hidden_dim()
    test_ffn_gradient_flow()
