"""Test cases for Positional Encoding."""
import torch
from src import PositionalEncoding


def test_pe_output_shape():
    """Test PE is added correctly and output shape preserved."""
    B, T, d_model = 2, 10, 16
    pe = PositionalEncoding(d_model)
    x = torch.zeros(B, T, d_model)
    out = pe(x)
    print(f"[PE Test 1] input: ({B}, {T}, {d_model}), output: {out.shape}")
    print(f"       expected: ({B}, {T}, {d_model})")
    print()


def test_pe_values_are_bounded():
    """Test PE values are in [-1, 1] range (sin/cos)."""
    d_model, T = 32, 100
    pe = PositionalEncoding(d_model)
    x = torch.zeros(1, T, d_model)
    out = pe(x)
    pe_only = out - x  # isolate the encoding
    in_range = (pe_only >= -1.01).all() and (pe_only <= 1.01).all()
    print(f"[PE Test 2] all PE values in [-1, 1]: {in_range}")
    print()


def test_pe_deterministic():
    """Test PE gives same result on repeated calls."""
    d_model, T = 16, 8
    pe = PositionalEncoding(d_model)
    x = torch.randn(2, T, d_model)
    out1 = pe(x)
    out2 = pe(x)
    same = torch.allclose(out1, out2)
    print(f"[PE Test 3] deterministic: {same}")
    print()


def test_pe_not_trainable():
    """Test PE buffer is not a parameter (no grad)."""
    d_model = 16
    pe = PositionalEncoding(d_model)
    param_count = sum(1 for _ in pe.parameters())
    print(f"[PE Test 4] trainable parameters: {param_count} | expected: 0")
    print()


def test_pe_in_state_dict():
    """Test PE buffer is saved in state_dict."""
    d_model = 16
    pe = PositionalEncoding(d_model)
    keys = list(pe.state_dict().keys())
    has_pe = 'pe' in keys
    print(f"[PE Test 5] state_dict keys: {keys}")
    print(f"       contains 'pe': {has_pe}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  PositionalEncoding tests")
    print("=" * 50)
    print()
    test_pe_output_shape()
    test_pe_values_are_bounded()
    test_pe_deterministic()
    test_pe_not_trainable()
    test_pe_in_state_dict()
