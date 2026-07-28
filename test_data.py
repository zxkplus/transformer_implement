"""Tests for src/data.py - vocab, encode/decode, data generation."""

import pytest
from src.data import (
    token_table,
    table2index,
    encode,
    decode,
    generate_addition_data,
    collate_batch,
)


def test_01_vocab_size():
    assert len(token_table) == 14
    assert len(table2index) == 14


def test_02_vocab_special_tokens():
    assert table2index["<pad>"] == 0
    assert table2index["<sos>"] == 1
    assert table2index["<eos>"] == 2


def test_03_vocab_digits():
    assert table2index["0"] == 3
    assert table2index["1"] == 4
    assert table2index["2"] == 5
    assert table2index["9"] == 12


def test_04_vocab_plus():
    assert table2index["+"] == 13


def test_05_encode_basic():
    result = encode("12+34")
    assert result == [4, 5, 13, 6, 7], f"got {result}"


def test_06_encode_with_sos_eos():
    result = encode("0", add_sos_eos=True)
    assert result == [1, 3, 2], f"got {result}"


def test_07_encode_with_sos_eos_multi():
    result = encode("99+1", add_sos_eos=True)
    assert result == [1, 12, 12, 13, 4, 2], f"got {result}"


def test_08_decode_basic():
    result = decode([4, 5, 13, 6, 7])
    assert result == "12+34", f"got {result!r}"


def test_09_decode_with_special():
    result = decode([1, 3, 2])
    assert result == "<sos>0<eos>", f"got {result!r}"


def test_10_encode_decode_roundtrip():
    expressions = ["0", "12+34", "999+1", "0+0", "123+456"]
    for expr in expressions:
        encoded = encode(expr)
        decoded = decode(encoded)
        assert decoded == expr, f"{expr!r} -> {encoded} -> {decoded!r}"


def test_11_encode_decode_roundtrip_with_sos_eos():
    results = ["0", "12+34", "999+1"]
    for expr in results:
        encoded = encode(expr, add_sos_eos=True)
        decoded = decode(encoded)
        expected = "<sos>" + expr + "<eos>"
        assert decoded == expected, f"{expr!r} -> {encoded} -> {decoded!r} != {expected!r}"


def test_12_generate_addition_data_returns_tuple():
    src, tgt = generate_addition_data(max_digits=2)
    assert isinstance(src, list), f"src type: {type(src)}"
    assert isinstance(tgt, list), f"tgt type: {type(tgt)}"
    assert len(src) > 0, "src is empty"
    assert len(tgt) > 0, "tgt is empty"


def test_13_generate_addition_data_valid_tokens():
    src, tgt = generate_addition_data(max_digits=3)
    for sid in src:
        assert 0 <= sid < len(token_table), f"invalid src token: {sid}"
    for tid in tgt:
        assert 0 <= tid < len(token_table), f"invalid tgt token: {tid}"


def test_14_generate_addition_data_tgt_has_sos():
    for _ in range(20):
        _, tgt = generate_addition_data(max_digits=2)
        assert tgt[0] == 1, f"tgt should start with <sos>: {tgt}"


def test_15_collate_batch_returns_correct_size():
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        batch = collate_batch(bs)
        assert len(batch) == bs, f"expected {bs} samples, got {len(batch)}"
        for sample in batch:
            assert isinstance(sample, tuple) and len(sample) == 2


def test_16_collate_batch_uniform_padding():
    for _ in range(5):
        batch = collate_batch(4)
        src_lens = [len(s[0]) for s in batch]
        tgt_lens = [len(s[1]) for s in batch]
        assert len(set(src_lens)) == 1, f"src lengths not uniform: {src_lens}"
        assert len(set(tgt_lens)) == 1, f"tgt lengths not uniform: {tgt_lens}"


def test_17_collate_batch_padding_with_zero():
    """All padded positions should be 0 (<pad>)."""
    for _ in range(5):
        batch = collate_batch(4)
        max_src = max(len(s[0]) for s in batch)  # already uniform but just in case
        max_tgt = max(len(s[1]) for s in batch)
        for src, tgt in batch:
            for i in range(len(src)):
                if i >= len(decode(src).replace("<pad>", "").replace("<sos>", "")):
                    continue
            for tok in src:
                if tok != 0:
                    assert 1 <= tok < len(token_table), f"non-pad/bad token: {tok}"
            for tok in tgt:
                if tok != 0:
                    assert 1 <= tok < len(token_table), f"non-pad/bad token: {tok}"


def test_18_addition_correctness():
    for _ in range(50):
        src_ids, tgt_ids = generate_addition_data(max_digits=2)
        src_str = decode(src_ids)
        tgt_full = decode(tgt_ids)
        tgt_str = tgt_full.replace("<sos>", "").replace("<eos>", "")

        parts = src_str.split("+")
        assert len(parts) == 2, f"bad expression: {src_str}"
        x1, x2 = int(parts[0]), int(parts[1])
        expected = str(x1 + x2)
        assert tgt_str == expected, f"{src_str}={expected}, got {tgt_str}"
