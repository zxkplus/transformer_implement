"""
测试用例 —— 让函数自己告诉我们哪里不对劲
"""
import torch
import torch.nn.functional as F
import math

# 引入你的实现
from main import scaled_dot_product_attention

def test_01_output_shape():
    """测试 1: 输出形状对不对"""
    B, T, D = 2, 4, 8
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    print(f"[Test 1] 输入形状: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"         输出形状: out={out.shape}, attn={attn.shape}")
    print(f"         期望输出形状: out={q.shape}, attn=({B}, {T}, {T})")
    print()

def test_02_weights_sum_to_one():
    """测试 2: 注意力权重每一行加起来是不是 1"""
    B, T, D = 1, 3, 4
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    row_sums = attn.sum(dim=-1)  # 每行加起来
    print(f"[Test 2] 注意力权重每行之和: {row_sums}")
    print(f"         期望: 全 1.0（或接近）")
    print()

def test_03_scale_effect():
    """测试 3: 尺度缩放能感受到吗？
    如果 d_k 很大，不除 sqrt(d_k) 的话 softmax 会变 one-hot
    """
    B, T, D = 1, 5, 64
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    entropy = -(attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1).mean()
    print(f"[Test 3] 注意力的平均熵: {entropy.item():.4f}")
    print(f"         如果有缩放，熵应该比较高（分布均匀）")
    print(f"         如果没缩放或缩放错了，熵接近 0（one-hot）")
    print()

def test_04_mask_works():
    """测试 4: mask 能不能正确遮住不该看的位置"""
    B, T, D = 1, 4, 4
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
    out, attn = scaled_dot_product_attention(q, k, v, mask)
    print(f"[Test 4] 下三角 mask 后的注意力矩阵:")
    print(attn)
    print(f"         mask 右上角应该全是 0")
    upper_tri = attn[0].triu(diagonal=1)
    print(f"         右上角值的和: {upper_tri.sum().item()}（期望 0.0）")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("  跑你的 attention 函数测试啦！")
    print("=" * 50)
    test_01_output_shape()
    test_02_weights_sum_to_one()
    test_03_scale_effect()
    test_04_mask_works()
