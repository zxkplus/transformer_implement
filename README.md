# Transformer From Scratch 🏗️

从零手撕 Transformer——逐块构建、逐层理解。

## 教学方式

这不是一份完整的代码库，而是一份**交互式学习记录**。每块积木由学员手动敲出，导师（Codex agent）提供提示、引导和测试验证。学员动手，导师动嘴。

---

## 当前进度

✅ **1. Scaled Dot-Product Attention**
✅ **2. Multi-Head Attention**
✅ **3. Positional Encoding**
✅ **4. Position-wise Feed-Forward Network** — 已完成

`
FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2,  d_ff = 4 * d_model
`

- 实现位置：[main.py](main.py) — PositionwiseFeedForward(nn.Module)
- 测试验证：[test_ffn.py](test_ffn.py) — 4 项测试全部通过
  - 输出形状正确 ✅
  - Position-wise 独立（改 token 0 不影响其他） ✅
  - 隐藏维度 = 4 * d_model ✅
  - 梯度流通 ✅

**下一块积木：⬇️ Transformer Encoder Block**

---

## 学习路线图（待完成）

`
Scaled Dot-Product Attention    ← ✅ 已完成
    ↓
Multi-Head Attention            ← ✅ 已完成
    ↓
Positional Encoding             ← ✅ 已完成
    ↓
Feed-Forward Network            ← ✅ 已完成
    ↓
Transformer Encoder Block       ← ⏳ 待开始
    ↓
完整 Transformer
    ↓
CUDA 版：从 Python 到 C++
`

---

## 验证方法

`ash
pytest test_attention.py test_multihead.py test_positional.py test_ffn.py
`

所有测试应通过，无报错。
