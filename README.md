# Transformer From Scratch 🏗️

从零手撕 Transformer——逐块构建、逐层理解。

## 教学方式

这不是一份完整的代码库，而是一份**交互式学习记录**。每块积木由学员手动敲出，导师（Codex agent）提供提示、引导和测试验证。学员动手，导师动嘴。

---

## 当前进度

✅ **1. Scaled Dot-Product Attention** — 已完成并提交

`
Attention(Q, K, V) = softmax(QK^T / √d_k) V
`

- 实现位置：[main.py](main.py) — scaled_dot_product_attention()
- 测试验证：[test_attention.py](test_attention.py) — 4 项测试全部通过

✅ **2. Multi-Head Attention** — 已完成并提交

`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
`

- 实现位置：[main.py](main.py) — MultiHeadAttention(nn.Module)
- 测试验证：[test_multihead.py](test_multihead.py) — 5 项测试全部通过

✅ **3. Positional Encoding** — 已完成并提交

`
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
`

- 实现位置：[main.py](main.py) — PositionalEncoding(nn.Module)
- 测试验证：[test_positional.py](test_positional.py) — 5 项测试全部通过
- 向量化广播实现，与论文公式逐元素误差 < 6e-8

**下一块积木：⬇️ Feed-Forward Network**

---

## 学习路线图（待完成）

`
Scaled Dot-Product Attention    ← ✅ 已完成
    ↓
Multi-Head Attention            ← ✅ 已完成
    ↓
Positional Encoding             ← ✅ 已完成
    ↓
Feed-Forward Network            ← ⏳ 待开始
    ↓
Transformer Block (Encoder)
    ↓
Layer Normalization & Residual
    ↓
完整 Transformer
    ↓
CUDA 版：从 Python 到 C++
`

---

## 遗留问题

1. MultiHeadAttention.forward() 中 T_v 变量未使用，可清理
2. MultiHeadAttention 中各投影层命名（Qline 等）可统一为 W_q / W_k / W_v / W_o 风格

## 验证方法

`ash
pytest test_attention.py test_multihead.py test_positional.py
`

所有测试应通过，无报错。

## 提交记录

- 最新 — eat: add multi-head attention
  - 新增 MultiHeadAttention(nn.Module) 类
  - 新增 	est_multihead.py（5 项测试用例）
- 3393d90 — eat: add scaled dot-product attention
  - 新增 .gitignore（Python 标准忽略规则）
  - 新增 main.py（scaled_dot_product_attention 函数）
  - 新增 	est_attention.py（4 项测试用例）
