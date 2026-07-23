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
  - 输出形状正确 ✅
  - 注意力权重每行和为 1 ✅
  - 缩放因子生效（熵正常） ✅
  - Mask 正确遮断 ✅

✅ **2. Multi-Head Attention** — 已完成并提交

`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
`

- 实现位置：[main.py](main.py) — MultiHeadAttention(nn.Module)
- 测试验证：[test_multihead.py](test_multihead.py) — 5 项测试全部通过
  - 输出形状正确 ✅
  - 注意力权重每头每行和为 1 ✅
  - Causal mask 遮断正确 ✅
  - Cross-attention（Q/KV 不同序列长度） ✅
  - 梯度正常反向传播 ✅

**下一块积木：⬇️ Positional Encoding**

---

## 学习路线图（待完成）

`
Scaled Dot-Product Attention    ← ✅ 已完成
    ↓
Multi-Head Attention            ← ✅ 已完成
    ↓
Positional Encoding             ← ⏳ 待开始
    ↓
Feed-Forward Network
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

1. scaled_dot_product_attention() 中 	orch.sqrt(torch.tensor(q.shape[-1])) 可优化为 math.sqrt(q.shape[-1]) — 纯风格问题，不影响正确性
2. MultiHeadAttention.forward() 中 T_v 变量未使用，可清理

## 验证方法

`ash
pytest test_attention.py test_multihead.py
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
