# Transformer From Scratch 🏗️

从零手撕 Transformer——逐块构建、逐层理解。

## 教学方式

这不是一份完整的代码库，而是一份**交互式学习记录**。每块积木由学员手动敲出，导师（Codex agent）提供提示、引导和测试验证。学员动手，导师动嘴。

---

## 当前进度

✅ **1. Scaled Dot-Product Attention** — 已完成并提交

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- 实现位置：[main.py](main.py) — `scaled_dot_product_attention()`
- 测试验证：[test_attention.py](test_attention.py) — 4 项测试全部通过
  - 输出形状正确 ✅
  - 注意力权重每行和为 1 ✅
  - 缩放因子生效（熵正常） ✅
  - Mask 正确遮断 ✅

**下一块积木：⬇️ Multi-Head Attention**

---

## 学习路线图（待完成）

```
Scaled Dot-Product Attention    ← ✅ 已完成
    ↓
Multi-Head Attention            ← ⏳ 待开始
    ↓
Positional Encoding
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
```

---

## 遗留问题 / 接力点

1. **Multi-Head Attention 实现**应当从 `MultiHeadAttention(nn.Module)` 类开始
2. 关于"多头不创建 h 个独立线性层，而是先映射再 reshape"的做法，学员还需理解并回答这个问题后再动手写
3. `scaled_dot_product_attention()` 中 `torch.sqrt(torch.tensor(q.shape[-1]))` 可优化为 `math.sqrt(q.shape[-1])` — 纯风格问题，不影响正确性

## 验证方法

```bash
python test_attention.py
```

所有测试应通过，无报错。

## 提交记录

- `3393d90` — `feat: add scaled dot-product attention`
  - 新增 `.gitignore`（Python 标准忽略规则）
  - 新增 `main.py`（`scaled_dot_product_attention` 函数）
  - 新增 `test_attention.py`（4 项测试用例）
