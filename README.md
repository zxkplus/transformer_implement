# Transformer From Scratch 🏗️

从零手撕 Transformer——逐块构建、逐层理解。

这不是一份完整的代码库，而是一份**交互式学习记录**。每块积木由学员手动敲出，导师（Codex agent）提供提示、引导和测试验证。学员动手，导师动嘴。

---

## 项目结构

```
transformer_implement/
├── src/                        # 模块化源码包
│   ├── __init__.py             # 统一 re-export 所有组件
│   ├── attention.py            # Scaled Dot-Product Attention + MultiHeadAttention
│   ├── positional.py           # PositionalEncoding
│   ├── ffn.py                  # PositionwiseFeedForward
│   ├── encoder.py              # EncoderBlock + TransformerEncoder
│   ├── decoder.py              # DecoderBlock + TransformerDecoder
│   ├── transformer.py          # 端到端 Transformer（整合 PE + Encoder + Decoder）
│   └── data.py                 # 数据处理：词表、编码、padding、数据生成
├── main.py                     # 快速入口（from src import *）
├── test_attention.py           # Scaled Dot-Product Attention 测试（4 项）
├── test_multihead.py           # Multi-Head Attention 测试（5 项）
├── test_positional.py          # Positional Encoding 测试（5 项）
├── test_ffn.py                 # Feed-Forward Network 测试（4 项）
├── test_encoder.py             # Encoder Block + TransformerEncoder 测试（7 项）
├── test_decoder.py             # Decoder Block + TransformerDecoder 测试（7 项）
├── test_data.py                # 数据处理 pipeline 测试（18 项）
├── test_transformer.py         # 完整端到端 Transformer 测试（5 项）
├── .vscode/
│   └── settings.json           # VS Code 配置：conda 环境、pytest runner
├── AGENTS.md                   # Agent 行为指南（仅供 agent 读取）
└── README.md                   # ← 本文件，随进度同步维护
```

---

## 组件总览

| 组件 | 实现类 / 函数 | 测试文件 | 测试数 | 状态 |
|---|---|---|---|---|
| Scaled Dot-Product Attention | `scaled_dot_product_attention()` | `test_attention.py` | 4 | ✅ |
| Multi-Head Attention | `MultiHeadAttention` | `test_multihead.py` | 5 | ✅ |
| Positional Encoding | `PositionalEncoding` | `test_positional.py` | 5 | ✅ |
| Position-wise FFN | `PositionwiseFeedForward` | `test_ffn.py` | 4 | ✅ |
| Encoder Block + TransformerEncoder | `EncoderBlock` / `TransformerEncoder` | `test_encoder.py` | 7 | ✅ |
| Decoder Block + TransformerDecoder | `DecoderBlock` / `TransformerDecoder` | `test_decoder.py` | 7 | ✅ |
| 完整端到端 Transformer | `Transformer` | `test_transformer.py` | 5 | ✅ |
| 数据处理 | `encode` / `decode` / `generate_addition_data` / `collate_batch` | `test_data.py` | 18 | ✅ |

---

## 当前进度

### ✅ Python 基础实现（全部完成）

所有 8 个模块（7 个核心模块 + 数据处理 pipeline）已在 `src/` 包中实现，55 项测试全部通过。

| 阶段 | 完成内容 |
|---|---|
| 1. 注意力机制 | `scaled_dot_product_attention` + `MultiHeadAttention` |
| 2. 位置编码 | `PositionalEncoding`（sin/cos 固定编码） |
| 3. 前馈网络 | `PositionwiseFeedForward`（ReLU, d_ff = 4 × d_model） |
| 4. Encoder | `EncoderBlock` + `TransformerEncoder`（残差 + LayerNorm） |
| 5. Decoder | `DecoderBlock` + `TransformerDecoder`（自注意力 + 交叉注意力 + causal mask） |
| 6. 完整 Transformer | `Transformer`（独立 src/tgt embedding + 动态 mask 切片） |
| 7. 代码拆包 | `src/` 包拆分完成，`__init__.py` 统一导出 |

### ✅ 数据处理（已完成）

为数字加法任务提供完整数据 pipeline，核心函数位于 `src/data.py`：

| 函数 | 功能 | 说明 |
|---|---|---|
| `encode(expr, add_sos_eos=False)` | 字符串 → token ID 列表 | 可选添加 `<sos>` / `<eos>` |
| `decode(tokens)` | token ID 列表 → 字符串 | 双向无损 |
| | `generate_addition_data(max_digits=3)` | 生成单条加法样本 | 返回原始 (src_ids, tgt_ids)，不做 padding |
| `collate_batch(batch_size)` | 批量采样 + batch 级 padding | src / tgt 分别 pad 到各自最大长度 |

---

## 学习路线图

```
Scaled Dot-Product Attention    ← ✅
    ↓
Multi-Head Attention            ← ✅
    ↓
Positional Encoding             ← ✅
    ↓
Feed-Forward Network            ← ✅
    ↓
Transformer Encoder             ← ✅
    ↓
Transformer Decoder             ← ✅
    ↓
完整端到端 Transformer          ← ✅
    ↓
代码拆包 → src/ 包              ← ✅
    ↓
数据处理 pipeline               ← ✅
    ↓
训练循环                        ← ⏳
    ↓
──────────────────── Python 基础实现阶段完结 ────────────────────
    ↓
推理 demo                       ← 📋 规划中
    ↓
注意力可视化                    ← 📋 规划中
    ↓
CUDA 版：从 Python 到 C++      ← 📋 规划中
```

---

## 开发约定

### 环境

- **Conda 环境**：`torch3.10`（Python 3.10.19, PyTorch 2.10.0+cu130）
- **激活方式**：`conda activate torch3.10`
- **conda 路径**：`/home/industai/anaconda3`
- 项目首次使用前需安装 pytest：
  ```bash
  pip install pytest
  ```

### 测试

- **框架**：pytest（VS Code 已配置 pytest runner）
- **命名**：`test_<模块>.py` 对应测试文件，`test_<NN>_<描述>` 对应测试函数
- **运行所有测试**：
  ```bash
  pytest
  ```
- **运行单个文件**：
  ```bash
  pytest test_transformer.py -v
  ```
- **新增组件时**：必须补充对应的测试文件，并确保全部测试通过后再提交

### 代码风格

- 缩进：4 空格，无 tab
- 命名：函数/变量 `snake_case`，`nn.Module` 子类 `PascalCase`
- 导入顺序：标准库 → 第三方 (`torch`, `torch.nn`) → 本地
- Type hints：鼓励但不强制

### 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

| 类型 | 用途 | 示例 |
|---|---|---|
| `feat:` | 新组件 | `feat: add multi-head attention` |
| `test:` | 测试增补 | `test: add encoder block tests` |
| `docs:` | 文档更新 | `docs: sync README with current progress` |
| `refactor:` | 结构调整 | `refactor: move modules into src/ package` |

保持每个 commit 聚焦于一个组件。

### README 维护规则

**本文件由 agent 负责同步维护。** 当工程进度变化时，agent 必须同步更新：

1. **组件总览表** — 新增组件须添加行，更新状态
2. **当前进度** — 新完成的块须展开写伪代码 + 测试结果
3. **学习路线图** — 标记完成/待开始/规划中
4. **下一步待办** — 反映当前最紧急的下一个任务
5. **测试命令** — 新测试文件须加入验证方法

---

## 验证方法

```bash
conda activate torch3.10
pytest
```

运行全部 55 项测试，所有测试应通过，无报错。数据层测试（`test_data.py`）已完成 18 项。

当前测试分布：

| 测试文件 | 测试数 |
|---|---|
| `test_attention.py` | 4 |
| `test_multihead.py` | 5 |
| `test_positional.py` | 5 |
| `test_ffn.py` | 4 |
| `test_encoder.py` | 7 |
| `test_decoder.py` | 7 |
| `test_transformer.py` | 5 |
| `test_data.py` | 18 |
| **合计** | **55** |

---

## 下一步方向

Python 全部 8 个组件已实现，后续进入**应用与深入**阶段：

### 第 1 步：训练循环 ⏳（进行中）

用本仓库的 Transformer 在**数字加法**任务上跑起来：

- **词表设计** ✅
- **数据 pipeline** ✅ `src/data.py` encode → decode → pad → generate → collate 完整链路
- **训练循环** ⏳ 进行中（骨架已写，待修复 3 个已知 bug）

  | 问题 | 说明 | 修复方向 |
  |---|---|---|
  | 缺 `import torch` | `torch.tensor()` 需要库引用 | 顶部加 `import torch` |
  | `vocab_size=13` | 词表实际 14 个 token，`+` 索引 13 | 改为 `len(token_table)` |
  | `src_mask` 不识别 | `Transformer.forward()` 无此参数 | 改为 `model(src, tgt)` |

- **验证** ✅ `test_data.py` 已编写 18 项测试，全部通过

### 第 2 步：推理 demo + 注意力可视化 📋

- 把 encoder 自注意力和 decoder 交叉注意力的权重矩阵画出来
- 理解不同 head 学到了什么模式

### 第 3 步：CUDA C++ 迁移 📋

- 将 Python 核心计算（attention、FFN、layer norm）转写为 CUDA kernel
- 深入理解 GPU 编程和 Transformer 底层计算

---

*本 README 随工程进度由 agent 同步维护。*
