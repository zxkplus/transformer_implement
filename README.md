# Transformer From Scratch 🏗️

从零手撕 Transformer——逐块构建、逐层理解。

这不是一份完整的代码库，而是一份**交互式学习记录**。每块积木由学员手动敲出，导师（Codex agent）提供提示、引导和测试验证。学员动手，导师动嘴。

---

## 项目结构

```
transformer_implement/
├── main.py                 # 全部模块实现（单文件，暂未拆包）
├── test_attention.py       # Scaled Dot-Product Attention 测试（4 项）
├── test_multihead.py       # Multi-Head Attention 测试（5 项）
├── test_positional.py      # Positional Encoding 测试（5 项）
├── test_ffn.py             # Feed-Forward Network 测试（4 项）
├── test_encoder.py         # Encoder Block + TransformerEncoder 测试（7 项）
├── test_decoder.py         # Decoder Block + TransformerDecoder 测试（7 项）
├── test_transformer.py     # 完整端到端 Transformer 测试（5 项）
├── .vscode/
│   └── settings.json       # VS Code 配置：conda 环境、pytest runner
├── AGENTS.md               # Agent 行为指南（仅供 agent 读取）
└── README.md               # ← 本文件，随进度同步维护
```

**演进路线**：随着组件增多，`main.py` 将拆分为 `src/` 包，每个模块一个文件（`src/attention.py`、`src/encoder.py` ...），由 `src/__init__.py` 统一导出。

---

## 组件总览

| 组件 | 实现类 / 函数 | 测试文件 | 测试数 | 状态 |
|---|---|---|---|---|
| Scaled Dot-Product Attention | `scaled_dot_product_attention()` | `test_attention.py` | 4 | ✅ |
| Multi-Head Attention | `MultiHeadAttention` | `test_multihead.py` | 5 | ✅ |
| Positional Encoding | `PositionalEncoding` | `test_positional.py` | 5 | ✅ |
| Position-wise FFN | `PositionwiseFeedForward` | `test_ffn.py` | 4 | ✅ |
| Encoder Block | `EncoderBlock` | `test_encoder.py` | 4 | ✅ |
| Transformer Encoder | `TransformerEncoder` | `test_encoder.py` | 3 | ✅ |
| Decoder Block | `DecoderBlock` | `test_decoder.py` | 4 | ✅ |
| Transformer Decoder | `TransformerDecoder` | `test_decoder.py` | 3 | ✅ |
| 完整端到端 Transformer | `Transformer` | `test_transformer.py` | 5 | ✅ |

---

## 当前进度

### ✅ 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

- 实现：[main.py](main.py) — `scaled_dot_product_attention(q, k, v, mask=None)`
- 测试：[test_attention.py](test_attention.py) — 4 项
  - 输出形状正确 ✅
  - 注意力权重每行和为 1 ✅
  - 缩放因子生效（熵不 collapse） ✅
  - Mask 正确遮住未来位置 ✅

### ✅ 2. Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i = Attention(Q W_Q_i, K W_K_i, V W_V_i)
```

- 实现：[main.py](main.py) — `MultiHeadAttention(d_model, num_heads)`
- 测试：[test_multihead.py](test_multihead.py) — 5 项
  - 输出形状 `(B, T, d_model)` ✅
  - 每头注意力权重行和为 1 ✅
  - Causal mask 生效 ✅
  - 交叉注意力（Q/K/V 不同源） ✅
  - 梯度流通 ✅

### ✅ 3. Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- 实现：[main.py](main.py) — `PositionalEncoding(d_model, max_len=50000)`
- 测试：[test_positional.py](test_positional.py) — 5 项
  - 输出形状不变 ✅
  - 值域在 [-1, 1] ✅
  - 确定性（相同输入相同输出） ✅
  - 无可训练参数（register_buffer） ✅
  - state_dict 中包含 `pe` ✅

### ✅ 4. Position-wise Feed-Forward Network

```
FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2,  d_ff = 4 * d_model
```

- 实现：[main.py](main.py) — `PositionwiseFeedForward(d_model)`
- 测试：[test_ffn.py](test_ffn.py) — 4 项
  - 输出形状不变 ✅
  - Position-wise 独立（改 token 0 不影响其他） ✅
  - 隐藏维度 = 4 * d_model ✅
  - 梯度流通 ✅

### ✅ 5. Encoder Block

```
x = LayerNorm(x + MultiHead(x, x, x))
x = LayerNorm(x + FFN(x))
```

- 实现：[main.py](main.py) — `EncoderBlock(d_model, num_heads)` + `TransformerEncoder(d_model, num_heads, num_layers)`
- 测试：[test_encoder.py](test_encoder.py) — 7 项
  - EncoderBlock 输出形状正确 ✅
  - 梯度流通 ✅
  - TransformerEncoder 叠加 N 层形状不变 ✅
  - 自注意力跨 token 传播信息 ✅
  - 残差连接活跃（输出 ≠ 输入） ✅
  - 确定性 ✅
  - 多层 ≠ 单层（叠加有效） ✅

### ✅ 6. Decoder Block

```
x = LayerNorm(x + MaskedSelfAttn(x, x, x))
x = LayerNorm(x + CrossAttn(x, enc_output, enc_output))
x = LayerNorm(x + FFN(x))
```

- 实现：[main.py](main.py) — `DecoderBlock(d_model, num_heads)` + `TransformerDecoder(d_model, num_heads, num_layers)`
- 测试：[test_decoder.py](test_decoder.py) — 7 项
  - DecoderBlock 输出形状正确 ✅
  - 梯度流通 ✅
  - TransformerDecoder 叠加 N 层形状不变 ✅
  - Causal mask 前向不崩 ✅
  - 交叉注意力支持不同长度（T_q ≠ T_kv） ✅
  - Encoder → Decoder 端到端联调 ✅
  - 确定性 ✅

### ✅ 7. 完整端到端 Transformer

```
Transformer(src, tgt) = OutputProj(Decoder(tgt+PE, Encoder(src+PE)))

src:  token IDs → Embedding → +PE → TransformerEncoder → encoder_output
tgt:  token IDs → Embedding → +PE → TransformerDecoder(encoder_output) → Linear → logits
```

- 实现：[main.py](main.py) — `Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, max_len=5000)`
- 测试：[test_transformer.py](test_transformer.py) — 5 项
  - 输出形状 `(B, T_tgt, tgt_vocab_size)` ✅
  - 梯度流经全部参数 ✅
  - 源序列/目标序列不同长度 ✅
  - 确定性（eval 模式） ✅
  - Causal mask 按实际序列长度动态切片 ✅

**关键设计点**：
- 源和目标使用独立的 `nn.Embedding` 层
- Causal mask 在 `forward` 中根据 `target.size(1)` 动态切片
- Decoder 先将目标 embedding 传入 self-attention，再将 encoder output 传入 cross-attention

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
CUDA 版：从 Python 到 C++       ← ⏳ 规划中
```

---

## 开发约定

### 环境

- **Conda 环境**：`pytorch_env_v1`（Python 3.12.4, PyTorch 2.6.0+cu124）
- **激活方式**：`conda activate pytorch_env_v1`（PowerShell 中也可直接调用 `D:\ProgramData\anaconda3\envs\pytorch_env_v1\python.exe`）
- **conda 路径**：`D:\ProgramData\anaconda3`

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

### 代码结构演进

当 `main.py` 组件增多到难以管理时，按以下步骤拆分：

1. 创建 `src/` 目录 + `src/__init__.py`
2. 将每个模块拆为独立文件（`src/attention.py`, `src/encoder.py` ...）
3. 在 `src/__init__.py` 中统一 re-export
4. 更新测试文件中的 import 路径
5. 运行 `pytest` 确认全部通过
6. 同步更新本 README 的项目结构图

---

## 验证方法

```bash
conda activate pytorch_env_v1
pytest
```

运行全部 37 项测试，所有测试应通过，无报错。

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
| **合计** | **37** |

---

## 下一步方向

所有基础组件已全部实现并测试完毕，Transformer 的 From-Scratch 实现阶段**基本完结**。后续方向：

1. **CUDA C++ 迁移**：将 Python 实现转写为 CUDA C++，深入理解底层计算（远期规划）
2. **代码拆包**：当 `main.py` 足够庞大时，拆为 `src/` 包，每个模块一个文件
3. **推理/训练示例**：用本仓库的 Transformer 跑一个小规模的机器翻译 demo

---

*本 README 随工程进度由 agent 同步维护。*
