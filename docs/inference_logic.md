# AlphaFold2 推理逻辑与函数调用顺序

## 目录

1. [推理流程概述](#推理流程概述)
2. [完整调用链](#完整调用链)
3. [详细调用顺序](#详细调用顺序)
4. [循环推理](#循环推理)
5. [性能优化建议](#性能优化建议)

---

## 推理流程概述

AlphaFold2 的推理过程可以分为以下几个主要阶段：

```
输入准备 → 嵌入层 → Evoformer → 结构模块 → 输出后处理
```

### 推理模式特点

- 关闭 Dropout
- 不计算 MLM 损失
- 可选循环推理（recycling）
- 可选返回置信度分数

---

## 完整调用链

### 1. 基本推理（不使用预训练嵌入）

```
main.py
│
└─> model = Alphafold2(dim=256, ...)
    │
    └─> model.eval()  # 设置为评估模式
        │
        └─> output = model(seq, msa, mask, msa_mask)
            │
            ├─> [输入嵌入阶段]
            │   ├─> self.token_emb(seq)                    # 序列嵌入
            │   ├─> self.token_emb(msa)                    # MSA 嵌入
            │   └─> self.to_pairwise_repr(x)               # 生成成对表示
            │
            ├─> [位置编码阶段]
            │   └─> self.pos_emb(relative_positions)       # 相对位置编码
            │
            ├─> [主干处理阶段]
            │   └─> self.net(x, m, mask, msa_mask)         # Evoformer
            │       │
            │       └─> for EvoformerBlock in layers:
            │           │
            │           ├─> MsaAttentionBlock(m, x)         # MSA 注意力
            │           │   ├─> AxialAttention(row)         # 行注意力
            │           │   └─> AxialAttention(col)         # 列注意力
            │           │
            │           ├─> FeedForward(m)                  # MSA FFN
            │           │
            │           ├─> PairwiseAttentionBlock(x, m)    # 成对注意力
            │           │   ├─> OuterMean(m)                # 外积均值
            │           │   ├─> TriangleMultiply(x) * 2     # 三角乘法
            │           │   └─> AxialAttention(x) * 2       # 三角注意力
            │           │
            │           └─> FeedForward(x)                  # 成对 FFN
            │
            ├─> [预测头阶段]
            │   ├─> self.to_distogram_logits(x)            # 距离图
            │   ├─> self.to_prob_theta(x)                  # θ 角
            │   ├─> self.to_prob_phi(x)                    # φ 角
            │   └─> self.to_prob_omega(x)                  # ω 角
            │
            └─> [结构模块阶段] (if predict_coords)
                │
                ├─> single_repr = self.msa_to_single_repr_dim(m[0])
                ├─> pairwise_repr = self.trunk_to_pairwise_repr_dim(x)
                │
                └─> for i in range(structure_module_depth):
                    │
                    ├─> rotations = quaternion_to_matrix(quaternions)
                    │
                    ├─> single_repr = IPABlock(single_repr, pairwise_repr, rotations, translations)
                    │   │
                    │   ├─> IPA attention (不变点注意力)
                    │   └─> FeedForward
                    │
                    ├─> quat_update, trans_update = self.to_quaternion_update(single_repr)
                    │
                    ├─> quaternions = quaternion_multiply(quaternions, quat_update)
                    │
                    └─> translations += rotate(trans_update, rotations)
                │
                ├─> local_points = self.to_points(single_repr)
                │
                └─> coords = rotate(local_points, rotations) + translations
```

---

### 2. 使用 ESM 嵌入的推理

```
main.py
│
├─> model = Alphafold2(dim=256, ...)
├─> wrapped_model = ESMEmbedWrapper(alphafold2=model)
│
└─> wrapped_model.eval()
    │
    └─> output = wrapped_model(seq, msa, ...)
        │
        ├─> [ESM 嵌入阶段]
        │   ├─> self.batch_converter.from_lists(seq_data)  # 转换为 ESM 格式
        │   ├─> self.model(tokens)                         # ESM-1b 前向
        │   └─> seq_embeds = results['representations'][33]
        │
        ├─> [MSA 嵌入阶段]
        │   └─> for each msa_row:
        │       ├─> msa_embeds[i] = ESM(msa_row)
        │       └─> project(msa_embeds[i])
        │
        └─> self.alphafold2(seq, msa, seq_embed=seq_embeds, msa_embed=msa_embeds)
            └─> ... (与基本推理相同)
```

---

### 3. 使用 MSA Transformer 嵌入的推理

```
main.py
│
├─> model = Alphafold2(dim=256, ...)
├─> wrapped_model = MSAEmbedWrapper(alphafold2=model)
│
└─> wrapped_model.eval()
    │
    └─> output = wrapped_model(seq, msa, ...)
        │
        ├─> [MSA Transformer 嵌入阶段]
        │   ├─> seq_and_msa = cat([seq, msa], dim=1)
        │   ├─> self.batch_converter(seq_and_msa)
        │   └─> embeds = self.model(tokens)
        │       │
        │       └─> MSA Transformer 前向:
        │           ├─> Token embedding
        │           ├─> Position embedding
        │           └─> for TransformerLayer:
        │               ├─> Row attention (行绑定)
        │               └─> Column attention
        │
        ├─> [分离嵌入]
        │   ├─> seq_embed = embeds[:, 0]
        │   └─> msa_embed = embeds[:, 1:]
        │
        └─> self.alphafold2(seq, msa, seq_embed=seq_embeds, msa_embed=msa_embeds)
            └─> ... (与基本推理相同)
```

---

## 详细调用顺序

### 阶段 1: 输入准备与嵌入

#### 函数调用序列

```python
# 1. 序列嵌入
x = self.token_emb(seq)  # alphafold2.py:1076
if exists(seq_embed):
    x += seq_embed       # alphafold2.py:1079

# 2. MSA 嵌入
m = self.token_emb(msa)  # alphafold2.py:1094
if exists(msa_embed):
    m = m + msa_embed    # alphafold2.py:1097
m = m + rearrange(x, 'b n d -> b () n d')  # alphafold2.py:1102

# 3. 生成成对表示
x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim=-1)  # alphafold2.py:1118
x = rearrange(x_left, 'b i d -> b i () d') + rearrange(x_right, 'b j d-> b () j d')  # alphafold2.py:1120

# 4. 相对位置编码
seq_rel_dist = positions[i] - positions[j]  # alphafold2.py:1128
rel_pos_emb = self.pos_emb(seq_rel_dist)    # alphafold2.py:1131
x = x + rel_pos_emb                          # alphafold2.py:1133
```

**数据流**:
```
seq (b, n)
  → token_emb → (b, n, dim)
  → + seq_embed → (b, n, dim)
  → to_pairwise_repr → (b, n, dim*2)
  → chunk → left(b, n, dim), right(b, n, dim)
  → outer_sum → (b, n, n, dim)
  → + pos_emb → (b, n, n, dim)
```

---

### 阶段 2: 模板处理（如果提供）

#### 函数调用序列

```python
if exists(templates_feats):
    # 1. 模板嵌入
    t = self.to_template_embed(templates_feats)  # alphafold2.py:1158

    # 2. 多层成对注意力
    for _ in range(self.templates_embed_layers):
        t = self.template_pairwise_embedder(t, mask=t_mask)  # alphafold2.py:1168
        # 内部调用:
        #   → OuterMean (如果有 MSA)
        #   → TriangleMultiply (outgoing)
        #   → TriangleMultiply (ingoing)
        #   → AxialAttention (row)
        #   → AxialAttention (col)

    # 3. 点注意力池化
    template_pooled = self.template_pointwise_attn(
        x_point,
        context=t_point,
        mask=x_mask_point,
        context_mask=t_mask_point
    )  # alphafold2.py:1183

    # 4. 添加到成对表示
    x = x + template_pooled  # alphafold2.py:1196

# 5. 模板角度特征
if exists(templates_angles):
    t_angle_feats = self.template_angle_mlp(templates_angles)  # alphafold2.py:1202
    m = torch.cat((m, t_angle_feats), dim=1)  # alphafold2.py:1203
```

**数据流**:
```
templates_feats (b, t, n, n, 32)
  → to_template_embed → (b, t, n, n, dim)
  → PairwiseAttentionBlock x4 → (b, t, n, n, dim)
  → attention pooling → (b, n, n, dim)
  → add to x → x (b, n, n, dim)
```

---

### 阶段 3: Evoformer 主干

#### 单个 EvoformerBlock 的调用顺序

```python
# alphafold2.py:736-748
def forward(inputs):
    x, m, mask, msa_mask = inputs

    # 1. MSA 注意力
    m = self.msa_attn(m, mask=msa_mask, pairwise_repr=x) + m
    # 内部调用:
    #   → LayerNorm(m)
    #   → AxialAttention(row, edges=x)  # 成对表示作为偏置
    #   → AxialAttention(col)

    # 2. MSA 前馈
    m = self.msa_ff(m) + m
    # 内部调用:
    #   → LayerNorm(m)
    #   → Linear(dim, dim*mult*2)
    #   → GEGLU()
    #   → Dropout()
    #   → Linear(dim*mult, dim)

    # 3. 成对注意力
    x = self.attn(x, mask=mask, msa_repr=m, msa_mask=msa_mask) + x
    # 内部调用:
    #   → OuterMean(m)  # MSA → 成对
    #   → TriangleMultiply(outgoing)
    #   → TriangleMultiply(ingoing)
    #   → AxialAttention(row, edges=x)
    #   → AxialAttention(col, edges=x)

    # 4. 成对前馈
    x = self.ff(x) + x

    return x, m, mask, msa_mask
```

#### 完整 Evoformer 调用

```python
# alphafold2.py:791-792
x, m = self.net(x, m, mask=x_mask, msa_mask=msa_mask)

# 内部 (alphafold2.py:790-793):
inp = (x, m, mask, msa_mask)
x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
# 使用梯度检查点顺序执行各层
```

**数据流**:
```
初始:
  x: (b, n, n, dim)  成对表示
  m: (b, msa, n, dim)  MSA 表示

每层 EvoformerBlock:
  m → MsaAttention(m, x as bias) → m'
  m' → FFN → m''
  m'' → OuterMean → 成对信息
  x → TriangleUpdate → x'
  x' → TriangleAttention → x''
  x'' → FFN → x'''

输出:
  x: (b, n, n, dim)  更新的成对表示
  m: (b, msa, n, dim)  更新的 MSA 表示
```

---

### 阶段 4: 辅助预测

#### 函数调用序列

```python
# 1. 对称化成对表示
trunk_embeds = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # alphafold2.py:1242

# 2. 距离图预测
distance_pred = self.to_distogram_logits(trunk_embeds)  # alphafold2.py:1243
# 内部:
#   → LayerNorm(trunk_embeds)
#   → Linear(dim, DISTOGRAM_BUCKETS=37)

# 3. 角度预测 (如果启用)
if self.predict_angles:
    theta_logits = self.to_prob_theta(x)      # alphafold2.py:1237
    phi_logits = self.to_prob_phi(x)          # alphafold2.py:1238
    omega_logits = self.to_prob_omega(trunk_embeds)  # alphafold2.py:1258
    # 每个内部:
    #   → Linear(dim, num_buckets)
```

**输出形状**:
```
distance_pred: (b, n, n, 37)   # 37 个距离桶
theta_logits:  (b, n, n, 25)   # 25 个 θ 角桶
phi_logits:    (b, n, n, 13)   # 13 个 φ 角桶
omega_logits:  (b, n, n, 25)   # 25 个 ω 角桶
```

---

### 阶段 5: 结构模块

#### 准备阶段

```python
# alphafold2.py:1267-1270
single_msa_repr_row = m[:, 0]  # 取 MSA 第一行
single_repr = self.msa_to_single_repr_dim(single_msa_repr_row)
pairwise_repr = self.trunk_to_pairwise_repr_dim(x)

# 转换为 float32
single_repr, pairwise_repr = map(lambda t: t.float(), (single_repr, pairwise_repr))
```

#### 迭代细化循环

```python
# alphafold2.py:1281-1319
# 初始化
quaternions = [1, 0, 0, 0]  # 单位四元数
translations = [0, 0, 0]    # 零向量

for i in range(structure_module_depth):  # 默认 4 次
    # 1. 四元数 → 旋转矩阵
    rotations = quaternion_to_matrix(quaternions)  # (b, n, 3, 3)

    if not is_last:
        rotations = rotations.detach()  # 节省内存

    # 2. 不变点注意力
    single_repr = self.ipa_block(
        single_repr,
        mask=mask,
        pairwise_repr=pairwise_repr,
        rotations=rotations,
        translations=translations
    )
    # 内部 IPA 调用:
    #   → prepare query points (在当前坐标系中)
    #   → prepare key/value points
    #   → compute distances (3D 空间中的距离)
    #   → attention over points
    #   → aggregate
    #   → transform back
    #   → FFN

    # 3. 预测更新
    quat_update, trans_update = self.to_quaternion_update(single_repr).chunk(2, dim=-1)
    quat_update = F.pad(quat_update, (1, 0), value=1.)  # 填充为 7D

    # 4. 应用更新
    quaternions = quaternion_multiply(quaternions, quat_update)
    translations = translations + einsum('b n c, b n c r -> b n r', trans_update, rotations)

# 5. 计算最终坐标
local_points = self.to_points(single_repr)  # (b, n, 3)
rotations = quaternion_to_matrix(quaternions)
coords = einsum('b n c, b n c d -> b n d', local_points, rotations) + translations
```

**数据流**:
```
迭代 0:
  quaternions: 单位四元数 (无旋转)
  translations: 零向量
  ↓
  IPA(single_repr, rotations, translations)
  ↓
  预测 Δquat, Δtrans
  ↓
  更新 quaternions, translations

迭代 1:
  quaternions: 上次的四元数 * Δquat
  translations: 上次的平移 + rotate(Δtrans)
  ↓
  IPA(...) → 预测 → 更新
  ↓
  ...

迭代 depth-1:
  最终 quaternions, translations
  ↓
  local_points (局部坐标)
  ↓
  rotate + translate
  ↓
  coords (全局坐标)
```

---

## 循环推理

### 单次推理

```python
model = Alphafold2(...)
output = model(seq, msa, mask=mask, msa_mask=msa_mask, predict_coords=True)
# 返回: coords (b, n, 3)
```

### 循环推理（3 次迭代）

```python
model = Alphafold2(...)
recyclables = None

for recycle_iter in range(3):
    # 返回可循环数据
    coords, ret = model(
        seq, msa,
        mask=mask,
        msa_mask=msa_mask,
        recyclables=recyclables,
        return_recyclables=True,
        return_aux_logits=True
    )

    # 提取可循环数据
    recyclables = ret.recyclables
    # recyclables 包含:
    #   - coords: 上次预测的坐标
    #   - single_msa_repr_row: MSA 第一行表示
    #   - pairwise_repr: 成对表示

# 最后一次迭代的输出
final_coords = coords
```

### 循环输入的使用

```python
# alphafold2.py:1137-1149
if exists(recyclables):
    # 1. 更新 MSA 第一行
    m[:, 0] = m[:, 0] + self.recycling_msa_norm(recyclables.single_msa_repr_row)

    # 2. 更新成对表示
    x = x + self.recycling_pairwise_norm(recyclables.pairwise_repr)

    # 3. 从坐标计算距离并嵌入
    distances = torch.cdist(recyclables.coords, recyclables.coords, p=2)
    boundaries = torch.linspace(2, 20, steps=recycling_distance_buckets)
    discretized_distances = torch.bucketize(distances, boundaries[:-1])
    distance_embed = self.recycling_distance_embed(discretized_distances)

    x = x + distance_embed
```

**效果**:
- 每次迭代使用上次的预测作为先验
- 逐步细化结构预测
- 通常 3 次循环可以得到很好的结果

---

## 性能优化建议

### 1. 批处理

```python
# 不推荐: 逐个处理
results = []
for seq_i, msa_i in zip(seqs, msas):
    output = model(seq_i, msa_i)
    results.append(output)

# 推荐: 批处理
batch_seq = torch.stack(seqs)
batch_msa = torch.stack(msas)
batch_output = model(batch_seq, batch_msa)
```

### 2. 混合精度

```python
from torch.cuda.amp import autocast

model = Alphafold2(...).cuda()

with autocast():
    output = model(seq, msa)  # 自动使用 fp16
# 结构模块自动使用 fp32（代码中已实现）
```

### 3. 梯度检查点

```python
# 已在代码中实现（alphafold2.py:792）
x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
```

### 4. 序列长度限制

```python
# 对于长序列，考虑分块或降采样 MSA
max_seq_len = 512
max_msa = 128

seq = seq[:, :max_seq_len]
msa = msa[:, :max_msa, :max_seq_len]
```

### 5. 预计算嵌入

```python
# 一次性计算所有序列的 ESM 嵌入
esm_wrapper = ESMEmbedWrapper(alphafold2=model)

# 批量嵌入
all_seq_embeds = []
for batch in dataloader:
    with torch.no_grad():
        embeds = get_esm_embedd(batch)
    all_seq_embeds.append(embeds)

# 之后直接使用
output = model(seq, msa, seq_embed=precomputed_embed)
```

---

## 内存优化

### 函数调用中的内存消耗

1. **Evoformer**: O(batch * depth * seq_len² * dim)
   - 最大消耗：成对表示 (seq_len²)

2. **结构模块**: O(batch * depth * seq_len * dim)
   - 相对较小

3. **注意力**: O(batch * heads * seq_len²)
   - 临时激活值

### 优化策略

```python
# 1. 使用梯度检查点（已实现）
checkpoint_sequential(...)

# 2. 降低批次大小
batch_size = 1  # 对于长序列

# 3. 减少 MSA 数量
max_msa = 64  # 而不是 128

# 4. 减少模型深度（如果可以）
model = Alphafold2(depth=4)  # 而不是 6

# 5. 使用更小的维度
model = Alphafold2(dim=128, heads=4)  # 而不是 dim=256, heads=8
```

---

## 总结

### 关键调用路径

```
输入 → 嵌入 → Evoformer → 预测 → 结构细化 → 输出
```

### 计算瓶颈

1. **Evoformer**: 成对注意力 O(n³)
2. **结构模块**: IPA O(n² * depth)
3. **预训练嵌入**: ESM/MSA Transformer

### 优化优先级

1. 减少序列长度
2. 减少 MSA 数量
3. 使用梯度检查点
4. 批处理
5. 混合精度

### 典型推理时间（单GPU）

- 序列长度 100: ~1-2 秒
- 序列长度 200: ~5-10 秒
- 序列长度 400: ~30-60 秒
- 序列长度 800: ~2-5 分钟

*(时间取决于硬件、MSA 大小、模型配置等)*

---

## 参考代码示例

### 完整推理示例

```python
import torch
from alphafold2_pytorch import Alphafold2

# 1. 创建模型
model = Alphafold2(
    dim = 256,
    depth = 6,
    heads = 8,
    dim_head = 64,
    predict_coords = True,
    structure_module_depth = 4
).cuda().eval()

# 2. 准备输入
seq = torch.randint(0, 21, (1, 128)).cuda()
msa = torch.randint(0, 21, (1, 5, 128)).cuda()
mask = torch.ones_like(seq).bool().cuda()
msa_mask = torch.ones_like(msa).bool().cuda()

# 3. 推理
with torch.no_grad():
    coords = model(
        seq,
        msa,
        mask = mask,
        msa_mask = msa_mask
    )

# 4. 输出
print(f"Predicted coordinates shape: {coords.shape}")  # (1, 128, 3)
```

### 循环推理示例

```python
recyclables = None

for i in range(3):
    with torch.no_grad():
        if i < 2:
            # 前两次：返回可循环数据
            coords, ret = model(
                seq, msa,
                mask = mask,
                msa_mask = msa_mask,
                recyclables = recyclables,
                return_recyclables = True,
                return_aux_logits = True
            )
            recyclables = ret.recyclables
        else:
            # 最后一次：只返回坐标
            coords = model(
                seq, msa,
                mask = mask,
                msa_mask = msa_mask,
                recyclables = recyclables
            )

print(f"Final coordinates: {coords.shape}")
```

---

这份文档提供了 AlphaFold2 推理过程的完整函数调用顺序和数据流。如需更多细节，请参考源代码和 `core_functions.md`。
