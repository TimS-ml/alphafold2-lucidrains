# AlphaFold2 核心函数逻辑文档

## 目录

1. [概述](#概述)
2. [核心数据结构](#核心数据结构)
3. [主要模块](#主要模块)
4. [关键函数](#关键函数)

---

## 概述

本文档详细介绍 AlphaFold2 PyTorch 实现中的核心函数和模块的工作逻辑。

---

## 核心数据结构

### 1. Recyclables (可循环数据)

**位置**: `alphafold2_pytorch/alphafold2.py:35-49`

```python
@dataclass
class Recyclables:
    coords: torch.Tensor              # 3D 坐标 (batch, seq_len, 3)
    single_msa_repr_row: torch.Tensor # MSA 单行表示 (batch, seq_len, dim)
    pairwise_repr: torch.Tensor       # 成对表示 (batch, seq_len, seq_len, dim)
```

**功能**:
- 存储 AlphaFold2 循环推理过程中的中间结果
- 用于迭代细化：将上一次预测的结果作为下一次的输入
- 包含三个关键信息：
  - 预测的 3D 坐标
  - MSA 的单序列表示
  - 残基对之间的关系表示

**使用场景**:
在多次循环推理中，每次迭代使用上一次的输出来改进预测，逐步提高结构预测的准确性。

---

### 2. ReturnValues (返回值容器)

**位置**: `alphafold2_pytorch/alphafold2.py:51-69`

```python
@dataclass
class ReturnValues:
    distance: torch.Tensor = None     # 距离图 (batch, seq_len, seq_len, num_bins)
    theta: torch.Tensor = None        # θ 角 (batch, seq_len, seq_len, num_bins)
    phi: torch.Tensor = None          # φ 角 (batch, seq_len, seq_len, num_bins)
    omega: torch.Tensor = None        # ω 角 (batch, seq_len, seq_len, num_bins)
    msa_mlm_loss: torch.Tensor = None # MLM 损失
    recyclables: Recyclables = None   # 可循环数据
```

**功能**:
统一管理模型的各种输出，包括：
- 距离预测（distogram）
- 二面角预测（torsion angles）
- 自监督学习损失
- 可循环的中间结果

---

## 主要模块

### 1. Attention (多头注意力)

**位置**: `alphafold2_pytorch/alphafold2.py:175-302`

**核心逻辑**:
```
输入: x (查询)
     context (键值，默认同 x)
     attn_bias (注意力偏置)
     mask (掩码)
     tie_dim (MSA 全局注意力维度)

步骤:
1. 计算 Q, K, V：
   q = self.to_q(x)
   k, v = self.to_kv(context).chunk(2)

2. 缩放 Q：
   q = q * self.scale  # scale = dim_head ** -0.5

3. 计算注意力分数：
   if tie_dim:  # MSA 列全局注意力
       q = q.mean(dim=1)  # 对 MSA 行求平均
   dots = einsum('...id, ...jd -> ...ij', q, k)

4. 添加注意力偏置（成对表示信息）：
   dots = dots + attn_bias

5. 应用掩码和 softmax：
   dots = dots.masked_fill(~mask, -inf)
   attn = softmax(dots)

6. 聚合值：
   out = einsum('...ij, ...jd -> ...id', attn, v)

7. 门控机制：
   gates = sigmoid(self.gating(x))
   out = out * gates

8. 输出投影：
   return self.to_out(out)
```

**特点**:
- 支持注意力偏置注入（用于成对表示）
- 支持 MSA 全局注意力（tie_dim）
- 门控机制控制信息流

---

### 2. TriangleMultiplicativeModule (三角乘法模块)

**位置**: `alphafold2_pytorch/alphafold2.py:400-495`

**核心逻辑**:
```
输入: x (成对表示，batch, i, j, dim)
     mask (成对掩码)

步骤:
1. 投影到左右表示：
   left = self.left_proj(x)   # (b, i, j, d)
   right = self.right_proj(x)  # (b, i, j, d)

2. 应用门控：
   left = left * sigmoid(self.left_gate(x))
   right = right * sigmoid(self.right_gate(x))

3. 三角更新（通过中间节点 k）：
   if mix == 'outgoing':
       out = einsum('ikd, jkd -> ijd', left, right)  # i->k, j->k 推断 i-j
   elif mix == 'ingoing':
       out = einsum('kjd, kid -> ijd', left, right)  # k->i, k->j 推断 i-j

4. 输出门控：
   out = out * sigmoid(self.out_gate(x))

5. 返回：
   return self.to_out(out)
```

**原理**:
利用三角不等式传播距离信息：
- 如果知道 i-k 和 j-k 的关系，可以推断 i-j 的关系
- outgoing: 从两个节点到共同节点 k 的出边
- ingoing: 从共同节点 k 到两个节点的入边

---

### 3. EvoformerBlock (Evoformer 块)

**位置**: `alphafold2_pytorch/alphafold2.py:682-749`

**核心逻辑**:
```
输入: x (成对表示)
     m (MSA 表示)
     mask (成对掩码)
     msa_mask (MSA 掩码)

步骤:
1. MSA 更新：
   m = MsaAttentionBlock(m, pairwise_repr=x) + m
   m = FeedForward(m) + m

   说明：
   - 行注意力：在同一序列的不同位置间传播信息
   - 列注意力：在不同序列的相同位置间传播信息
   - 成对表示作为注意力偏置

2. 成对表示更新：
   x = OuterMean(m) + x               # 从 MSA 生成成对信息
   x = TriangleMultiply(x) + x        # 三角更新
   x = TriangleAttention(x) + x       # 三角注意力
   x = FeedForward(x) + x

返回: (x, m)
```

**信息流**:
- MSA → 成对表示（OuterMean）
- 成对表示 → MSA（注意力偏置）
- 双向信息流使两个表示相互增强

---

### 4. Structure Module (结构模块)

**位置**: `alphafold2_pytorch/alphafold2.py:1264-1339`

**核心逻辑**:
```
输入: single_repr (单链表示, batch, n, dim)
     pairwise_repr (成对表示, batch, n, n, dim)
     mask

初始化:
- quaternions = [1, 0, 0, 0]  # 单位四元数（无旋转）
- translations = [0, 0, 0]     # 零平移

迭代细化（structure_module_depth 次）:
for i in range(depth):
    # 1. 转换为旋转矩阵
    rotations = quaternion_to_matrix(quaternions)

    # 2. 不变点注意力（考虑当前坐标系）
    single_repr = IPABlock(
        single_repr,
        pairwise_repr=pairwise_repr,
        rotations=rotations,
        translations=translations
    )

    # 3. 预测更新
    quat_update, trans_update = predict_update(single_repr)

    # 4. 应用更新
    quaternions = quaternion_multiply(quaternions, quat_update)
    translations = translations + rotate(trans_update, rotations)

最终坐标:
local_points = self.to_points(single_repr)  # 局部坐标
coords = rotate(local_points, rotations) + translations

返回: coords (batch, n, 3)
```

**原理**:
- 使用不变点注意力（IPA）：考虑3D空间中的几何约束
- 迭代更新旋转和平移：逐步细化结构
- 局部坐标系：每个残基有自己的坐标系，最后转换到全局坐标

---

## 关键函数

### 1. Alphafold2.forward()

**位置**: `alphafold2_pytorch/alphafold2.py:1001-1339`

**完整流程**:

```
输入处理:
├─ 序列嵌入（token embedding + 预训练嵌入）
├─ MSA 嵌入
├─ 成对表示生成（外积）
└─ 相对位置编码

循环输入（如果提供）:
├─ 添加上次的 MSA 表示
├─ 添加上次的成对表示
└─ 添加上次预测坐标的距离嵌入

模板处理（如果提供）:
├─ 模板特征嵌入
├─ 多层成对注意力处理
├─ 点注意力池化多个模板
└─ 模板角度特征拼接到 MSA

额外 MSA 处理:
└─ Extra MSA Evoformer（全局列注意力）

主干处理:
└─ Main Evoformer（多层 EvoformerBlock）

辅助预测:
├─ 距离图预测（对称化的成对表示）
├─ 二面角预测（θ, φ, ω）
└─ MLM 损失（训练时）

结构预测（如果 predict_coords=True）:
├─ 准备单链和成对表示
├─ 转换为 float32 精度
├─ 迭代结构细化（IPA + 坐标更新）
└─ 计算最终 3D 坐标

返回:
└─ 坐标 或 ReturnValues
```

---

### 2. OuterMean (外积均值)

**位置**: `alphafold2_pytorch/alphafold2.py:499-553`

**核心逻辑**:
```
输入: m (MSA 表示, batch, num_msa, seq_len, dim)
     mask (MSA 掩码)

步骤:
1. 投影：
   left = self.left_proj(m)   # (b, m, i, d)
   right = self.right_proj(m)  # (b, m, j, d)

2. 外积：
   outer = left[:,:,i,:] * right[:,:,j,:]  # (b, m, i, j, d)

3. 掩码均值（在 MSA 维度）：
   if mask:
       outer = masked_mean(outer, mask, dim=1)
   else:
       outer = outer.mean(dim=1)  # (b, i, j, d)

4. 输出投影：
   return self.proj_out(outer)
```

**功能**:
从 MSA 中提取成对信息，利用协同进化信号（相关突变）推断残基对之间的关系。

---

### 3. AxialAttention (轴向注意力)

**位置**: `alphafold2_pytorch/alphafold2.py:304-398`

**核心逻辑**:
```
输入: x (2D 特征图, batch, height, width, dim)
     edges (边信息，用于注意力偏置)
     mask

行注意力 (row_attn=True):
- 固定行索引 h，在列维度做注意力
- x[b, h, :, :] -> attention -> x'[b, h, :, :]
- 复杂度：O(n² * n) = O(n³)

列注意力 (col_attn=True):
- 固定列索引 w，在行维度做注意力
- x[b, :, w, :] -> attention -> x'[b, :, w, :]
- 复杂度：O(n * n²) = O(n³)

总复杂度：O(n³) vs 全局注意力的 O(n⁴)
```

**优势**:
降低成对表示上的注意力复杂度，从 O(n⁴) 降到 O(n³)。

---

### 4. MLM.noise() (掩码噪声)

**位置**: `alphafold2_pytorch/mlm.py:87-139`

**核心逻辑**:
```
输入: seq (MSA 序列)
     mask (MSA 掩码)

步骤:
1. 选择要掩码的位置（15%）：
   mlm_mask = sample(mask, prob=0.15)

2. 对选中的位置应用三种策略：
   a) 80%: 替换为 [MASK] token
      seq[mlm_mask] = MASK_ID

   b) 10%: 替换为随机 token
      random_mask = sample(mlm_mask, prob=0.1)
      seq[random_mask] = random_token

   c) 10%: 保持不变
      keep_mask = sample(mlm_mask, prob=0.1)
      seq[keep_mask] = seq[keep_mask]

3. 返回：
   return noised_seq, mlm_mask
```

**目的**:
BERT 风格的自监督学习，让模型学习蛋白质序列的表示。

---

### 5. 预训练嵌入集成

#### ESMEmbedWrapper

**位置**: `alphafold2_pytorch/embeds.py:155-214`

**流程**:
```
1. 加载 ESM-1b 模型（650M 参数）
2. 对序列编码：
   seq_embed = ESM(seq)  # (batch, seq_len, 1280)
3. 对 MSA 逐行编码：
   for each msa_row:
       msa_embed[i] = ESM(msa_row)
4. 投影到模型维度：
   seq_embed = linear(seq_embed)
   msa_embed = linear(msa_embed)
5. 传递给 AlphaFold2：
   output = alphafold2(seq, msa, seq_embed, msa_embed)
```

#### MSAEmbedWrapper

**位置**: `alphafold2_pytorch/embeds.py:73-153`

**流程**:
```
1. 加载 MSA Transformer 模型
2. 拼接序列和 MSA：
   seq_and_msa = cat([seq, msa], dim=1)
3. 一次性编码（行绑定注意力）：
   embeds = MSATransformer(seq_and_msa)
4. 分离嵌入：
   seq_embed = embeds[:, 0]
   msa_embed = embeds[:, 1:]
5. 传递给 AlphaFold2
```

**区别**:
- ESM: 单序列模型，逐行处理
- MSA Transformer: 专门的 MSA 模型，利用行间关系

---

## 总结

### 核心创新

1. **双重表示**:
   - MSA 表示：捕获进化信息
   - 成对表示：捕获残基对关系
   - 双向信息流：相互增强

2. **三角更新**:
   - 利用三角不等式传播几何信息
   - 全局一致性约束

3. **不变点注意力（IPA）**:
   - 考虑 3D 几何约束
   - 等变性：旋转和平移不变

4. **循环细化**:
   - 迭代改进预测
   - 使用上次预测作为先验

### 关键设计

- **轴向注意力**: 降低计算复杂度
- **门控机制**: 控制信息流
- **混合精度**: float32 用于结构模块，保证精度
- **梯度检查点**: 节省内存
- **零初始化**: 训练稳定性

---

## 参考

- 原论文: Jumper et al., "Highly accurate protein structure prediction with AlphaFold" (Nature 2021)
- 本实现: https://github.com/lucidrains/alphafold2
