"""
AlphaFold2 的 PyTorch 实现

这是一个基于 PyTorch 的 AlphaFold2 蛋白质结构预测模型实现。
主要组件包括：
- Evoformer：处理多序列比对(MSA)和成对表示的核心模块
- Structure Module：基于不变点注意力(IPA)的结构细化模块
- 掩码语言模型(MLM)：用于 MSA 的自监督学习
"""

import torch
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from inspect import isfunction
from functools import partial
from dataclasses import dataclass
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from alphafold2_pytorch.utils import *
import alphafold2_pytorch.constants as constants
from alphafold2_pytorch.mlm import MLM

# structure module

from invariant_point_attention import IPABlock
from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix

# constants

@dataclass
class Recyclables:
    """
    可循环使用的中间结果

    在 AlphaFold2 的循环推理过程中，这些中间结果会被重复使用，
    以逐步细化蛋白质结构预测。

    属性:
        coords: 3D 坐标 (batch, seq_len, 3)
        single_msa_repr_row: MSA 的单个表示行 (batch, seq_len, dim)
        pairwise_repr: 成对残基表示 (batch, seq_len, seq_len, dim)
    """
    coords: torch.Tensor
    single_msa_repr_row: torch.Tensor
    pairwise_repr: torch.Tensor

@dataclass
class ReturnValues:
    """
    模型输出值容器

    属性:
        distance: 距离图预测 (batch, seq_len, seq_len, num_distance_bins)
        theta: θ 二面角预测 (batch, seq_len, seq_len, num_theta_bins)
        phi: φ 二面角预测 (batch, seq_len, seq_len, num_phi_bins)
        omega: ω 二面角预测 (batch, seq_len, seq_len, num_omega_bins)
        msa_mlm_loss: MSA 掩码语言模型损失
        recyclables: 可循环使用的中间结果
    """
    distance: torch.Tensor = None
    theta: torch.Tensor = None
    phi: torch.Tensor = None
    omega: torch.Tensor = None
    msa_mlm_loss: torch.Tensor = None
    recyclables: Recyclables = None

# helpers

def exists(val):
    """检查值是否存在（不为 None）"""
    return val is not None

def default(val, d):
    """
    返回默认值（如果 val 不存在）

    参数:
        val: 要检查的值
        d: 默认值或返回默认值的函数
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth = 1):
    """
    将值转换为元组，如果已经是元组则保持不变

    参数:
        val: 要转换的值
        depth: 元组的长度
    """
    return val if isinstance(val, tuple) else (val,) * depth

def init_zero_(layer):
    """
    将层的权重和偏置初始化为零

    参数:
        layer: PyTorch 层（通常是 nn.Linear）
    """
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# helper classes

class Always(nn.Module):
    """
    始终返回固定值的模块（用于条件禁用某些嵌入）

    参数:
        val: 要返回的固定值
    """
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val

# feed forward

class GEGLU(nn.Module):
    """
    门控线性单元（Gated Linear Unit）使用 GELU 激活

    将输入分成两部分：值和门控信号，然后使用 GELU 激活的门控信号
    来调制值。这种机制在 Transformer 中被证明比标准 FFN 更有效。
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    """
    前馈神经网络（带有 GEGLU 激活）

    参数:
        dim: 输入和输出维度
        mult: 隐藏层维度的倍数（默认为 4）
        dropout: Dropout 概率

    结构:
        LayerNorm -> Linear -> GEGLU -> Dropout -> Linear
    最后一层线性层初始化为零，这有助于训练稳定性。
    """
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),  # *2 是因为 GEGLU 需要分成两部分
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])  # 零初始化最后一层，增强训练稳定性

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)

# attention

class Attention(nn.Module):
    """
    Used in multiple algorithms: 7, 8, 13, 14 (Gated Self-Attention)
    多头自注意力机制（带门控）

    这是 AlphaFold2 中使用的注意力机制，具有以下特点：
    - 多头注意力
    - 门控机制（gating）用于控制信息流
    - 支持注意力偏置（用于融合成对表示信息）
    - 支持绑定维度（tie_dim），用于 MSA 列全局注意力

    参数:
        dim: 输入维度
        seq_len: 序列长度（可选）
        heads: 注意力头数
        dim_head: 每个头的维度
        dropout: Dropout 概率
        gating: 是否使用门控机制
    """
    def __init__(
        self,
        dim,
        seq_len = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5  # 缩放因子，用于稳定梯度

        # Q, K, V 投影层
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # K 和 V 共享投影
        self.to_out = nn.Linear(inner_dim, dim)

        # 门控层：初始化为恒等变换（权重为0，偏置为1，sigmoid后接近1）
        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)  # 输出层零初始化

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        """
        前向传播

        参数:
            x: 查询输入 (batch, seq_len, dim)
            mask: 查询掩码
            attn_bias: 注意力偏置（用于注入成对表示信息）
            context: 上下文输入（用于交叉注意力，如果为 None 则为自注意力）
            context_mask: 上下文掩码
            tie_dim: 绑定维度，用于 MSAColumnGlobalAttention
                    （对 MSA 的行进行平均，实现全局列注意力）

        返回:
            输出张量 (batch, seq_len, dim)
        """
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)  # 如果没有提供 context，则使用 x（自注意力）

        # 计算 Q, K, V
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        # 重排为多头格式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale - 缩放查询向量以稳定训练

        q = q * self.scale

        # query / key similarities - 计算注意力得分

        if exists(tie_dim):
            # MSAColumnGlobalAttention：对 MSA 的行进行平均
            # 这用于额外的 MSA 处理，可以减少计算量

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)  # 对行求平均

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias - 添加注意力偏置（用于成对表示到 MSA 的通信）

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking - 应用掩码（用于处理填充和无效位置）

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention - 计算注意力权重

        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate - 聚合值向量

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads - 合并多头

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating - 应用门控机制

        gates = self.gating(x)
        out = out * gates.sigmoid()  # 门控值在 [0, 1] 之间

        # combine to out - 投影到输出维度

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    """
    轴向注意力（Axial Attention）

    在 2D 特征图上分别沿行或列方向进行注意力计算，这样可以降低
    计算复杂度从 O(n^4) 到 O(n^3)，其中 n 是序列长度。

    参数:
        dim: 输入维度
        heads: 注意力头数
        row_attn: 是否使用行注意力
        col_attn: 是否使用列注意力
        accept_edges: 是否接受边（成对表示）作为注意力偏置
        global_query_attn: 是否使用全局查询注意力（用于 MSA 列注意力）
        **kwargs: 传递给 Attention 的其他参数
    """
    def __init__(
        self,
        dim,
        heads,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        global_query_attn = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim = dim, heads = heads, **kwargs)

        # 如果接受边，将边转换为注意力偏置
        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges = None, mask = None):
        """
        前向传播

        参数:
            x: 输入特征图 (batch, height, width, dim)
            edges: 边信息（成对表示），用于生成注意力偏置
            mask: 掩码 (batch, height, width)

        返回:
            输出特征图 (batch, height, width, dim)
        """
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention - 根据行/列方向设置重排方式

        if self.col_attn:
            # 列注意力：固定列，在行上做注意力
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            # 行注意力：固定行，在列上做注意力
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        # 如果提供了边信息，转换为注意力偏置
        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        # 如果使用全局查询注意力，绑定轴向维度
        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask = mask, attn_bias = attn_bias, tie_dim = tie_dim)
        out = rearrange(out, output_fold_eq, h = h, w = w)

        return out

class TriangleMultiplicativeModule(nn.Module):
    """
    Algorithm 11 (Outgoing) / Algorithm 12 (Incoming)
    三角乘法模块（Triangle Multiplicative Module）

    这是 AlphaFold2 的核心创新之一，用于更新成对表示。
    它利用三角不等式约束来传播距离和方向信息。

    工作原理：
    对于残基 i, j, k，如果已知 i-k 和 j-k 的关系，
    可以推断 i-j 的关系（三角形的三边关系）。

    参数:
        dim: 输入和输出维度
        hidden_dim: 隐藏层维度
        mix: 混合方式 ('ingoing' 或 'outgoing')
            - 'outgoing': 沿着出边聚合 (i->k, j->k 推断 i-j) - Algorithm 11
            - 'ingoing': 沿着入边聚合 (k->i, k->j 推断 i-j) - Algorithm 12
    """
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        # 左右投影层（用于生成要相乘的两个表示）
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        # 门控层
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity - 初始化门控为恒等变换

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        # 根据混合方式设置不同的爱因斯坦求和公式
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'  # 共享出边 k
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'  # 共享入边 k

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        """
        前向传播

        参数:
            x: 成对表示 (batch, seq_len, seq_len, dim)
            mask: 掩码 (batch, seq_len, seq_len)

        返回:
            更新后的成对表示 (batch, seq_len, seq_len, dim)
        """
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        # 投影到左右表示
        left = self.left_proj(x)
        right = self.right_proj(x)

        # 应用掩码（用于处理填充）
        if exists(mask):
            left = left * mask
            right = right * mask

        # 计算门控值
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        # 应用门控
        left = left * left_gate
        right = right * right_gate

        # 三角更新：通过中间节点 k 连接 i 和 j
        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate  # 输出门控
        return self.to_out(out)

# evoformer blocks

class OuterMean(nn.Module):
    """
    Algorithm 9 / Algorithm 10 - OuterProductMean
    外积均值（Outer Product Mean）

    从 MSA 表示生成成对表示的模块。
    计算 MSA 中每对位置的外积，然后在 MSA 维度上求平均。

    这允许模型从 MSA 中的协同进化信息推断残基对之间的关系。

    参数:
        dim: MSA 表示的维度
        hidden_dim: 隐藏层维度
        eps: 数值稳定性常数
    """
    def __init__(
        self,
        dim,
        hidden_dim = None,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        """
        前向传播

        参数:
            x: MSA 表示 (batch, num_msa, seq_len, dim)
            mask: MSA 掩码 (batch, num_msa, seq_len)

        返回:
            成对表示 (batch, seq_len, seq_len, dim)
        """
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        # 计算外积：每个 MSA 序列的每对位置之间的乘积
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')

        if exists(mask):
            # masked mean - 如果 MSA 中有填充，使用掩码均值
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim = 1) / (mask.sum(dim = 1) + self.eps)
        else:
            outer = outer.mean(dim = 1)  # 在 MSA 维度上求平均

        return self.proj_out(outer)

class PairwiseAttentionBlock(nn.Module):
    """
    Algorithm 6 - line 5-12: Pair stack operations
    成对注意力块（Pairwise Attention Block）

    更新成对残基表示的主要模块，包含：
    1. OuterMean: 从 MSA 生成成对信息 (Algorithm 9/10: OuterProductMean)
    2. Triangle Multiplication (出边/入边): 利用三角不等式传播信息
       - Algorithm 11: TriangleMultiplicationOutgoing
       - Algorithm 12: TriangleMultiplicationIncoming
    3. Triangle Attention (行/列): 沿着行列方向的注意力
       - Algorithm 13: TriangleAttentionStartingNode
       - Algorithm 14: TriangleAttentionEndingNode

    参数:
        dim: 特征维度
        seq_len: 序列长度
        heads: 注意力头数
        dim_head: 每个头的维度
        dropout: Dropout 概率
        global_column_attn: 是否在列注意力中使用全局查询
    """
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.,
        global_column_attn = False
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        # 三角注意力（出边：行注意力）
        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        # 三角注意力（入边：列注意力）
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)
        # 三角乘法更新（出边）
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')
        # 三角乘法更新（入边）
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
        msa_mask = None
    ):
        """
        前向传播

        参数:
            x: 成对表示 (batch, seq_len, seq_len, dim)
            mask: 成对掩码 (batch, seq_len, seq_len)
            msa_repr: MSA 表示 (batch, num_msa, seq_len, dim)
            msa_mask: MSA 掩码 (batch, num_msa, seq_len)

        返回:
            更新后的成对表示 (batch, seq_len, seq_len, dim)
        """
        # Algorithm 9/10 - OuterProductMean
        # 如果提供了 MSA，通过外积均值更新成对表示
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask = msa_mask)

        # Algorithm 11 - TriangleMultiplicationOutgoing
        # Algorithm 12 - TriangleMultiplicationIncoming
        # 三角乘法更新（出边和入边）
        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        # Algorithm 13 - TriangleAttentionStartingNode
        # Algorithm 14 - TriangleAttentionEndingNode
        # 三角注意力更新（行和列）
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
        return x

class MsaAttentionBlock(nn.Module):
    """
    Algorithm 6 - line 2-3: MSA row and column attention
    MSA 注意力块（MSA Attention Block）

    更新 MSA 表示的模块，包含：
    1. 行注意力（Row Attention）: 在序列维度上进行注意力，允许成对表示作为偏置
       - Algorithm 7: MSARowAttentionWithPairBias
    2. 列注意力（Column Attention）: 在 MSA 维度上进行注意力
       - Algorithm 8: MSAColumnAttention

    这使得模型可以：
    - 在同一 MSA 序列的不同位置之间传播信息（行注意力）
    - 在不同 MSA 序列的相同位置之间传播信息（列注意力）

    参数:
        dim: 特征维度
        seq_len: 序列长度
        heads: 注意力头数
        dim_head: 每个头的维度
        dropout: Dropout 概率
    """
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.
    ):
        super().__init__()
        # 行注意力：接受成对表示作为偏置
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        # 列注意力：在 MSA 序列间通信
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        """
        前向传播

        参数:
            x: MSA 表示 (batch, num_msa, seq_len, dim)
            mask: MSA 掩码 (batch, num_msa, seq_len)
            pairwise_repr: 成对表示，作为行注意力的偏置 (batch, seq_len, seq_len, dim)

        返回:
            更新后的 MSA 表示 (batch, num_msa, seq_len, dim)
        """
        # Algorithm 7 - MSARowAttentionWithPairBias
        # 行注意力：利用成对表示作为注意力偏置
        x = self.row_attn(x, mask = mask, edges = pairwise_repr) + x
        # Algorithm 8 - MSAColumnAttention
        # 列注意力：在 MSA 序列间传播信息
        x = self.col_attn(x, mask = mask) + x
        return x

# main evoformer class

class EvoformerBlock(nn.Module):
    """
    Algorithm 6 - EvoformerStack (single block)

    Evoformer 块

    AlphaFold2 的核心计算单元，同时更新 MSA 表示和成对表示。

    信息流：
    1. MSA 注意力 + FFN: 更新 MSA 表示
    2. 成对注意力 + FFN: 更新成对表示（可以从 MSA 获取信息）

    这种双向信息流允许 MSA 和成对表示相互增强。

    参数:
        dim: 特征维度
        seq_len: 序列长度
        heads: 注意力头数
        dim_head: 每个头的维度
        attn_dropout: 注意力层的 Dropout
        ff_dropout: 前馈层的 Dropout
        global_column_attn: 是否在成对表示的列注意力中使用全局查询
    """
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, global_column_attn = global_column_attn),
            FeedForward(dim = dim, dropout = ff_dropout),
            MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim = dim, dropout = ff_dropout),
        ])

    def forward(self, inputs):
        """
        前向传播

        参数:
            inputs: 元组 (x, m, mask, msa_mask)
                x: 成对表示 (batch, seq_len, seq_len, dim)
                m: MSA 表示 (batch, num_msa, seq_len, dim)
                mask: 成对掩码 (batch, seq_len, seq_len)
                msa_mask: MSA 掩码 (batch, num_msa, seq_len)

        返回:
            更新后的 (x, m, mask, msa_mask)
        """
        x, m, mask, msa_mask = inputs
        attn, ff, msa_attn, msa_ff = self.layer

        # Algorithm 6 - line 2-4: MSA stack (row attention, column attention, transition)
        # msa attention and transition - MSA 注意力和前馈

        m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        m = msa_ff(m) + m

        # Algorithm 6 - line 5-10: Communication (outer product mean, triangular operations)
        # Algorithm 6 - line 11-12: Pair stack (triangular attention, transition)
        # pairwise attention and transition - 成对注意力和前馈

        x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = ff(x) + x

        return x, m, mask, msa_mask

class Evoformer(nn.Module):
    """
    Evoformer 主模块

    堆叠多个 EvoformerBlock，形成深度神经网络。
    使用 checkpoint_sequential 来节省内存（梯度检查点技术）。

    参数:
        depth: 堆叠的 EvoformerBlock 数量
        **kwargs: 传递给每个 EvoformerBlock 的参数
    """
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        """
        前向传播

        参数:
            x: 成对表示 (batch, seq_len, seq_len, dim)
            m: MSA 表示 (batch, num_msa, seq_len, dim)
            mask: 成对掩码 (batch, seq_len, seq_len)
            msa_mask: MSA 掩码 (batch, num_msa, seq_len)

        返回:
            更新后的 (x, m)
        """
        inp = (x, m, mask, msa_mask)
        # 使用梯度检查点技术顺序执行各层，节省内存
        x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        return x, m

class Alphafold2(nn.Module):
    """
    AlphaFold2 主模型

    这是完整的 AlphaFold2 蛋白质结构预测模型，包括：
    1. 输入嵌入层（序列、MSA、模板）
    2. Extra MSA Evoformer（处理额外的 MSA）
    3. Main Evoformer（核心特征提取）
    4. Structure Module（3D 坐标预测）
    5. 辅助任务（距离图、角度预测、MLM）

    参数:
        dim: 模型的主要特征维度
        max_seq_len: 最大序列长度
        depth: 主 Evoformer 的深度（层数）
        heads: 注意力头数
        dim_head: 每个注意力头的维度
        max_rel_dist: 相对位置编码的最大距离
        num_tokens: 氨基酸类型数量（包括特殊 token）
        num_embedds: 预训练嵌入的维度（如 ProtTrans, ESM）
        max_num_msas: 最大 MSA 序列数
        max_num_templates: 最大模板数量
        extra_msa_evoformer_layers: 额外 MSA Evoformer 的层数
        attn_dropout: 注意力层的 Dropout 概率
        ff_dropout: 前馈层的 Dropout 概率
        templates_dim: 模板特征维度
        templates_embed_layers: 模板嵌入层数
        templates_angles_feats_dim: 模板角度特征维度
        predict_angles: 是否预测二面角
        symmetrize_omega: 是否对 omega 角进行对称化
        predict_coords: 是否预测 3D 坐标
        structure_module_depth: 结构模块的迭代次数
        structure_module_heads: 结构模块的注意力头数
        structure_module_dim_head: 结构模块每个头的维度
        disable_token_embed: 是否禁用 token 嵌入（使用预训练嵌入时）
        mlm_mask_prob: MLM 的掩码概率
        mlm_random_replace_token_prob: MLM 随机替换概率
        mlm_keep_token_same_prob: MLM 保持不变概率
        mlm_exclude_token_ids: MLM 排除的 token ID
        recycling_distance_buckets: 循环时距离离散化的桶数
    """
    def __init__(
        self,
        *,
        dim,
        max_seq_len = 2048,
        depth = 6,
        heads = 8,
        dim_head = 64,
        max_rel_dist = 32,
        num_tokens = constants.NUM_AMINO_ACIDS,
        num_embedds = constants.NUM_EMBEDDS_TR,
        max_num_msas = constants.MAX_NUM_MSA,
        max_num_templates = constants.MAX_NUM_TEMPLATES,
        extra_msa_evoformer_layers = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        templates_dim = 32,
        templates_embed_layers = 4,
        templates_angles_feats_dim = 55,
        predict_angles = False,
        symmetrize_omega = False,
        predict_coords = False,                # structure module related keyword arguments below
        structure_module_depth = 4,
        structure_module_heads = 1,
        structure_module_dim_head = 4,
        disable_token_embed = False,
        mlm_mask_prob = 0.15,
        mlm_random_replace_token_prob = 0.1,
        mlm_keep_token_same_prob = 0.1,
        mlm_exclude_token_ids = (0,),
        recycling_distance_buckets = 32
    ):
        super().__init__()
        self.dim = dim

        # token embedding - 氨基酸序列嵌入

        self.token_emb = nn.Embedding(num_tokens + 1, dim) if not disable_token_embed else Always(0)
        self.to_pairwise_repr = nn.Linear(dim, dim * 2)  # 投影到成对表示
        self.disable_token_embed = disable_token_embed

        # positional embedding - 相对位置嵌入

        self.max_rel_dist = max_rel_dist
        self.pos_emb = nn.Embedding(max_rel_dist * 2 + 1, dim)  # 对称的相对位置编码

        # extra msa embedding - 额外的 MSA 处理模块

        self.extra_msa_evoformer = Evoformer(
            dim = dim,
            depth = extra_msa_evoformer_layers,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            global_column_attn = True  # 使用全局列注意力以减少计算
        )

        # template embedding - 模板结构嵌入

        self.to_template_embed = nn.Linear(templates_dim, dim)
        self.templates_embed_layers = templates_embed_layers

        # 模板成对嵌入器（多层处理模板信息）
        self.template_pairwise_embedder = PairwiseAttentionBlock(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            seq_len = max_seq_len
        )

        # 模板点注意力（池化多个模板）
        self.template_pointwise_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            dropout = attn_dropout
        )

        # 模板角度特征的 MLP
        self.template_angle_mlp = nn.Sequential(
            nn.Linear(templates_angles_feats_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # projection for angles - 角度预测头（如果需要）

        self.predict_angles = predict_angles
        self.symmetrize_omega = symmetrize_omega

        if predict_angles:
            self.to_prob_theta = nn.Linear(dim, constants.THETA_BUCKETS)  # θ 角
            self.to_prob_phi   = nn.Linear(dim, constants.PHI_BUCKETS)    # φ 角
            self.to_prob_omega = nn.Linear(dim, constants.OMEGA_BUCKETS)  # ω 角

        # custom embedding projection - 预训练嵌入投影

        self.embedd_project = nn.Linear(num_embedds, dim)

        # main trunk modules - 主干 Evoformer

        self.net = Evoformer(
            dim = dim,
            depth = depth,
            seq_len = max_seq_len,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # MSA SSL MLM - MSA 掩码语言模型（自监督学习）

        self.mlm = MLM(
            dim = dim,
            num_tokens = num_tokens,
            mask_id = num_tokens,  # 嵌入表的最后一个 token 用于掩码
            mask_prob = mlm_mask_prob,
            keep_token_same_prob = mlm_keep_token_same_prob,
            random_replace_token_prob = mlm_random_replace_token_prob,
            exclude_token_ids = mlm_exclude_token_ids
        )

        # calculate distogram logits - 距离图预测

        self.to_distogram_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, constants.DISTOGRAM_BUCKETS)
        )

        # to coordinate output - 3D 坐标预测（结构模块）

        self.predict_coords = predict_coords
        self.structure_module_depth = structure_module_depth

        # 从 MSA 和主干到结构模块的投影
        self.msa_to_single_repr_dim = nn.Linear(dim, dim)
        self.trunk_to_pairwise_repr_dim = nn.Linear(dim, dim)

        # 不变点注意力（IPA）模块，使用 float32 精度
        with torch_default_dtype(torch.float32):
            self.ipa_block = IPABlock(
                dim = dim,
                heads = structure_module_heads,
            )

            self.to_quaternion_update = nn.Linear(dim, 6)  # 四元数和平移更新

        init_zero_(self.ipa_block.attn.to_out)  # 零初始化 IPA 输出

        self.to_points = nn.Linear(dim, 3)  # 局部坐标预测

        # aux confidence measure - 辅助置信度度量（lDDT）

        self.lddt_linear = nn.Linear(dim, 1)

        # recycling params - 循环参数（用于迭代细化）

        self.recycling_msa_norm = nn.LayerNorm(dim)
        self.recycling_pairwise_norm = nn.LayerNorm(dim)
        self.recycling_distance_embed = nn.Embedding(recycling_distance_buckets, dim)
        self.recycling_distance_buckets = recycling_distance_buckets

    def forward(
        self,
        seq,
        msa = None,
        mask = None,
        msa_mask = None,
        extra_msa = None,
        extra_msa_mask = None,
        seq_index = None,
        seq_embed = None,
        msa_embed = None,
        templates_feats = None,
        templates_mask = None,
        templates_angles = None,
        embedds = None,
        recyclables = None,
        return_trunk = False,
        return_confidence = False,
        return_recyclables = False,
        return_aux_logits = False
    ):
        # Algorithm 2 - Inference (AlphaFold Model Inference)
        """
        前向传播

        参数:
            seq: 氨基酸序列 (batch, seq_len) 或预训练嵌入
            msa: 多序列比对 (batch, num_msa, seq_len)
            mask: 序列掩码 (batch, seq_len)
            msa_mask: MSA 掩码 (batch, num_msa, seq_len)
            extra_msa: 额外的 MSA（用于预处理）(batch, num_extra_msa, seq_len)
            extra_msa_mask: 额外 MSA 的掩码
            seq_index: 序列索引（用于自定义位置编码）
            seq_embed: 预计算的序列嵌入
            msa_embed: 预计算的 MSA 嵌入
            templates_feats: 模板特征 (batch, num_templates, seq_len, seq_len, templates_dim)
            templates_mask: 模板掩码 (batch, num_templates, seq_len)
            templates_angles: 模板角度特征 (batch, num_templates, seq_len, templates_angles_feats_dim)
            embedds: 预训练嵌入（如 ESM、ProtTrans）
            recyclables: 循环的中间结果（用于迭代细化）
            return_trunk: 是否只返回主干输出（不进行结构预测）
            return_confidence: 是否返回置信度分数
            return_recyclables: 是否返回可循环的中间结果
            return_aux_logits: 是否返回辅助 logits

        返回:
            根据参数不同返回不同内容：
            - 默认：ReturnValues（包含距离图、角度等）
            - predict_coords=True: 3D 坐标或 (坐标, 辅助信息)
        """
        assert not (self.disable_token_embed and not exists(seq_embed)), 'sequence embedding must be supplied if one has disabled token embedding'
        assert not (self.disable_token_embed and not exists(msa_embed)), 'msa embedding must be supplied if one has disabled token embedding'

        # if MSA is not passed in, just use the sequence itself
        # 如果没有提供 MSA，使用序列本身作为单序列 MSA

        if not exists(msa):
            msa = rearrange(seq, 'b n -> b () n')
            msa_mask = rearrange(mask, 'b n -> b () n')

        # assert on sequence length - 检查序列长度一致性

        assert msa.shape[-1] == seq.shape[-1], 'sequence length of MSA and primary sequence must be the same'

        # Algorithm 2 - line 1: Initialize representations
        # variables - 获取批次大小、序列长度和设备

        b, n, device = *seq.shape[:2], seq.device
        n_range = torch.arange(n, device = device)

        # unpack (AA_code, atom_pos) - 解包序列（如果包含原子位置）

        if isinstance(seq, (list, tuple)):
            seq, seq_pos = seq

        # Algorithm 2 - line 2: Embed input features (sequence embedding)
        # embed main sequence - 嵌入主序列

        x = self.token_emb(seq)

        if exists(seq_embed):
            x += seq_embed  # 添加预训练嵌入

        # mlm for MSAs - 训练时对 MSA 应用掩码语言模型

        if self.training and exists(msa):
            original_msa = msa
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

            # 添加噪声：随机掩码、替换或保持
            noised_msa, replaced_msa_mask = self.mlm.noise(msa, msa_mask)
            msa = noised_msa

        # Algorithm 2 - line 3: Embed MSA features
        # embed multiple sequence alignment (msa) - 嵌入多序列比对

        if exists(msa):
            m = self.token_emb(msa)

            if exists(msa_embed):
                m = m + msa_embed  # 添加预训练 MSA 嵌入

            # add single representation to msa representation
            # 将单序列表示加到 MSA 的每一行

            m = m + rearrange(x, 'b n d -> b () n d')

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(msa).bool())

        elif exists(embedds):
            # 使用预训练嵌入（如 ESM、ProtTrans）代替 MSA
            m = self.embedd_project(embedds)

            # get msa_mask to all ones if none was passed
            msa_mask = default(msa_mask, lambda: torch.ones_like(embedds[..., -1]).bool())
        else:
            raise Error('either MSA or embeds must be given')

        # Algorithm 2 - line 4: Initialize pair representation with relative positions
        # derive pairwise representation - 生成成对残基表示

        x_left, x_right = self.to_pairwise_repr(x).chunk(2, dim = -1)
        # 创建成对残基嵌入：外积式组合（i, j）
        x = rearrange(x_left, 'b i d -> b i () d') + rearrange(x_right, 'b j d-> b () j d')
        # 创建成对掩码
        x_mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j') if exists(mask) else None

        # add relative positional embedding - 添加相对位置编码

        seq_index = default(seq_index, lambda: torch.arange(n, device = device))
        # 计算相对距离矩阵
        seq_rel_dist = rearrange(seq_index, 'i -> () i ()') - rearrange(seq_index, 'j -> () () j')
        # 裁剪到最大相对距离范围并偏移到正数
        seq_rel_dist = seq_rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = self.pos_emb(seq_rel_dist)

        x = x + rel_pos_emb

        # Algorithm 2 - line 5-6: Add previous cycle outputs (recycling)
        # add recyclables, if present - 添加循环信息（用于迭代细化）

        if exists(recyclables):
            # 更新 MSA 的第一行（单序列表示）
            m[:, 0] = m[:, 0] + self.recycling_msa_norm(recyclables.single_msa_repr_row)
            # 更新成对表示
            x = x + self.recycling_pairwise_norm(recyclables.pairwise_repr)

            # 从上一次迭代的坐标计算距离，并嵌入到成对表示中
            distances = torch.cdist(recyclables.coords, recyclables.coords, p=2)
            boundaries = torch.linspace(2, 20, steps = self.recycling_distance_buckets, device = device)
            discretized_distances = torch.bucketize(distances, boundaries[:-1])
            distance_embed = self.recycling_distance_embed(discretized_distances)

            x = x + distance_embed

        # Algorithm 2 - line 7-9: Process template features
        # embed templates, if present - 嵌入模板结构信息

        if exists(templates_feats):
            _, num_templates, *_ = templates_feats.shape

            # embed template - 嵌入模板

            t = self.to_template_embed(templates_feats)
            # 创建模板的成对掩码
            t_mask_crossed = rearrange(templates_mask, 'b t i -> b t i ()') * rearrange(templates_mask, 'b t j -> b t () j')

            # 将批次和模板维度合并以进行批处理
            t = rearrange(t, 'b t ... -> (b t) ...')
            t_mask_crossed = rearrange(t_mask_crossed, 'b t ... -> (b t) ...')

            # 多层处理模板成对表示
            for _ in range(self.templates_embed_layers):
                t = self.template_pairwise_embedder(t, mask = t_mask_crossed)

            # 恢复模板维度
            t = rearrange(t, '(b t) ... -> b t ...', t = num_templates)
            t_mask_crossed = rearrange(t_mask_crossed, '(b t) ... -> b t ...', t = num_templates)

            # template pos emb - 模板位置嵌入（点注意力池化多个模板）

            # 重排以进行点注意力
            x_point = rearrange(x, 'b i j d -> (b i j) () d')
            t_point = rearrange(t, 'b t i j d -> (b i j) t d')
            x_mask_point = rearrange(x_mask, 'b i j -> (b i j) ()')
            t_mask_point = rearrange(t_mask_crossed, 'b t i j -> (b i j) t')

            # 使用注意力池化模板
            template_pooled = self.template_pointwise_attn(
                x_point,
                context = t_point,
                mask = x_mask_point,
                context_mask = t_mask_point
            )

            # 应用掩码（只保留有有效模板的位置）
            template_pooled_mask = rearrange(t_mask_point.sum(dim = -1) > 0, 'b -> b () ()')
            template_pooled = template_pooled * template_pooled_mask

            # 恢复原始形状并添加到成对表示
            template_pooled = rearrange(template_pooled, '(b i j) () d -> b i j d', i = n, j = n)
            x = x + template_pooled

        # Algorithm 2 - line 10: Add template torsion angle features to MSA
        # add template angle features to MSAs - 将模板角度特征添加到 MSA
        # 通过 MLP 处理后连接到 MSA

        if exists(templates_angles):
            t_angle_feats = self.template_angle_mlp(templates_angles)
            m = torch.cat((m, t_angle_feats), dim = 1)  # 在 MSA 维度上连接
            msa_mask = torch.cat((msa_mask, templates_mask), dim = 1)

        # Algorithm 2 - line 11-12: Extra MSA stack
        # embed extra msa, if present - 处理额外的 MSA（如果存在）

        if exists(extra_msa):
            extra_m = self.token_emb(msa)
            extra_msa_mask = default(extra_msa_mask, torch.ones_like(extra_m).bool())

            # 额外 MSA Evoformer 处理
            x, extra_m = self.extra_msa_evoformer(
                x,
                extra_m,
                mask = x_mask,
                msa_mask = extra_msa_mask
            )

        # Algorithm 2 - line 13: Main Evoformer stack (48 blocks)
        # trunk - 主干 Evoformer

        x, m = self.net(
            x,
            m,
            mask = x_mask,
            msa_mask = msa_mask
        )

        # ready output container - 准备输出容器

        ret = ReturnValues()

        # Algorithm 2 - line 14-17: Predict auxiliary outputs (distogram and angles)
        # calculate theta and phi before symmetrization
        # 在对称化之前计算 theta 和 phi 角（这两个角是非对称的）

        if self.predict_angles:
            ret.theta_logits = self.to_prob_theta(x)
            ret.phi_logits = self.to_prob_phi(x)

        # embeds to distogram - 从嵌入预测距离图

        trunk_embeds = (x + rearrange(x, 'b i j d -> b j i d')) * 0.5  # 对称化成对表示
        distance_pred = self.to_distogram_logits(trunk_embeds)
        ret.distance = distance_pred

        # calculate mlm loss, if training - 计算 MLM 损失（训练时）

        msa_mlm_loss = None
        if self.training and exists(msa):
            num_msa = original_msa.shape[1]
            msa_mlm_loss = self.mlm(m[:, :num_msa], original_msa, replaced_msa_mask)

        # determine angles, if specified - 预测角度（如果指定）

        if self.predict_angles:
            # omega 角可以选择使用对称化的表示
            omega_input = trunk_embeds if self.symmetrize_omega else x
            ret.omega_logits = self.to_prob_omega(omega_input)

        # 如果不预测坐标或只需要主干输出，提前返回
        if not self.predict_coords or return_trunk:
            return ret

        # Algorithm 2 - line 18: Extract single and pair representations for structure module
        # derive single and pairwise embeddings for structural refinement
        # 为结构细化准备单链和成对嵌入

        single_msa_repr_row = m[:, 0]  # MSA 的第一行（主序列）

        single_repr = self.msa_to_single_repr_dim(single_msa_repr_row)
        pairwise_repr = self.trunk_to_pairwise_repr_dim(x)

        # prepare float32 precision for equivariance
        # 使用 float32 精度以确保等变性（结构模块需要高精度）

        original_dtype = single_repr.dtype
        single_repr, pairwise_repr = map(lambda t: t.float(), (single_repr, pairwise_repr))

        # Algorithm 2 - line 19-24: Structure module with iterative IPA refinement (see Algorithm 20)
        # iterative refinement with equivariant transformer in high precision
        # 使用等变 Transformer 进行迭代细化（高精度）

        with torch_default_dtype(torch.float32):

            # Algorithm 20 - line 1: Initialize backbone frames (quaternions and translations)
            # 初始化旋转（单位四元数）和平移（零向量）
            quaternions = torch.tensor([1., 0., 0., 0.], device = device)  # [w, x, y, z]
            quaternions = repeat(quaternions, 'd -> b n d', b = b, n = n)
            translations = torch.zeros((b, n, 3), device = device)

            # Algorithm 20 - line 2-11: Iterate over structure module layers
            # go through the layers and apply invariant point attention and feedforward
            # 迭代应用不变点注意力和更新

            for i in range(self.structure_module_depth):
                is_last = i == (self.structure_module_depth - 1)

                # 将四元数转换为旋转矩阵
                # the detach comes from AlphaFold2 official implementation
                # 除了最后一层外，detach 旋转矩阵以节省内存
                rotations = quaternion_to_matrix(quaternions)

                if not is_last:
                    rotations = rotations.detach()

                # Algorithm 20 - line 3-5: Invariant Point Attention (IPA)
                # 不变点注意力（IPA）- 考虑当前坐标系
                single_repr = self.ipa_block(
                    single_repr,
                    mask = mask,
                    pairwise_repr = pairwise_repr,
                    rotations = rotations,
                    translations = translations
                )

                # Algorithm 20 - line 6-9: Update backbone frames
                # update quaternion and translation - 更新四元数和平移

                quaternion_update, translation_update = self.to_quaternion_update(single_repr).chunk(2, dim = -1)
                quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)  # 填充为 [1, ...]

                # 应用四元数更新（组合旋转）
                quaternions = quaternion_multiply(quaternions, quaternion_update)
                # 应用平移更新（在当前坐标系中）
                translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)

            # Algorithm 20 - line 10-11: Compute final atom positions
            # 计算最终坐标：局部坐标 -> 旋转 -> 平移
            points_local = self.to_points(single_repr)
            rotations = quaternion_to_matrix(quaternions)
            coords = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations

        coords.type(original_dtype)

        # 如果需要返回可循环的中间结果（用于下一次迭代）
        if return_recyclables:
            coords, single_msa_repr_row, pairwise_repr = map(torch.detach, (coords, single_msa_repr_row, pairwise_repr))
            ret.recyclables = Recyclables(coords, single_msa_repr_row, pairwise_repr)

        if return_aux_logits:
            return coords, ret

        if return_confidence:
            return coords, self.lddt_linear(single_repr.float())

        return coords
