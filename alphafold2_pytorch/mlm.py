"""
掩码语言模型（Masked Language Model, MLM）

用于 MSA 的自监督学习。通过随机掩码 MSA 中的氨基酸，
训练模型预测被掩码的氨基酸，从而学习蛋白质序列的表示。

这是 BERT 风格的预训练任务，已被证明对蛋白质结构预测有帮助。
"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from alphafold2_pytorch import constants
from einops import rearrange

# MSA MLM

def get_mask_subset_with_prob(mask, prob):
    """
    根据概率从掩码中采样一个子集

    参数:
        mask: 布尔掩码 (batch, seq_len)，True 表示有效位置
        prob: 采样概率

    返回:
        新的布尔掩码，表示被选中的位置
    """
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)  # 最多掩码的位置数

    # 计算每个序列的有效 token 数量
    num_tokens = mask.sum(dim=-1, keepdim=True)
    # 确保不超过目标掩码比例
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    # 随机采样
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    # 创建新掩码
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

class MLM(nn.Module):
    """
    掩码语言模型（Masked Language Model）

    用于 MSA 的自监督训练。实现类似 BERT 的掩码策略：
    - 80% 的时间：用 [MASK] token 替换
    - 10% 的时间：用随机 token 替换
    - 10% 的时间：保持不变

    参数:
        dim: 嵌入维度
        num_tokens: 词汇表大小（氨基酸类型数）
        mask_id: 掩码 token 的 ID
        mask_prob: 掩码概率（默认 0.15）
        random_replace_token_prob: 随机替换的概率（默认 0.1）
        keep_token_same_prob: 保持不变的概率（默认 0.1）
        exclude_token_ids: 排除的 token ID（如填充、起始、结束符号）
    """
    def __init__(
        self,
        dim,
        num_tokens,
        mask_id,
        mask_prob = 0.15,
        random_replace_token_prob = 0.1,
        keep_token_same_prob = 0.1,
        exclude_token_ids = (0,)
    ):
        super().__init__()
        self.to_logits = nn.Linear(dim, num_tokens)  # 输出层，预测氨基酸类型
        self.mask_id = mask_id

        self.mask_prob = mask_prob
        self.exclude_token_ids = exclude_token_ids  # 不进行掩码的 token（如 PAD）
        self.keep_token_same_prob = keep_token_same_prob
        self.random_replace_token_prob = random_replace_token_prob

    def noise(self, seq, mask):
        """
        对序列添加噪声（掩码策略）

        参数:
            seq: MSA 序列 (batch, num_msa, seq_len)
            mask: MSA 掩码 (batch, num_msa, seq_len)

        返回:
            noised_seq: 添加噪声后的序列
            mlm_mask: MLM 掩码（True 表示被掩码的位置）
        """
        num_msa = seq.shape[1]
        seq = rearrange(seq, 'b n ... -> (b n) ...')
        mask = rearrange(mask, 'b n ... -> (b n) ...')

        # prepare masks for noising sequence - 准备掩码

        excluded_tokens_mask = mask

        # 排除特殊 token（如 PAD）
        for token_id in self.exclude_token_ids:
            excluded_tokens_mask = excluded_tokens_mask & (seq != token_id)

        # 根据概率选择要掩码的位置
        mlm_mask = get_mask_subset_with_prob(excluded_tokens_mask, self.mask_prob)

        # keep some tokens the same - 10% 的时间保持不变

        replace_token_with_mask = get_mask_subset_with_prob(mlm_mask, 1. - self.keep_token_same_prob)

        # replace with mask - 用 [MASK] token 替换

        seq = seq.masked_fill(mlm_mask, self.mask_id)

        # generate random tokens - 生成随机 token（10% 的时间随机替换）

        random_replace_token_prob_mask = get_mask_subset_with_prob(mlm_mask, (1 - self.keep_token_same_prob) * self.random_replace_token_prob)

        random_tokens = torch.randint(1, constants.NUM_AMINO_ACIDS, seq.shape).to(seq.device)

        # 确保不会替换成排除的 token 类型
        for token_id in self.exclude_token_ids:
            random_replace_token_prob_mask = random_replace_token_prob_mask & (random_tokens != token_id)

        # noise sequence - 应用噪声

        noised_seq = torch.where(random_replace_token_prob_mask, random_tokens, seq)
        # 恢复原始形状
        noised_seq = rearrange(noised_seq, '(b n) ... -> b n ...', n = num_msa)
        mlm_mask = rearrange(mlm_mask, '(b n) ... -> b n ...', n = num_msa)

        return noised_seq, mlm_mask

    def forward(self, seq_embed, original_seq, mask):
        """
        计算 MLM 损失

        参数:
            seq_embed: 序列嵌入 (batch, num_msa, seq_len, dim)
            original_seq: 原始序列（标签）(batch, num_msa, seq_len)
            mask: MLM 掩码（被掩码的位置）(batch, num_msa, seq_len)

        返回:
            交叉熵损失
        """
        logits = self.to_logits(seq_embed)
        # 只计算被掩码位置的损失
        seq_logits = logits[mask]
        seq_labels = original_seq[mask]

        loss = F.cross_entropy(seq_logits, seq_labels, reduction = 'mean')
        return loss
