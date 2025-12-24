"""
预训练蛋白质语言模型嵌入封装器

本模块提供了将预训练蛋白质语言模型（如 ProtTrans、ESM、MSA Transformer）
集成到 AlphaFold2 的封装类。这些模型可以提供高质量的序列和 MSA 嵌入，
替代或增强原始的 token 嵌入。
"""

import torch
import torch.nn.functional as F
from torch import nn

from alphafold2_pytorch.utils import get_msa_embedd, get_esm_embedd, get_prottran_embedd, exists
from alphafold2_pytorch.constants import MSA_MODEL_PATH, MSA_EMBED_DIM, ESM_MODEL_PATH, ESM_EMBED_DIM, PROTTRAN_EMBED_DIM

from einops import rearrange

class ProtTranEmbedWrapper(nn.Module):
    """
    ProtTrans (ProtBERT) 嵌入封装器

    ProtTrans 是基于 BERT 架构的蛋白质语言模型，在大规模蛋白质序列上预训练。
    它可以为蛋白质序列生成上下文相关的嵌入表示。

    参数:
        alphafold2: AlphaFold2 模型实例

    用法:
        model = Alphafold2(...)
        wrapped_model = ProtTranEmbedWrapper(alphafold2=model)
        output = wrapped_model(seq, msa, ...)
    """
    def __init__(self, *, alphafold2):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        self.alphafold2 = alphafold2
        # 投影层：将 ProtTrans 嵌入投影到 AlphaFold2 的维度
        self.project_embed = nn.Linear(PROTTRAN_EMBED_DIM, alphafold2.dim)
        # 加载 ProtBERT 模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
        self.model = AutoModel.from_pretrained('Rostlab/prot_bert')

    def forward(self, seq, msa, msa_mask = None, **kwargs):
        """
        前向传播

        参数:
            seq: 氨基酸序列 (batch, seq_len)
            msa: 多序列比对 (batch, num_msa, seq_len)
            msa_mask: MSA 掩码 (batch, num_msa, seq_len)
            **kwargs: 传递给 alphafold2 的其他参数

        返回:
            AlphaFold2 的输出
        """
        device = seq.device
        num_msa = msa.shape[1]
        # 展平 MSA 以便批处理
        msa_flat = rearrange(msa, 'b m n -> (b m) n')

        # 使用 ProtTrans 获取嵌入
        seq_embed = get_prottran_embedd(seq, self.model, self.tokenizer, device = device)
        msa_embed = get_prottran_embedd(msa_flat, self.model, self.tokenizer, device = device)

        # 投影到目标维度
        seq_embed, msa_embed = map(self.project_embed, (seq_embed, msa_embed))
        # 恢复 MSA 的形状
        msa_embed = rearrange(msa_embed, '(b m) n d -> b m n d', m = num_msa)

        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, msa_mask = msa_mask, **kwargs)

class MSAEmbedWrapper(nn.Module):
    """
    ESM MSA Transformer 嵌入封装器

    MSA Transformer 是专门为多序列比对设计的 Transformer 模型，
    它使用行绑定注意力（row-tied attention）来高效处理 MSA。
    这个模型特别适合利用进化信息来改进结构预测。

    参数:
        alphafold2: AlphaFold2 模型实例

    用法:
        model = Alphafold2(...)
        wrapped_model = MSAEmbedWrapper(alphafold2=model)
        output = wrapped_model(seq, msa, ...)
    """
    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2

        # 从 torch.hub 加载 ESM MSA Transformer 模型
        model, alphabet = torch.hub.load(*MSA_MODEL_PATH)
        batch_converter = alphabet.get_batch_converter()

        self.model = model
        self.batch_converter = batch_converter
        # 投影层（如果维度匹配则使用恒等映射）
        self.project_embed = nn.Linear(MSA_EMBED_DIM, alphafold2.dim) if MSA_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa, msa_mask = None, **kwargs):
        """
        前向传播

        参数:
            seq: 氨基酸序列 (batch, seq_len)
            msa: 多序列比对 (batch, num_msa, seq_len)
            msa_mask: MSA 掩码 (batch, num_msa, seq_len)
            **kwargs: 传递给 alphafold2 的其他参数

        返回:
            AlphaFold2 的输出
        """
        assert seq.shape[-1] == msa.shape[-1], 'sequence and msa must have the same length if you wish to use MSA transformer embeddings'
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        # 将序列和 MSA 拼接在一起
        seq_and_msa = torch.cat((seq.unsqueeze(1), msa), dim = 1)

        if exists(msa_mask):
            # in the event that there are rows in the MSA that are completely padding
            # 如果 MSA 中有完全是填充的行，需要特殊处理
            # process each batch element individually, so that padding isn't processed
            # with row-tied attention
            # 单独处理每个批次元素，避免行绑定注意力处理填充行

            # 计算每个批次元素有多少有效的 MSA 行
            num_msa = msa_mask.any(dim = -1).sum(dim = -1).tolist()
            seq_and_msa_list = seq_and_msa.unbind(dim = 0)
            num_rows = seq_and_msa.shape[1]

            embeds = []
            for num, batch_el in zip(num_msa, seq_and_msa_list):
                batch_el = rearrange(batch_el, '... -> () ...')
                # 只保留有效的行
                batch_el = batch_el[:, :num]
                embed = get_msa_embedd(batch_el, model, batch_converter, device = device)
                # 填充回原始大小
                embed = F.pad(embed, (0, 0, 0, 0, 0, num_rows - num), value = 0.)
                embeds.append(embed)

            embeds = torch.cat(embeds, dim = 0)
        else:
            # 如果没有掩码，直接处理整个 MSA
            embeds = get_msa_embedd(seq_and_msa, model, batch_converter, device = device)

        # 投影到目标维度
        embeds = self.project_embed(embeds)
        # 分离序列嵌入和 MSA 嵌入
        seq_embed, msa_embed = embeds[:, 0], embeds[:, 1:]

        return self.alphafold2(seq, msa, seq_embed = seq_embed, msa_embed = msa_embed, msa_mask = msa_mask, **kwargs)

class ESMEmbedWrapper(nn.Module):
    """
    ESM-1b 单序列嵌入封装器

    ESM-1b 是 Facebook 开发的大规模蛋白质语言模型（650M 参数），
    在数百万蛋白质序列上预训练。它为每个残基生成上下文相关的嵌入，
    这些嵌入捕获了氨基酸在序列中的语义信息。

    与 MSA Transformer 不同，ESM-1b 处理单个序列，因此对每条 MSA 序列
    分别生成嵌入。

    参数:
        alphafold2: AlphaFold2 模型实例

    用法:
        model = Alphafold2(...)
        wrapped_model = ESMEmbedWrapper(alphafold2=model)
        output = wrapped_model(seq, msa, ...)
    """
    def __init__(self, *, alphafold2):
        super().__init__()
        self.alphafold2 = alphafold2

        # 从 torch.hub 加载 ESM-1b 模型
        model, alphabet = torch.hub.load(*ESM_MODEL_PATH)
        batch_converter = alphabet.get_batch_converter()

        self.model = model
        self.batch_converter = batch_converter
        # 投影层（如果维度匹配则使用恒等映射）
        self.project_embed = nn.Linear(ESM_EMBED_DIM, alphafold2.dim) if ESM_EMBED_DIM != alphafold2.dim else nn.Identity()

    def forward(self, seq, msa=None, **kwargs):
        """
        前向传播

        参数:
            seq: 氨基酸序列 (batch, seq_len)
            msa: 多序列比对 (batch, num_msa, seq_len)，可选
            **kwargs: 传递给 alphafold2 的其他参数

        返回:
            AlphaFold2 的输出
        """
        model, batch_converter, device = self.model, self.batch_converter, seq.device

        # 获取序列嵌入
        seq_embeds = get_esm_embedd(seq, model, batch_converter, device = device)
        seq_embeds = self.project_embed(seq_embeds)

        if msa is not None:
            # 如果提供了 MSA，分别为每条 MSA 序列生成嵌入
            flat_msa = rearrange(msa, 'b m n -> (b m) n')
            msa_embeds = get_esm_embedd(flat_msa, model, batch_converter, device = device)
            msa_embeds = rearrange(msa_embeds, '(b m) n d -> b m n d')
            msa_embeds = self.project_embed(msa_embeds)
        else:
            msa_embeds = None

        return self.alphafold2(seq, msa, seq_embed = seq_embeds, msa_embed = msa_embeds, **kwargs)
