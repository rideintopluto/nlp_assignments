"""CBOW：输入词向量采用标准差 0.01 的正态初始化。

放在项目的 cbow/model.py。用于与原来的默认初始化做对照实验。
模型结构、参数名称及 forward 接口与原版一致。
"""

import torch
from torch import nn


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

        # 默认嵌入初始化为 N(0, 1)，整体缩小 100 倍后标准差为 0.01。
        # 不额外抽取随机数，保持与原版相同的初始方向及输出层初始化。
        with torch.no_grad():
            self.embedding.weight.mul_(0.01)

    def forward(self, context_ids):
        word_vectors = self.embedding(context_ids)
        context_vector = word_vectors.mean(dim=1)
        logits = self.output(context_vector)

        return logits
