"""CBOW 词向量评估：读取检查点、查询近邻和导出词向量。

只读取输入嵌入层 embedding.weight；不训练、不更新任何参数。
余弦相似度用于查看词向量的几何关系，不是准确率或置信度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


@dataclass
class WordVectors:
    checkpoint_path: Path
    language: str
    epochs: int
    dim: int
    word_to_id: dict[str, int]
    id_to_word: list[str]
    vectors: torch.Tensor
    normalized_vectors: torch.Tensor
    nonzero_rows: torch.Tensor
    training_config: dict[str, Any]
    loss_history: list[float]

    @property
    def run_id(self) -> str:
        return self.checkpoint_path.parent.name

    @property
    def label(self) -> str:
        return f"epoch={self.epochs}, dim={self.dim}, run={self.run_id}"


def load_word_vectors(
    checkpoint_path: str | Path, device: str | torch.device = "cuda"
) -> WordVectors:
    """读取当前 train.py 保存的 model.pt，并检查词表与权重是否对应。"""
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "CBOW":
        raise ValueError(f"不是当前 CBOW 项目的检查点：{path}")

    state = checkpoint.get("model_state_dict", {})
    vectors = state.get("embedding.weight")
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError(f"找不到二维 embedding.weight：{path}")
    if not vectors.is_floating_point() or not torch.isfinite(vectors).all().item():
        raise ValueError(f"词向量必须为有限浮点数：{path}")

    words = checkpoint.get("id_to_word")
    word_to_id = checkpoint.get("word_to_id")
    if not isinstance(words, list) or not words or not isinstance(word_to_id, dict):
        raise ValueError(f"检查点缺少词表：{path}")
    if not all(isinstance(word, str) and word and not any(c.isspace() for c in word) for word in words):
        raise ValueError(f"词表中存在空词或含空白符的词：{path}")
    if len(set(words)) != len(words) or word_to_id != {word: i for i, word in enumerate(words)}:
        raise ValueError(f"word_to_id 与 id_to_word 不一致：{path}")

    config = checkpoint.get("training_config", {})
    model_config = checkpoint.get("model_config", {})
    language = config.get("language")
    epochs = checkpoint.get("completed_epochs")
    dim = vectors.shape[1]
    if language not in ("zh", "en") or not isinstance(epochs, int) or epochs < 1 or dim < 1:
        raise ValueError(f"缺少有效的语言、训练轮数或维度信息：{path}")
    if vectors.shape[0] != len(words) or model_config != {"vocab_size": len(words), "embedding_dim": dim}:
        raise ValueError(f"词表、模型配置与词向量形状不一致：{path}")
    if config.get("epochs") != epochs or config.get("embedding_dim") != dim:
        raise ValueError(f"训练配置与实际完成轮数或维度不一致：{path}")
    losses = checkpoint.get("loss_history", [])
    if not isinstance(losses, list) or len(losses) != epochs:
        raise ValueError(f"损失记录长度与训练轮数不一致：{path}")

    vectors = vectors.detach().to(device=device, dtype=torch.float32)
    nonzero = torch.linalg.vector_norm(vectors, dim=1) > 1e-12
    return WordVectors(
        checkpoint_path=path,
        language=language,
        epochs=epochs,
        dim=dim,
        word_to_id=word_to_id,
        id_to_word=words,
        vectors=vectors,
        normalized_vectors=F.normalize(vectors, p=2, dim=1, eps=1e-12),
        nonzero_rows=nonzero,
        training_config=config,
        loss_history=losses,
    )


@torch.inference_mode()
def most_similar(model: WordVectors, query: str, top_k: int = 10) -> list[tuple[str, float]]:
    """返回余弦相似度最高的其他词；排除查询词自身和零向量。

    保留其他所有词，包括停用词、标点和数字，不根据结果好坏筛选。
    同分时按原词表顺序排列。不存在的词抛出 KeyError，不替换成别的词。
    """
    if top_k < 1:
        raise ValueError("top_k 必须大于等于 1。")
    if query not in model.word_to_id:
        raise KeyError(query)
    query_id = model.word_to_id[query]
    if not model.nonzero_rows[query_id].item():
        raise ValueError(f"查询词 {query!r} 的向量为零，无法定义余弦相似度。")

    # 两个单位向量的点积就是余弦相似度。
    scores = model.normalized_vectors @ model.normalized_vectors[query_id]
    scores = scores.clamp(-1.0, 1.0)
    scores[~model.nonzero_rows] = -torch.inf
    scores[query_id] = -torch.inf
    count = min(top_k, int(model.nonzero_rows.sum().item()) - 1)
    if count < 1:
        return []
    indices = torch.argsort(scores, descending=True, stable=True)[:count]
    ids = indices.cpu().tolist()
    similarities = scores[indices].cpu().tolist()
    return [(model.id_to_word[i], float(score)) for i, score in zip(ids, similarities)]


def export_word_vectors(model: WordVectors, output_path: str | Path) -> None:
    """导出 word2vec 文本格式：首行词数与维度，之后每行一个词及其向量。

    导出原始输入嵌入，不是为了计算余弦相似度而归一化后的副本。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = model.vectors.cpu()
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(f"{len(model.id_to_word)} {model.dim}\n")
        for word, vector in zip(model.id_to_word, vectors):
            numbers = " ".join(f"{value:.9g}" for value in vector.tolist())
            file.write(f"{word} {numbers}\n")
