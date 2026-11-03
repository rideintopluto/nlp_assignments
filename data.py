"""CBOW 数据处理模块：读取语料、建立词表、构造样本和按批次加载数据。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def read_corpus(file_path: str | Path) -> list[list[str]]:
    """按行读取已分词语料；保留词和标点，跳过空行。"""
    sentences = []

    with Path(file_path).open("r", encoding="utf-8-sig") as file:
        for line in file:
            words = line.strip().split()
            if words:
                sentences.append(words)

    if not sentences:
        raise ValueError(f"语料没有非空文本行：{file_path}")

    return sentences


def build_vocab(
    sentences: list[list[str]],
) -> tuple[dict[str, int], list[str]]:
    """按首次出现顺序编号，返回 word_to_id 和 id_to_word。"""
    word_to_id = {}
    id_to_word = []

    for sentence in sentences:
        for word in sentence:
            if word not in word_to_id:
                new_id = len(id_to_word)
                word_to_id[word] = new_id
                id_to_word.append(word)

    return word_to_id, id_to_word


def build_samples(
    sentences: list[list[str]],
    word_to_id: dict[str, int],
    window_size: int = 2,
) -> list[tuple[list[int], int]]:
    """构造 (上下文编号列表, 目标编号)，window_size 表示单侧词数。"""
    if window_size < 1:
        raise ValueError("window_size 必须是大于等于 1 的整数。")

    samples = []

    # 每行独立处理，窗口不会跨行。
    for sentence in sentences:
        sentence_ids = []
        for word in sentence:
            sentence_ids.append(word_to_id[word])

        # 只使用左右上下文完整的位置。
        for i in range(window_size, len(sentence_ids) - window_size):
            left_context = sentence_ids[i - window_size:i]
            right_context = sentence_ids[i + 1:i + window_size + 1]
            context_ids = left_context + right_context
            target_id = sentence_ids[i]
            samples.append((context_ids, target_id))

    return samples


def create_dataloader(
    samples: list[tuple[list[int], int]],
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    """将样本转换成 CPU 整数张量，返回按批次读取的 DataLoader。"""
    if not samples:
        raise ValueError("训练样本为空，请检查语料长度和 window_size。")

    context_list = []
    target_list = []

    for context_ids, target_id in samples:
        context_list.append(context_ids)
        target_list.append(target_id)

    contexts_tensor = torch.tensor(context_list, dtype=torch.long)
    targets_tensor = torch.tensor(target_list, dtype=torch.long)
    dataset = TensorDataset(contexts_tensor, targets_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

