"""检查数据模块、模型模块和 GPU 前向计算。

在项目根目录运行：
    .\.venv\Scripts\python.exe inspect_data.py --lang zh
    .\.venv\Scripts\python.exe inspect_data.py --lang en
"""

import argparse
from pathlib import Path

import torch

from cbow.data import build_samples, build_vocab, create_dataloader, read_corpus
from cbow.model import CBOW


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 CBOW 数据与 GPU 前向计算")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    args = parser.parse_args()

    # 根据脚本位置定位语料，不依赖终端的当前工作目录。
    project_dir = Path(__file__).resolve().parent
    corpus_path = project_dir / "data" / f"{args.lang}.txt"

    sentences = read_corpus(corpus_path)
    word_to_id, id_to_word = build_vocab(sentences)
    samples = build_samples(sentences, word_to_id, window_size=2)
    data_loader = create_dataloader(samples, batch_size=128, shuffle=False)

    print("语料语言：", args.lang)
    print("文本行数：", len(sentences))
    print("词表大小：", len(id_to_word))
    print("训练样本数量：", len(samples))

    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境无法使用 CUDA，请检查解释器及 PyTorch 安装。")

    device = torch.device("cuda")
    model = CBOW(vocab_size=len(id_to_word), embedding_dim=100).to(device)
    model.eval()

    batch_contexts, batch_targets = next(iter(data_loader))
    batch_contexts = batch_contexts.to(device)
    batch_targets = batch_targets.to(device)

    # 这里只做前向检查，不计算梯度，也不更新模型参数。
    with torch.no_grad():
        logits = model(batch_contexts)

    print("显卡：", torch.cuda.get_device_name(0))
    print("输入形状：", batch_contexts.shape)
    print("输出形状：", logits.shape)
    print("模型设备：", next(model.parameters()).device)
    print("输入设备：", batch_contexts.device)
    print("目标设备：", batch_targets.device)
    print("输出设备：", logits.device)


if __name__ == "__main__":
    main()

