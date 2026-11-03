r"""CBOW 项目的训练入口，放在项目根目录。

在项目根目录的 VSCode 终端运行：
    .\.venv\Scripts\python.exe train.py --lang zh
    .\.venv\Scripts\python.exe train.py --lang en

需要已有的 cbow/data.py、cbow/model.py 和 cbow/training.py。
语料路径为 data/zh.txt 或 data/en.txt；文件内容应已按空格分词。
正式训练使用 CUDA。每次运行创建独立输出目录，保存最终模型和训练记录。
"""

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 GPU 训练中文或英文 CBOW 模型")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="语料语言")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数，默认 10")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小，默认 128")
    parser.add_argument("--embedding-dim", type=int, default=100, help="词向量维度，默认 100")
    parser.add_argument("--window-size", type=int, default=2, help="单侧上下文词数，默认 2")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率，默认 0.001")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--log-interval", type=int, default=100, help="每多少批打印进度；0 表示仅打印每轮结果")
    args = parser.parse_args()

    for name in ("epochs", "batch_size", "embedding_dim", "window_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于等于 1。")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须是有限的正数。")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed 必须在 0 到 4294967295 之间。")
    if args.log_interval < 0:
        parser.error("--log-interval 不能小于 0。")
    return args


def write_json(path: Path, content: dict) -> None:
    """使用 UTF-8 保存 JSON，保留可读的中文。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def main() -> None:
    args = parse_args()

    # 延后导入，让 --help 即使在未安装 PyTorch 时也可以使用。
    import torch

    from cbow.data import build_samples, build_vocab, create_dataloader, read_corpus
    from cbow.model import CBOW
    from cbow.training import train_model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 Python 环境无法使用 CUDA。请使用之前 inspect_data.py 检查通过的解释器。"
        )
    device = torch.device("cuda")

    # 以本文件所在位置定位项目，不依赖终端当前所在目录。
    project_dir = Path(__file__).resolve().parent
    corpus_path = project_dir / "data" / f"{args.lang}.txt"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"找不到语料文件：{corpus_path}")

    # 在创建模型和打乱样本前设置种子，便于比较实验。
    # 不保证跨 PyTorch 版本、不同显卡的结果逐位一致。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sentences = read_corpus(corpus_path)
    word_to_id, id_to_word = build_vocab(sentences)
    samples = build_samples(sentences, word_to_id, window_size=args.window_size)
    data_loader = create_dataloader(samples, batch_size=args.batch_size, shuffle=True)
    model = CBOW(vocab_size=len(id_to_word), embedding_dim=args.embedding_dim)

    # 中文、英文及不同训练次数分别保存，避免覆盖之前的模型。
    started_at = datetime.now().astimezone()
    run_id = started_at.strftime("%Y%m%d_%H%M%S_%f")
    output_dir = project_dir / "outputs" / args.lang / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "model_type": "CBOW",
        "language": args.lang,
        "corpus_path": str(corpus_path),
        "sentence_count": len(sentences),
        "vocab_size": len(id_to_word),
        "sample_count": len(samples),
        "embedding_dim": args.embedding_dim,
        "window_size": args.window_size,
        "context_size": 2 * args.window_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "seed": args.seed,
        "log_interval": args.log_interval,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "pytorch_version": str(torch.__version__),
        "started_at": started_at.isoformat(timespec="seconds"),
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "vocab.json", {"word_to_id": word_to_id, "id_to_word": id_to_word})

    print(f"语料语言：{args.lang}")
    print(f"文本行数：{len(sentences)} | 词表大小：{len(id_to_word)} | 样本数：{len(samples)}")
    print(f"轮数：{args.epochs} | 批大小：{args.batch_size} | 学习率：{args.lr}")
    print(f"输出目录：{output_dir}", flush=True)

    loss_history = train_model(
        model=model,
        train_loader=data_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        log_interval=args.log_interval,
    )

    # 仅保存模型参数、普通字典和列表，便于后续重新创建 CBOW 后加载。
    # 权重转成 CPU 张量保存，之后可加载到 CPU 或 GPU。
    # 这是最终模型文件，未包含优化器状态，不用于精确断点续训。
    checkpoint = {
        "format_version": 1,
        "model_type": "CBOW",
        "model_config": {
            "vocab_size": len(id_to_word),
            "embedding_dim": args.embedding_dim,
        },
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "training_config": config,
        "completed_epochs": len(loss_history),
        "loss_history": loss_history,
    }
    torch.save(checkpoint, output_dir / "model.pt")

    with (output_dir / "loss_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss"])
        writer.writerows(enumerate(loss_history, start=1))

    print("训练完成。最终模型、词表、配置和每轮训练损失已保存。")
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()
