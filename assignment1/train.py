"""Train and evaluate a CNN on the cat/dog dataset."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from catdog.data import CatDogDataset, build_transforms, find_split_dir
from catdog.engine import evaluate, train_one_epoch
from catdog.model import CatDogCNN
from catdog.utils import (
    confusion_matrix,
    per_class_accuracy,
    plot_confusion_matrix,
    plot_history,
    save_history,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 CNN 完成猫狗图片二分类")
    parser.add_argument("--data-dir", type=Path, required=True, help="包含 train 和 val 的数据目录")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="训练输出目录")
    parser.add_argument("--epochs", type=int, default=20, help="最大训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--image-size", type=int, default=128, help="输入图片边长")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 正则化系数")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout 概率")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值；0 表示关闭早停")
    parser.add_argument("--num-workers", type=int, default=0, help="数据读取进程数，Windows 建议从 0 开始")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--check-only", action="store_true", help="只检查数据集，不训练")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs 必须大于等于 1")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于等于 1")
    if args.image_size < 32:
        parser.error("--image-size 必须大于等于 32")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout 必须位于 [0, 1) 区间")
    return args


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA。请运行 check_environment.py。")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = find_split_dir(args.data_dir, "train")
    val_dir = find_split_dir(args.data_dir, "val")
    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset = CatDogDataset(train_dir, train_transform)
    val_dataset = CatDogDataset(val_dir, val_transform)

    print(f"训练目录：{train_dir.resolve()}")
    print(f"验证目录：{val_dir.resolve()}")
    print(f"训练集：{len(train_dataset)} 张，{train_dataset.class_counts}")
    print(f"验证集：{len(val_dataset)} 张，{val_dataset.class_counts}")
    if args.check_only:
        print("数据集检查通过。")
        return

    device = select_device(args.device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU"
    print(f"训练设备：{device} ({device_name})")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = CatDogCNN(dropout=args.dropout).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history: list[dict[str, float | int]] = []
    checkpoint_path = args.output_dir / "best_model.pth"
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics.loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "learning_rate": learning_rate,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics.loss:.4f}, acc {train_metrics.accuracy:.2%} | "
            f"val loss {val_metrics.loss:.4f}, acc {val_metrics.accuracy:.2%} | "
            f"lr {learning_rate:.2e}"
        )

        improved = val_metrics.accuracy > best_accuracy or (
            val_metrics.accuracy == best_accuracy and val_metrics.loss < best_loss
        )
        if improved:
            best_accuracy = val_metrics.accuracy
            best_loss = val_metrics.loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "class_names": ["cat", "dog"],
                    "best_epoch": epoch,
                    "train_loss": train_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"验证集连续 {args.patience} 轮未改善，提前停止训练。")
                break

    elapsed_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(model, val_loader, criterion, device)
    matrix = confusion_matrix(final_metrics.labels, final_metrics.predictions)
    class_accuracy = per_class_accuracy(matrix)

    save_history(history, args.output_dir / "history.csv")
    plot_history(history, args.output_dir / "training_curves.png")
    plot_confusion_matrix(matrix, args.output_dir / "confusion_matrix.png")

    results = {
        "task": "cat_dog_binary_classification",
        "model": "CatDogCNN",
        "pretrained": False,
        "classes": ["cat", "dog"],
        "dataset": {
            "train_total": len(train_dataset),
            "train_counts": train_dataset.class_counts,
            "val_total": len(val_dataset),
            "val_counts": val_dataset.class_counts,
        },
        "hyperparameters": {
            "requested_epochs": args.epochs,
            "completed_epochs": len(history),
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "early_stopping_patience": args.patience,
            "seed": args.seed,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "device": str(device),
            "device_name": device_name,
            "cuda_version": torch.version.cuda,
        },
        "model_summary": {
            "trainable_parameters": trainable_parameters,
            "feature_channels": [32, 64, 128, 256],
        },
        "best_epoch": best_epoch,
        "best_train_loss": checkpoint["train_loss"],
        "best_train_accuracy": checkpoint["train_accuracy"],
        "best_val_loss": final_metrics.loss,
        "validation_accuracy": final_metrics.accuracy,
        "per_class_accuracy": class_accuracy,
        "confusion_matrix": matrix,
        "training_seconds": elapsed_seconds,
        "files": {
            "checkpoint": str(checkpoint_path),
            "history": str(args.output_dir / "history.csv"),
            "training_curves": str(args.output_dir / "training_curves.png"),
            "confusion_matrix": str(args.output_dir / "confusion_matrix.png"),
        },
    }
    save_json(results, args.output_dir / "results.json")

    print("\n训练完成")
    print(f"最佳轮次：{best_epoch}")
    print(f"验证集总体准确率：{final_metrics.accuracy:.2%}")
    print(f"猫类别准确率：{class_accuracy['cat']:.2%}")
    print(f"狗类别准确率：{class_accuracy['dog']:.2%}")
    print(f"训练耗时：{elapsed_seconds:.1f} 秒")
    print(f"所有结果已保存到：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
