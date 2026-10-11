"""Predict whether a single image contains a cat or a dog."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from catdog.model import CatDogCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测单张图片的猫狗类别")
    parser.add_argument("--image", type=Path, required=True, help="待预测图片")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pth"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"找不到图片：{args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到模型权重：{args.checkpoint}，请先运行 train.py")

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.device == "cuda" and not use_cuda:
        raise RuntimeError("指定了 CUDA，但当前环境无法使用 CUDA。")
    device = torch.device("cuda" if use_cuda else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CatDogCNN(dropout=float(checkpoint.get("dropout", 0.4))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(checkpoint.get("image_size", 128))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    with Image.open(args.image) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    class_names = checkpoint.get("class_names", ["cat", "dog"])
    index = int(probabilities.argmax().item())
    labels_zh = {"cat": "猫", "dog": "狗"}
    class_name = class_names[index]

    print(f"预测类别：{labels_zh.get(class_name, class_name)}")
    print(f"置信度：{probabilities[index].item():.2%}")
    print(f"猫概率：{probabilities[0].item():.2%}")
    print(f"狗概率：{probabilities[1].item():.2%}")


if __name__ == "__main__":
    main()
