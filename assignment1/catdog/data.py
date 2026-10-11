"""Dataset discovery and image preprocessing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ("cat", "dog")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _infer_label(path: Path, split_dir: Path) -> int | None:
    """Infer a cat/dog label from class folders or the file name."""
    relative = path.relative_to(split_dir)
    directory_tokens = {part.lower() for part in relative.parts[:-1]}

    has_cat_dir = bool(directory_tokens & {"cat", "cats"})
    has_dog_dir = bool(directory_tokens & {"dog", "dogs"})
    if has_cat_dir != has_dog_dir:
        return 0 if has_cat_dir else 1

    filename_tokens = set(re.findall(r"[a-z]+", path.stem.lower()))
    has_cat_name = bool(filename_tokens & {"cat", "cats"})
    has_dog_name = bool(filename_tokens & {"dog", "dogs"})
    if has_cat_name != has_dog_name:
        return 0 if has_cat_name else 1
    return None


def find_split_dir(data_dir: Path, split: str) -> Path:
    """Resolve the training or validation directory."""
    aliases = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation", "test"),
    }
    for name in aliases[split]:
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    expected = ", ".join(str(data_dir / name) for name in aliases[split])
    raise FileNotFoundError(f"找不到 {split} 数据目录。检查以下路径之一是否存在：{expected}")


def discover_samples(split_dir: Path) -> list[tuple[Path, int]]:
    """Find supported images and assign labels."""
    image_paths = sorted(
        path
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"目录中没有可读取的图片：{split_dir}")

    samples: list[tuple[Path, int]] = []
    unknown: list[Path] = []
    for path in image_paths:
        label = _infer_label(path, split_dir)
        if label is None:
            unknown.append(path)
        else:
            samples.append((path, label))

    if unknown:
        examples = "\n".join(f"  - {path}" for path in unknown[:5])
        raise ValueError(
            "无法从部分图片的目录或文件名判断 cat/dog 类别。"
            "请使用 cat、dog 子目录，或在文件名中加入 cat、dog。示例：\n"
            f"{examples}"
        )

    counts = [sum(label == index for _, label in samples) for index in range(2)]
    if 0 in counts:
        raise ValueError(
            f"数据必须同时包含猫和狗图片，当前统计：cat={counts[0]}, dog={counts[1]}"
        )
    return samples


class CatDogDataset(Dataset):
    """A dataset that supports class folders and flat Kaggle-style names."""

    def __init__(self, split_dir: Path, transform: Callable | None = None) -> None:
        self.split_dir = Path(split_dir)
        self.samples = discover_samples(self.split_dir)
        self.transform = transform
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"读取图片失败：{path}") from exc
        return image, label

    @property
    def class_counts(self) -> dict[str, int]:
        return {
            class_name: sum(label == index for label in self.targets)
            for index, class_name in enumerate(CLASS_NAMES)
        }


def build_transforms(image_size: int):
    """Return training and validation preprocessing pipelines."""
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, val_transform
