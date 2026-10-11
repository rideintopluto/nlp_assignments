"""Print the local PyTorch and CUDA environment."""

from __future__ import annotations

import platform
import sys

import torch
import torchvision


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"操作系统: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA 版本: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 显存: {total_memory:.1f} GB")
        test = torch.randn(512, 512, device="cuda")
        value = (test @ test).mean().item()
        print(f"CUDA 运算测试: 通过 ({value:.4f})")
    else:
        print("未检测到可用 CUDA。训练仍可使用 CPU，但速度会较慢。")


if __name__ == "__main__":
    main()
