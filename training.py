"""CBOW 训练模块：单轮训练和多轮训练。

放置位置：cbow/training.py，与 data.py、model.py 同级。
由项目入口导入 train_model，并传入已有的模型和 DataLoader。
本模块不读取语料、不创建词表，也不重新定义模型。
默认要求 CUDA；只有显式传入 device="cpu" 才会使用 CPU。
"""

import math
from time import perf_counter

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: Optimizer,
    device: str | torch.device,
    log_interval: int = 100,
) -> float:
    """训练一轮，返回按实际样本数加权的平均损失。

    调用前需将模型放到 device，再创建 optimizer；train_model 已完成这些操作。
    criterion 应使用默认的 reduction="mean"，且不设置类别权重或忽略目标。
    每批数据：(contexts, targets)，分别为 [B, 4] 和 [B] 的整数张量。
    model(contexts) 应返回 [B, vocab_size] 的原始分数（logits）。
    log_interval=0 时关闭批次日志。
    """
    if log_interval < 0:
        raise ValueError("log_interval 不能小于 0。")

    model.train()
    total_loss = 0.0
    total_samples = 0

    for step, (contexts, targets) in enumerate(train_loader, start=1):
        # data.py 保留 CPU 数据；训练时仅将当前批次送入指定设备。
        contexts = contexts.to(device)
        targets = targets.to(device)

        # 清空上一批的梯度，避免不同批次的梯度意外累积。
        optimizer.zero_grad(set_to_none=True)

        logits = model(contexts)
        # CrossEntropyLoss 直接接收 logits，不需要先调用 softmax。
        loss = criterion(logits, targets)
        batch_loss = loss.item()
        if not math.isfinite(batch_loss):
            raise FloatingPointError(
                f"第 {step} 批的损失为 {batch_loss}，已停止训练，请检查数据和学习率。"
            )

        # 反向传播计算梯度，然后由优化器更新模型参数。
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += batch_loss * batch_size
        total_samples += batch_size

        if log_interval and step % log_interval == 0:
            print(
                f"  批次 {step}/{len(train_loader)} | "
                f"当前批损失 {batch_loss:.4f} | "
                f"本轮累计平均损失 {total_loss / total_samples:.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError("DataLoader 没有提供训练样本，请检查数据和批次设置。")

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str | torch.device = "cuda",
    log_interval: int = 100,
) -> list[float]:
    """训练指定轮数，原地更新 model，返回每轮平均训练损失。

    默认配置是起步配置，可由调用方修改，不代表作业指定参数。
    使用 Adam 和交叉熵损失；所有轮次共享同一个优化器。
    每次调用都会创建新的优化器，因此本函数不负责断点续训。
    返回值只记录训练损失，不代表独立验证集或测试集表现。
    模型及词表的保存由项目入口负责。
    """
    if epochs < 1:
        raise ValueError("epochs 必须大于等于 1。")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必须是有限的正数。")
    if log_interval < 0:
        raise ValueError("log_interval 不能小于 0。")
    if len(train_loader) == 0:
        raise ValueError("DataLoader 为空，请先检查数据模块。")

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "当前 Python 环境无法使用 CUDA。请使用之前 GPU 检查通过的解释器，"
                "并检查 PyTorch 安装。本次训练不会自动改用 CPU。"
            )
        print(f"训练设备：{device} | {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"训练设备：{device}", flush=True)

    # 先移动模型，再创建优化器，保证优化器使用正确设备上的参数。
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(1, epochs + 1):
        print(f"开始第 {epoch}/{epochs} 轮训练", flush=True)
        start_time = perf_counter()
        average_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=log_interval,
        )
        # 等待 GPU 完成本轮计算，使本轮耗时包含最后一次参数更新。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start_time
        loss_history.append(average_loss)
        print(
            f"第 {epoch}/{epochs} 轮完成 | "
            f"平均训练损失 {average_loss:.4f} | 耗时 {elapsed:.1f} 秒",
            flush=True,
        )

    return loss_history
